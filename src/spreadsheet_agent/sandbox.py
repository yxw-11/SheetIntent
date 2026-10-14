"""Private oracle-side scoring; generated code runs with input-only mounts."""
from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path

import openpyxl

from spreadsheetbench_lite.data import discover_test_cases
from spreadsheetbench_lite.evaluate_v1 import compare_workbooks
from spreadsheet_code_agent_v0.preflight import run_preflight
from spreadsheet_code_agent_v0.runtime import (
    AgentExecResult, classify_runtime_error, preflight_diagnostic, runtime_diagnostic, verify_output,
)
from spreadsheet_code_agent_v0.run_agent_v0 import _first_repair_trigger

from .io import WORK as HERE, atomic


class InfrastructureError(RuntimeError):
    pass


@contextmanager
def network_filter():
    """Export a child-only seccomp filter; never change the parent process policy."""
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise InfrastructureError("libseccomp.so.2 unavailable; refusing unrestricted networking") from exc
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW, with explicit network denials below.
    if not ctx:
        raise InfrastructureError("Cannot allocate seccomp filter")
    try:
        # Block socket creation, inherited-socket I/O, and io_uring network bypasses.
        names = ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                 "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg", "shutdown",
                 "io_uring_setup", "io_uring_enter", "io_uring_register")
        for name in names:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                raise InfrastructureError(f"Required seccomp syscall unavailable: {name}")
            if lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise InfrastructureError(f"Cannot block network syscall: {name}")
        with tempfile.TemporaryFile(dir=HERE / "scratch") as rules:
            if lib.seccomp_export_bpf(ctx, rules.fileno()) != 0:
                raise InfrastructureError("Cannot export seccomp filter")
            rules.seek(0)
            yield rules.fileno()
    finally:
        lib.seccomp_release(ctx)


def sandbox_command(work, seccomp_fd=None):
    binary = shutil.which("bwrap")
    if not binary:
        raise InfrastructureError("bubblewrap unavailable; execution will NOT fall back to unrestricted Python")
    prefix = Path(sys.prefix).resolve()
    cmd = [binary, "--die-with-parent", "--unshare-all", "--new-session", "--clearenv"]
    # Spartan disables network namespaces. Share that namespace only with kernel-enforced
    # network denial; retain mount, PID, IPC and user namespace isolation.
    if seccomp_fd is not None:
        cmd += ["--share-net", "--seccomp", str(seccomp_fd)]
    for path in ("/usr", "/lib", "/lib64", str(prefix)):
        if Path(path).exists():
            cmd += ["--ro-bind", path, path]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(work), "/work", "--chdir", "/work",
            "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PATH", str(prefix / "bin") + ":/usr/bin", str(prefix / "bin/python"), "-I", "runner.py"]
    return cmd


RUNNER = '''import resource
from pathlib import Path
from openpyxl import load_workbook
resource.setrlimit(resource.RLIMIT_FSIZE, (256*1024*1024, 256*1024*1024))
ns = {"__name__": "__generated_transform__", "__file__": "/work/solution.py"}
exec(compile(Path("solution.py").read_text(), "solution.py", "exec"), ns)
wb = load_workbook("input.xlsx")
ns["transform"](wb)
wb.save("output.xlsx")
'''


def execute(code, source, destination, timeout, answer_position):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (HERE / "scratch").mkdir(parents=True, exist_ok=True)
    if destination.resolve() == Path(source).resolve():
        raise InfrastructureError("Input and output must be different files")
    # Repair retries reuse the destination; stale files must never count as success.
    destination.unlink(missing_ok=True)
    # Only one input and this candidate's output are visible. No project or answer mounts.
    with tempfile.TemporaryDirectory(prefix="candidate_", dir=HERE / "scratch") as temp:
        work = Path(temp)
        shutil.copyfile(source, work / "input.xlsx")
        (work / "solution.py").write_text(code)
        (work / "runner.py").write_text(RUNNER)
        try:
            with network_filter() as fd:
                proc = subprocess.run(sandbox_command(work, fd), pass_fds=(fd,),
                                      capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return AgentExecResult(False, -1, "", "Program timed out", True, "timeout")
        if "bwrap:" in proc.stderr:
            raise InfrastructureError(proc.stderr.strip())
        if proc.returncode < 0:
            raise InfrastructureError(f"Worker killed by signal {-proc.returncode}; not a model reward=0")
        output = work / "output.xlsx"
        if output.is_file() and not output.is_symlink():
            shutil.copyfile(output, destination)
        verification = verify_output(Path(source), destination, answer_position) if destination.exists() else {}
        return AgentExecResult(proc.returncode == 0 and bool(verification.get("openable")),
                               proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:], False,
                               classify_runtime_error(proc.stderr), verification)


def score(task, dataset, code, context, directory, timeout=180):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (HERE / "scratch").mkdir(parents=True, exist_ok=True)
    cases = discover_test_cases(Path(dataset), directory, task)
    if not cases or any(c.answer_path is None for c in cases):
        raise InfrastructureError(f"Missing input/oracle variants for {task.id}")
    # Validate references independently; an invalid oracle is not a model failure.
    for case in cases:
        valid, message = compare_workbooks(case.answer_path, case.answer_path, task.answer_position)
        if not valid:
            raise InfrastructureError(f"Invalid oracle/answer_position for {task.id}:{case.index}: {message}")
    started = time.monotonic()
    preflight = run_preflight(code, context).to_dict()
    diagnostic = preflight_diagnostic(preflight) if not preflight["ok"] else None
    rows = []
    for case in cases:
        if not preflight["ok"]:
            result = AgentExecResult(False, 1, "", json.dumps(preflight["errors"]), False, "preflight_error")
        else:
            result = execute(code, case.input_path, case.output_path, timeout, task.answer_position)
        ok, message = compare_workbooks(case.answer_path, case.output_path, task.answer_position) if result.ok else (False, "execution failed")
        rows.append({"variant": case.index, **asdict(result), "benchmark_exact": bool(ok), "evaluator_message": message})
        if diagnostic is None:
            if not result.ok:
                diagnostic = runtime_diagnostic(result, code)
            else:
                trigger = _first_repair_trigger({"cases": [asdict(result)], "_runtime_objects": [result]}, code)
                if trigger:
                    diagnostic = trigger["diagnostic"]
    record = {"task_id": task.id, "benchmark_exact": bool(rows) and all(r["benchmark_exact"] for r in rows),
              "exec_ok": all(r["ok"] for r in rows), "cases": rows, "preflight": preflight,
              "diagnostic": diagnostic, "seconds": time.monotonic() - started,
              "semantic_checked": {"scope": "artifact_integrity_only; not semantic correctness", "results": [r["verification"] for r in rows]},
              "workbook_executions": len(cases) if preflight["ok"] else 0}
    atomic(directory / "score.json", record)
    return record

