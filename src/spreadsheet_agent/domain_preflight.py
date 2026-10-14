"""Opt-in sheet-lifecycle check; frozen training and historical harness stay unchanged."""
from __future__ import annotations

import ast
from dataclasses import replace

from spreadsheet_code_agent_v0.preflight import run_preflight as original_preflight


def _string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _workbook(node):
    return isinstance(node, ast.Name) and node.id == "wb"


def run_preflight(code, inspection=None):
    result = original_preflight(code, inspection)
    if not any(e.category == "sheet_error" for e in result.errors):
        return result
    tree = ast.parse(code)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return node
        return tree

    def guarded(node, title):
        branch = node
        while branch in parents:
            parent = parents[branch]
            if isinstance(parent, (ast.If, ast.IfExp)):
                test = parent.test
                if (isinstance(test, ast.Compare) and len(test.ops) == 1
                        and _string(test.left) == title and len(test.comparators) == 1):
                    names = test.comparators[0]
                    if isinstance(names, ast.Attribute) and names.attr == "sheetnames" and _workbook(names.value):
                        positive = branch in parent.body if isinstance(parent, ast.If) else branch is parent.body
                        negative = branch in parent.orelse if isinstance(parent, ast.If) else branch is parent.orelse
                        if (positive and isinstance(test.ops[0], ast.In)) or (negative and isinstance(test.ops[0], ast.NotIn)):
                            return True
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                break
            branch = parent
        return False

    mutations = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and _workbook(node.func.value) and node.func.attr == "create_sheet"):
            title = next((kw.value for kw in node.keywords if kw.arg == "title"), node.args[0] if node.args else None)
            if _string(title) is not None:
                mutations.append((node, _string(title)))
        if isinstance(node, ast.Assign) and _string(node.value) is not None:
            if any(isinstance(t, ast.Attribute) and t.attr == "title" for t in node.targets):
                mutations.append((node, _string(node.value)))

    safe = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript) and _workbook(node.value)):
            continue
        title = _string(node.slice)
        if title is None:
            continue
        prior_mutation = any(name == title and scope(mutation) is scope(node)
                             and (mutation.lineno, mutation.col_offset) < (node.lineno, node.col_offset)
                             for mutation, name in mutations)
        if guarded(node, title) or prior_mutation:
            safe.add((node.lineno, f"Workbook has no sheet named {title.strip()!r}."))

    # Creation in a conditional branch is not a proof of existence. Execute it
    # and retain runtime diagnostics instead of rejecting based on the input alone.
    kept = []
    for issue in result.errors:
        if issue.category == "sheet_error" and (issue.line, issue.message) in safe:
            result.warnings.append(replace(issue, severity="warning", category="dynamic_sheet_lifecycle",
                message="Sheet access follows creation/rename or an existence guard; validate at runtime."))
        else:
            kept.append(issue)
    result.errors = kept
    result.ok = not kept
    return result

