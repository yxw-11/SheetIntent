"""Check imports and CUDA build without allocating model weights."""
import argparse
import importlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    for name in ("torch", "transformers", "peft", "vllm", "verl", "pyarrow", "openpyxl"):
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "imported"))
    import torch
    assert torch.version.cuda == "12.8", f"Expected CUDA 12.8 wheel, got {torch.version.cuda}"
    if args.gpu:
        assert torch.cuda.is_available(), "No compatible allocated CUDA device"
        print(torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
