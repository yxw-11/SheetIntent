"""Opt-in verl hook: initialize a NEW stage from common LoRA, not optimizer state."""
import os
from pathlib import Path
from .io import file_hash, read


def install():
    from verl.workers import fsdp_workers
    path = Path(os.environ["COMMON_ADAPTER"])
    expected = read(Path(os.environ["STAGE_MANIFEST"]))["adapter"]
    original = fsdp_workers.get_peft_model

    def initialized(model, config, *args, **kwargs):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        assert all(file_hash(path / name) == h for name, h in expected.items()), "Common adapter changed"
        trained = read(path / "adapter_config.json")
        assert config.r == trained["r"] and config.lora_alpha == trained["lora_alpha"]
        model = original(model, config, *args, **kwargs)
        state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(model)
        assert set(actual) == set(state) and all(actual[k].shape == v.shape for k, v in state.items())
        result = set_peft_model_state_dict(model, state)
        assert not result.unexpected_keys
        actual = get_peft_model_state_dict(model)
        assert all(actual[k].detach().cpu().equal(v.to(actual[k].dtype)) for k, v in state.items())
        print("Common policy adapter verified and restored", flush=True)
        return model

    fsdp_workers.get_peft_model = initialized


install()
