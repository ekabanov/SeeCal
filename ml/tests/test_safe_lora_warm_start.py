import json
import sys
import types

import pytest

from safe_lora_warm_start import configure_model_for_warm_start


class FakeConfig:
    pass


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()


def install_fake_trainer_utils(monkeypatch, calls):
    utils = types.ModuleType("mlx_vlm.trainer.utils")

    def freeze_model(model):
        calls.append(("freeze", model))

    def apply_lora_layers(model, adapter_path):
        calls.append(("load", model, adapter_path))
        return model

    utils.freeze_model = freeze_model
    utils.apply_lora_layers = apply_lora_layers
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.trainer", types.ModuleType("trainer"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.trainer.utils", utils)


def make_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 2.0},
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapters.safetensors").write_bytes(b"checkpoint")
    return adapter, config


def test_warm_start_freezes_before_loading_and_retains_config(tmp_path, monkeypatch):
    adapter, config = make_adapter(tmp_path)
    calls = []
    install_fake_trainer_utils(monkeypatch, calls)
    model = FakeModel()

    result = configure_model_for_warm_start(model, str(adapter))

    assert result is model
    assert [call[0] for call in calls] == ["freeze", "load"]
    assert model.config.lora == config


@pytest.mark.parametrize("missing", ["adapter_config.json", "adapters.safetensors"])
def test_warm_start_rejects_incomplete_adapter(tmp_path, monkeypatch, missing):
    adapter, _ = make_adapter(tmp_path)
    (adapter / missing).unlink()
    install_fake_trainer_utils(monkeypatch, [])

    with pytest.raises(FileNotFoundError, match="Missing adapter"):
        configure_model_for_warm_start(FakeModel(), str(adapter))
