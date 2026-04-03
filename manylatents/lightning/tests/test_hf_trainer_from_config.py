import torch

from manylatents.lightning.hf_trainer import HFTrainerConfig, HFTrainerModule


def test_hf_trainer_from_config_uses_model_config_source(monkeypatch):
    calls = {}

    class DummyNetwork:
        def __init__(self):
            self.dtype = None

        def to(self, dtype=None):
            self.dtype = dtype
            return self

    class DummyConfig:
        hidden_size = 128

    def fake_auto_config_from_pretrained(name, trust_remote_code=False, revision=None):
        calls["config_source"] = name
        calls["config_trust_remote_code"] = trust_remote_code
        calls["config_revision"] = revision
        return DummyConfig()

    def fake_model_from_config(cfg, **kwargs):
        calls["from_config_called"] = True
        calls["config_hidden_size"] = getattr(cfg, "hidden_size", None)
        calls["config_num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
        calls["from_config_kwargs"] = kwargs
        return DummyNetwork()

    def fake_tokenizer_from_pretrained(name, trust_remote_code=False):
        calls["tokenizer_source"] = name
        calls["tokenizer_trust_remote_code"] = trust_remote_code
        return object()

    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoConfig.from_pretrained",
        fake_auto_config_from_pretrained,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_config",
        fake_model_from_config,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    cfg = HFTrainerConfig(
        model_name_or_path="EleutherAI/pythia-70m",
        init_mode="from_config",
        model_config_name_or_path="EleutherAI/pythia-70m",
        model_config_overrides={"num_hidden_layers": 2},
        tokenizer_name="EleutherAI/pythia-70m",
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    module = HFTrainerModule(cfg)
    module.configure_model()

    assert calls.get("from_config_called", False) is True
    assert calls["config_source"] == "EleutherAI/pythia-70m"
    assert calls["config_num_hidden_layers"] == 2
    assert calls["tokenizer_source"] == "EleutherAI/pythia-70m"


def test_hf_trainer_from_config_tokenizer_falls_back_to_model_name(monkeypatch):
    calls = {}

    class DummyNetwork:
        pass

    class DummyConfig:
        pass

    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoConfig.from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_config",
        lambda *args, **kwargs: DummyNetwork(),
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        lambda name, **kwargs: calls.setdefault("tokenizer_source", name) or object(),
    )

    cfg = HFTrainerConfig(
        model_name_or_path="EleutherAI/pythia-70m",
        init_mode="from_config",
        tokenizer_name=None,
    )
    module = HFTrainerModule(cfg)
    module.configure_model()

    assert calls["tokenizer_source"] == "EleutherAI/pythia-70m"
