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

    def fake_auto_config_from_pretrained(name, trust_remote_code=False, revision=None, **kwargs):
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

    def fake_tokenizer_from_pretrained(name, trust_remote_code=False, **kwargs):
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


def test_hf_trainer_from_config_forces_fp32_master_weights(monkeypatch):
    """Regression: pythia configs ship `torch_dtype=float16` in config.json,
    and modern transformers' `from_config` honors it. The runner must force
    fp32 at construction so master weights remain fp32 — otherwise the
    student NaNs once gradients flow under mixed-precision training.

    Simulates the pythia case: source config carries fp16; assert that the
    config seen by `from_config` has been promoted to fp32.
    """
    seen = {}

    class DummyNetwork:
        def to(self, dtype=None):
            seen["post_to_dtype"] = dtype
            return self

    class DummyConfig:
        # Mimics what AutoConfig.from_pretrained returns for pythia: fp16.
        torch_dtype = torch.float16

    def fake_from_config(cfg, **kwargs):
        seen["from_config_torch_dtype"] = cfg.torch_dtype
        return DummyNetwork()

    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoConfig.from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_config",
        fake_from_config,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )

    cfg = HFTrainerConfig(
        model_name_or_path="EleutherAI/pythia-70m",
        init_mode="from_config",
        model_config_name_or_path="EleutherAI/pythia-70m",
        tokenizer_name="EleutherAI/pythia-70m",
        torch_dtype=torch.float32,
    )
    module = HFTrainerModule(cfg)
    module.configure_model()

    assert seen["from_config_torch_dtype"] == torch.float32, (
        "Master weights at from_config must be fp32 even when the source "
        "config carries fp16; got "
        f"{seen.get('from_config_torch_dtype')}"
    )


def test_hf_trainer_from_config_overrides_take_precedence_for_non_dtype_keys(monkeypatch):
    """The fp32 dtype force must not clobber unrelated `model_config_overrides`.
    Override `num_hidden_layers` and `hidden_size`; both must reach
    `from_config`, while `torch_dtype` is still fp32.
    """
    seen = {}

    class DummyNetwork:
        def to(self, dtype=None):
            return self

    class DummyConfig:
        torch_dtype = torch.float16
        num_hidden_layers = 24
        hidden_size = 1024

    def fake_from_config(cfg, **kwargs):
        seen["torch_dtype"] = cfg.torch_dtype
        seen["num_hidden_layers"] = cfg.num_hidden_layers
        seen["hidden_size"] = cfg.hidden_size
        return DummyNetwork()

    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoConfig.from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_config",
        fake_from_config,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )

    cfg = HFTrainerConfig(
        model_name_or_path="EleutherAI/pythia-70m",
        init_mode="from_config",
        model_config_name_or_path="EleutherAI/pythia-70m",
        model_config_overrides={"num_hidden_layers": 2, "hidden_size": 64},
        tokenizer_name="EleutherAI/pythia-70m",
        torch_dtype=torch.float32,
    )
    module = HFTrainerModule(cfg)
    module.configure_model()

    assert seen["torch_dtype"] == torch.float32
    assert seen["num_hidden_layers"] == 2
    assert seen["hidden_size"] == 64
