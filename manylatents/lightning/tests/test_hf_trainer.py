# manylatents/lightning/tests/test_hf_trainer.py
import pytest
import torch
from manylatents.lightning.hf_trainer import HFTrainerModule, HFTrainerConfig


def test_hf_trainer_module_instantiation():
    """Should instantiate with config."""
    config = HFTrainerConfig(
        model_name_or_path="gpt2",
        learning_rate=2e-5,
    )
    module = HFTrainerModule(config)

    assert module.config == config
    assert module.network is None  # Lazy init


def test_hf_trainer_config_defaults():
    """Config should have sensible defaults."""
    config = HFTrainerConfig(model_name_or_path="gpt2")

    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.0
    assert config.warmup_steps == 0


def test_hf_trainer_normalizes_string_torch_dtype():
    assert HFTrainerModule._normalize_torch_dtype("bf16") == torch.bfloat16
    assert HFTrainerModule._normalize_torch_dtype("bf16-mixed") == torch.bfloat16
    assert HFTrainerModule._normalize_torch_dtype("float32") == torch.float32
    assert HFTrainerModule._normalize_torch_dtype(None) is None


def test_hf_trainer_configure_model_from_config_mode(monkeypatch):
    """Should build model via AutoConfig+from_config when init_mode=from_config."""
    calls = {}

    class DummyNetwork:
        def __init__(self):
            self.dtype = None

        def to(self, dtype=None):
            self.dtype = dtype
            return self

    class DummyConfig:
        hidden_size = 768

    def fake_auto_config_from_pretrained(name, trust_remote_code=False, revision=None, **kwargs):
        calls["config_source"] = name
        calls["config_trust_remote_code"] = trust_remote_code
        calls["config_revision"] = revision
        return DummyConfig()

    def fake_model_from_pretrained(*args, **kwargs):
        calls["from_pretrained_called"] = True
        return DummyNetwork()

    def fake_model_from_config(cfg, **kwargs):
        calls["from_config_called"] = True
        calls["from_config_kwargs"] = kwargs
        calls["config_hidden_size"] = getattr(cfg, "hidden_size", None)
        calls["config_num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
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
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForCausalLM.from_config",
        fake_model_from_config,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    config = HFTrainerConfig(
        model_name_or_path="EleutherAI/pythia-70m",
        init_mode="from_config",
        model_config_name_or_path="EleutherAI/pythia-70m",
        model_config_overrides={"num_hidden_layers": 2},
        tokenizer_name="EleutherAI/pythia-70m",
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    assert calls.get("from_config_called", False) is True
    assert calls.get("from_pretrained_called", False) is False
    assert calls["config_source"] == "EleutherAI/pythia-70m"
    assert calls["config_num_hidden_layers"] == 2
    assert calls["tokenizer_source"] == "EleutherAI/pythia-70m"


def test_hf_trainer_uses_masked_lm_loader_for_encoder_only_models(monkeypatch):
    calls = {}

    class DummyNetwork:
        pass

    def fake_masked_lm_from_pretrained(*args, **kwargs):
        calls["masked_lm_called"] = kwargs
        return DummyNetwork()

    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoModelForMaskedLM.from_pretrained",
        fake_masked_lm_from_pretrained,
    )
    monkeypatch.setattr(
        "manylatents.lightning.hf_trainer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )

    config = HFTrainerConfig(
        model_name_or_path="bert-base-uncased",
        model_family="masked_lm",
    )
    module = HFTrainerModule(config)
    module.configure_model()

    assert "masked_lm_called" in calls


@pytest.mark.slow
def test_hf_trainer_module_forward_pass():
    """Integration test with actual tiny model."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",  # ~2MB model
        trust_remote_code=True,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    # Create dummy batch
    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world", "Test input"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    # Forward pass
    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.loss is not None
    assert outputs.logits is not None


@pytest.mark.slow
def test_hf_trainer_module_training_step():
    """Test training step computes loss."""
    config = HFTrainerConfig(model_name_or_path="sshleifer/tiny-gpt2")
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    loss = module.training_step(batch, 0)

    assert loss is not None
    assert loss.requires_grad


@pytest.mark.slow
def test_hf_trainer_with_activation_extractor():
    """Verify ActivationExtractor works with HF models - critical integration test."""
    from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

    config = HFTrainerConfig(model_name_or_path="sshleifer/tiny-gpt2")
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create probe batch
    batch = tokenizer(
        ["Hello world", "Test input", "Another sample"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )

    # Extract from last transformer block - use the actual HF path
    # For GPT2: model.transformer.h[-1] is the last block
    spec = LayerSpec(path="transformer.h[-1]", reduce="mean")
    extractor = ActivationExtractor([spec])

    module.eval()
    with torch.no_grad():
        with extractor.capture(module.network):
            _ = module.network(**batch)

    activations = extractor.get_activations()

    assert "transformer.h[-1]" in activations
    # Should have (batch_size, hidden_dim) after mean reduction
    assert activations["transformer.h[-1]"].shape[0] == 3  # 3 samples
    assert len(activations["transformer.h[-1]"].shape) == 2  # (batch, hidden)
