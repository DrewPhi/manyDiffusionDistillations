"""HuggingFace model wrapper for PyTorch Lightning."""
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import ModelOutput


def _hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@dataclass
class HFTrainerConfig:
    """Configuration for HFTrainerModule.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path
        init_mode: Model initialization mode ("pretrained" or "from_config")
        model_config_name_or_path: Optional config source when init_mode="from_config"
        model_config_overrides: Optional kwargs merged into loaded AutoConfig
        tokenizer_name: Optional tokenizer source (defaults to model_name_or_path)
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps for scheduler
        adam_epsilon: Epsilon for AdamW
        torch_dtype: Optional dtype for model (e.g., torch.bfloat16)
        trust_remote_code: Whether to trust remote code for model loading
        attn_implementation: Attention implementation (e.g., "flash_attention_2")
    """
    model_name_or_path: str
    model_revision: Optional[str] = None
    init_mode: str = "pretrained"
    model_config_name_or_path: Optional[str] = None
    model_config_overrides: Optional[Dict[str, Any]] = None
    tokenizer_name: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    torch_dtype: Optional[Any] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    model_family: str = "causal_lm"


class HFTrainerModule(LightningModule):
    """Lightning module wrapping HuggingFace causal LM.

    Features:
    - Lazy model initialization (for FSDP compatibility)
    - Exposes .network for activation extraction
    - Standard Lightning training interface
    """

    def __init__(self, config: HFTrainerConfig, datamodule=None):
        super().__init__()
        self.config = config
        self.network: Optional[torch.nn.Module] = None
        self.tokenizer = None
        # datamodule passed by experiment.py but not used here

        self.save_hyperparameters({"config": config.__dict__})

    @staticmethod
    def _normalize_torch_dtype(value: Any) -> Optional[torch.dtype]:
        if value is None or isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "bf16": torch.bfloat16,
                "bf16-mixed": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "16": torch.float16,
                "16-mixed": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "32": torch.float32,
            }
            if normalized in mapping:
                return mapping[normalized]
            if normalized == "auto":
                return None
        raise ValueError(f"Unsupported torch_dtype value: {value!r}")

    def configure_model(self) -> None:
        """Lazy model initialization for FSDP compatibility."""
        if self.network is not None:
            return

        normalized_torch_dtype = self._normalize_torch_dtype(self.config.torch_dtype)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": _hf_local_files_only(),
        }
        if self.config.model_revision is not None:
            model_kwargs["revision"] = self.config.model_revision
        if normalized_torch_dtype is not None:
            model_kwargs["torch_dtype"] = normalized_torch_dtype
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        model_family = str(self.config.model_family).lower()
        if model_family == "seq2seq_lm":
            pretrained_cls = AutoModelForSeq2SeqLM
        elif model_family == "masked_lm":
            pretrained_cls = AutoModelForMaskedLM
        else:
            pretrained_cls = AutoModelForCausalLM

        if self.config.init_mode == "pretrained":
            model_kwargs["pretrained_model_name_or_path"] = self.config.model_name_or_path
            self.network = pretrained_cls.from_pretrained(**model_kwargs)
        elif self.config.init_mode == "from_config":
            config_source = (
                self.config.model_config_name_or_path or self.config.model_name_or_path
            )
            model_config = AutoConfig.from_pretrained(
                config_source,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.model_revision,
                local_files_only=_hf_local_files_only(),
            )
            for key, value in (self.config.model_config_overrides or {}).items():
                setattr(model_config, key, value)
            from_config_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.config.trust_remote_code,
            }
            if self.config.attn_implementation is not None:
                from_config_kwargs["attn_implementation"] = self.config.attn_implementation
            self.network = pretrained_cls.from_config(
                model_config,
                **from_config_kwargs,
            )
            if normalized_torch_dtype is not None:
                self.network = self.network.to(dtype=normalized_torch_dtype)
        else:
            raise ValueError(
                f"Unsupported init_mode='{self.config.init_mode}'. Expected 'pretrained' or 'from_config'."
            )

        tokenizer_source = self.config.tokenizer_name or self.config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.tokenizer_revision or self.config.model_revision,
            local_files_only=_hf_local_files_only(),
        )

    def forward(self, **inputs) -> ModelOutput:
        """Forward pass through the model."""
        return self.network(**inputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard training step."""
        outputs: ModelOutput = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard validation step."""
        outputs: ModelOutput = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard test step."""
        outputs: ModelOutput = self(**batch)
        loss = outputs.loss
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW with optional warmup."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        if self.config.warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
