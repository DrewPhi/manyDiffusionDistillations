# manylatents/data/text.py
"""Text data module for HuggingFace language models."""
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)

SPLIT_INDEX_CACHE_SCHEMA_VERSION = 1
TRAIN_SUBSET_CACHE_SCHEMA_VERSION = 1


def _hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@dataclass
class TextDataConfig:
    """Configuration for text data module.

    Attributes:
        dataset_name: HuggingFace dataset name (e.g., "wikitext")
        dataset_config: Dataset config (e.g., "wikitext-2-raw-v1")
        tokenizer_name: Tokenizer to use (defaults to model name)
        max_length: Maximum sequence length
        batch_size: Training batch size
        num_workers: DataLoader workers
    """
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    dataset_path: Optional[str] = None
    dataset_revision: Optional[str] = None
    train_split: str = "train"
    val_split: str = "validation"
    test_split: Optional[str] = "test"
    probe_split: Optional[str] = None
    token_budget: Optional[int] = None
    train_example_offset: int = 0
    probe_ids_path: Optional[str] = None
    persist_probe_ids: bool = False
    exclude_probe_from_train: bool = False
    val_example_limit: Optional[int] = None
    test_example_limit: Optional[int] = None
    text_field: str = "text"
    tokenizer_name: Optional[str] = None
    lm_objective: str = "causal_lm"
    mlm_probability: float = 0.15
    max_length: int = 128
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 42
    data_order_seed: Optional[int] = None
    dataloader_seed: Optional[int] = None
    split_index_cache_dir: Optional[str] = None
    train_subset_cache_dir: Optional[str] = None
    streaming: bool = False
    streaming_shuffle_buffer: int = 10_000


class TextDataModule(LightningDataModule):
    """Lightning DataModule for text data with HuggingFace models.

    Loads a HuggingFace dataset, tokenizes it, and provides DataLoaders
    for training, validation, and probing.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: Optional[str] = "wikitext-2-raw-v1",
        dataset_path: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: Optional[str] = "test",
        probe_split: Optional[str] = None,
        token_budget: Optional[int] = None,
        train_example_offset: int = 0,
        probe_ids_path: Optional[str] = None,
        persist_probe_ids: bool = False,
        exclude_probe_from_train: bool = False,
        val_example_limit: Optional[int] = None,
        test_example_limit: Optional[int] = None,
        text_field: str = "text",
        tokenizer_name: str = "gpt2",
        lm_objective: str = "causal_lm",
        mlm_probability: float = 0.15,
        max_length: int = 128,
        batch_size: int = 8,
        num_workers: int = 0,
        probe_n_samples: int = 512,
        seed: int = 42,
        data_order_seed: Optional[int] = None,
        dataloader_seed: Optional[int] = None,
        split_index_cache_dir: Optional[str] = None,
        train_subset_cache_dir: Optional[str] = None,
        streaming: bool = False,
        streaming_shuffle_buffer: int = 10_000,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_path = dataset_path
        self.dataset_revision = dataset_revision
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.probe_split = probe_split or val_split
        self.token_budget = token_budget
        self.train_example_offset = int(train_example_offset)
        self.probe_ids_path = Path(probe_ids_path) if probe_ids_path else None
        self.persist_probe_ids = persist_probe_ids
        self.exclude_probe_from_train = exclude_probe_from_train
        self.val_example_limit = int(val_example_limit) if val_example_limit is not None else None
        self.test_example_limit = int(test_example_limit) if test_example_limit is not None else None
        self.text_field = text_field
        self.tokenizer_name = tokenizer_name
        self.lm_objective = str(lm_objective)
        self.mlm_probability = float(mlm_probability)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.probe_n_samples = probe_n_samples
        self.seed = seed
        self.data_order_seed = data_order_seed if data_order_seed is not None else seed
        self.dataloader_seed = dataloader_seed if dataloader_seed is not None else seed
        self.split_index_cache_dir = Path(split_index_cache_dir) if split_index_cache_dir else None
        self.train_subset_cache_dir = Path(train_subset_cache_dir) if train_subset_cache_dir else None
        self.streaming = bool(streaming)
        self.streaming_shuffle_buffer = int(streaming_shuffle_buffer)

        self.tokenizer = None
        self._mlm_collator = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.probe_dataset = None
        self.probe_source_ids: list[int] = []

    def _seed_worker(self, worker_id: int):
        # Ensure deterministic worker-local RNG streams.
        worker_seed = (int(self.dataloader_seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def prepare_data(self):
        """Download dataset and tokenizer."""
        from datasets import load_dataset

        load_args = self._dataset_load_args()
        load_dataset(**load_args)
        AutoTokenizer.from_pretrained(self.tokenizer_name, local_files_only=_hf_local_files_only())

    def _dataset_load_args(self) -> Dict[str, object]:
        dataset_id = self.dataset_path or self.dataset_name
        kwargs: Dict[str, object] = {}
        if self.dataset_revision is not None:
            kwargs["revision"] = self.dataset_revision
        if self.streaming:
            kwargs["streaming"] = True

        # Special-case local json/jsonl(.zst) directories used on cluster storage.
        if self.dataset_name == "json" and self.dataset_path is not None:
            data_files = self._resolve_json_data_files(Path(self.dataset_path))
            kwargs["path"] = "json"
            kwargs["data_files"] = data_files
            return kwargs

        kwargs["path"] = dataset_id
        if self.dataset_config is not None:
            kwargs["name"] = self.dataset_config
        return kwargs

    def _resolve_json_data_files(self, root: Path) -> Dict[str, list[str]]:
        def _candidates(split: str) -> list[str]:
            cands = [
                f"{split}.jsonl.zst",
                f"{split}.jsonl",
                f"{split}.json",
            ]
            if split == "validation":
                cands.extend(["val.jsonl.zst", "val.jsonl", "val.json"])
            if split == "val":
                cands.extend(["validation.jsonl.zst", "validation.jsonl", "validation.json"])
            return cands

        def _resolve_split_file(split: str) -> Optional[Path]:
            for name in _candidates(split):
                p = root / name
                if p.exists():
                    return p
            return None

        data_files: Dict[str, list[str]] = {}
        train_dir = root / self.train_split
        if train_dir.exists() and train_dir.is_dir():
            train_files = sorted(train_dir.glob("*.jsonl.zst"))
            if not train_files:
                train_files = sorted(train_dir.glob("*.jsonl"))
            if train_files:
                data_files[self.train_split] = [str(p) for p in train_files]
        else:
            train_file = _resolve_split_file(self.train_split)
            if train_file is not None:
                data_files[self.train_split] = [str(train_file)]

        val_file = _resolve_split_file(self.val_split)
        if val_file is not None:
            data_files[self.val_split] = [str(val_file)]

        if self.probe_split != self.val_split:
            probe_file = _resolve_split_file(self.probe_split)
            if probe_file is not None:
                data_files[self.probe_split] = [str(probe_file)]

        test_file = _resolve_split_file("test")
        if test_file is not None:
            data_files["test"] = [str(test_file)]

        if self.test_split is not None and self.test_split not in data_files:
            test_split_file = _resolve_split_file(self.test_split)
            if test_split_file is not None:
                data_files[self.test_split] = [str(test_split_file)]

        if self.train_split not in data_files:
            raise FileNotFoundError(
                f"Could not resolve train files for split '{self.train_split}' under '{root}'"
            )
        if self.val_split not in data_files:
            raise FileNotFoundError(
                f"Could not resolve validation files for split '{self.val_split}' under '{root}'"
            )
        if self.test_split is not None and self.test_split not in data_files:
            raise FileNotFoundError(
                f"Could not resolve test split '{self.test_split}' under '{root}'"
            )
        return data_files

    def _load_raw_dataset(self):
        from datasets import load_dataset

        load_args = self._dataset_load_args()
        try:
            return load_dataset(**load_args)
        except ValueError as exc:
            msg = str(exc)
            if "Compression type zstd not supported" in msg:
                raise RuntimeError(
                    "Dataset loader cannot read .zst files in this environment. "
                    "Install zstd support (e.g. `pip install zstandard`) in the active env."
                ) from exc
            raise

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): TextDataModule._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TextDataModule._jsonable(v) for v in value]
        return value

    def _split_index_cache_root(self) -> Path:
        if self.split_index_cache_dir is not None:
            return self.split_index_cache_dir
        hf_cache = os.environ.get("HF_DATASETS_CACHE")
        if hf_cache:
            return Path(hf_cache) / "manylatents_split_indices"
        home_cache = Path.home() / ".cache" / "huggingface" / "datasets" / "manylatents_split_indices"
        try:
            home_cache.mkdir(parents=True, exist_ok=True)
            return home_cache
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "manylatents_split_indices"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _train_subset_cache_root(self) -> Path:
        if self.train_subset_cache_dir is not None:
            return self.train_subset_cache_dir
        return self._split_index_cache_root() / "train_subsets"

    def _split_index_cache_path(self, split: str, split_length: Optional[int] = None) -> Path:
        key_payload = self._split_index_cache_key_payload(split, split_length)
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self._split_index_cache_root() / f"{key}.json"

    def _split_index_cache_key_payload(self, split: str, split_length: Optional[int] = None) -> Dict[str, object]:
        return {
            "schema_version": SPLIT_INDEX_CACHE_SCHEMA_VERSION,
            "dataset_load_args": self._jsonable(self._dataset_load_args()),
            "split": str(split),
            "text_field": str(self.text_field),
            "split_length": None if split_length is None else int(split_length),
        }

    def _split_index_cache_payload(
        self,
        split: str,
        split_length: int,
        valid_indices: list[int],
    ) -> Dict[str, object]:
        return {
            **self._split_index_cache_key_payload(split, split_length),
            "num_valid_indices": int(len(valid_indices)),
            "valid_indices": [int(v) for v in valid_indices],
        }

    def _cache_payload_matches(self, payload: object, split: str, split_length: Optional[int] = None) -> bool:
        if not isinstance(payload, dict):
            return False
        expected = self._split_index_cache_key_payload(split, split_length)
        for key, value in expected.items():
            if payload.get(key) != value:
                logger.warning(
                    "Ignoring stale split index cache for split='%s': expected %s=%r, found %r",
                    split,
                    key,
                    value,
                    payload.get(key),
                )
                return False
        valid_indices = payload.get("valid_indices")
        return isinstance(valid_indices, list)

    def _resolve_valid_indices(self, hf_dataset, split: str) -> list[int]:
        try:
            split_length = len(hf_dataset)
        except TypeError:
            split_length = None
        cache_path = self._split_index_cache_path(split, split_length)
        if cache_path.exists():
            logger.info("Split index cache hit for split='%s' at %s", split, cache_path)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if self._cache_payload_matches(payload, split, split_length):
                return [int(v) for v in payload["valid_indices"]]
            if isinstance(payload, list):
                logger.warning(
                    "Ignoring legacy split index cache without metadata for split='%s' at %s",
                    split,
                    cache_path,
                )
            else:
                logger.warning(
                    "Rebuilding invalid split index cache for split='%s' at %s",
                    split,
                    cache_path,
                )

        logger.info("Split index cache miss for split='%s'; scanning dataset", split)
        valid_indices = [
            i for i, ex in enumerate(hf_dataset)
            if ex[self.text_field].strip()
        ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._split_index_cache_payload(split, split_length, valid_indices)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info(
            "Cached %d valid indices for split='%s' at %s",
            len(valid_indices),
            split,
            cache_path,
        )
        return [int(v) for v in valid_indices]

    def _train_subset_cache_key_payload(self) -> Dict[str, object]:
        return {
            "schema_version": TRAIN_SUBSET_CACHE_SCHEMA_VERSION,
            "dataset_load_args": self._jsonable(self._dataset_load_args()),
            "train_split": str(self.train_split),
            "probe_split": str(self.probe_split),
            "text_field": str(self.text_field),
            "max_length": int(self.max_length),
            "token_budget": int(self.token_budget) if self.token_budget is not None else None,
            "train_example_offset": int(self.train_example_offset),
            "exclude_probe_from_train": bool(self.exclude_probe_from_train),
            "data_order_seed": int(self.data_order_seed),
        }

    def _train_subset_cache_path(self) -> Path:
        key = hashlib.sha256(
            json.dumps(
                self._train_subset_cache_key_payload(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return self._train_subset_cache_root() / f"{key}.json"

    def _train_subset_cache_payload(self, train_source_ids: list[int]) -> Dict[str, object]:
        return {
            **self._train_subset_cache_key_payload(),
            "probe_source_ids": [int(v) for v in self.probe_source_ids],
            "num_train_source_ids": int(len(train_source_ids)),
            "train_source_ids": [int(v) for v in train_source_ids],
        }

    def _train_subset_cache_matches(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        expected = self._train_subset_cache_key_payload()
        for key, value in expected.items():
            if payload.get(key) != value:
                logger.warning(
                    "Ignoring stale train subset cache: expected %s=%r, found %r",
                    key,
                    value,
                    payload.get(key),
                )
                return False
        if payload.get("probe_source_ids") != [int(v) for v in self.probe_source_ids]:
            logger.warning("Ignoring stale train subset cache due to probe source ID mismatch")
            return False
        return isinstance(payload.get("train_source_ids"), list)

    def _resolve_train_source_ids_uncached(self, train_valid_indices: list[int]) -> list[int]:
        train_source_ids = [int(v) for v in train_valid_indices]
        logger.info("Resolved %d valid train source IDs before filtering", len(train_source_ids))

        if self.exclude_probe_from_train and self.probe_split == self.train_split and self.probe_source_ids:
            probe_source_ids = set(int(v) for v in self.probe_source_ids)
            train_source_ids = [sid for sid in train_source_ids if sid not in probe_source_ids]
            if not train_source_ids:
                raise ValueError("Excluding probe examples from train leaves an empty train dataset")
            logger.info(
                "Excluded %d probe source IDs from train; %d remain",
                len(probe_source_ids),
                len(train_source_ids),
            )

        if self.token_budget is None:
            return train_source_ids
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0 when provided")
        if self.train_example_offset < 0:
            raise ValueError("train_example_offset must be >= 0")

        n_train_target = int(math.ceil(float(self.token_budget) / float(self.max_length)))
        start = min(int(self.train_example_offset), len(train_source_ids))
        remaining = max(0, len(train_source_ids) - start)
        n_train_target = min(n_train_target, remaining)
        if n_train_target <= 0:
            raise ValueError(
                "Token-budget subset is empty after applying train_example_offset="
                f"{self.train_example_offset} to dataset of size {len(train_source_ids)}"
            )

        if n_train_target < len(train_source_ids):
            generator = torch.Generator().manual_seed(int(self.data_order_seed))
            perm = torch.randperm(len(train_source_ids), generator=generator).tolist()
            selected_positions = perm[start:start + n_train_target]
            train_source_ids = [train_source_ids[pos] for pos in selected_positions]
        logger.info(
            "Selected %d train source IDs after token budget / offset filtering",
            len(train_source_ids),
        )
        return [int(v) for v in train_source_ids]

    def _load_or_sample_probe_indices(self, dataset: "TokenizedDataset", n_probe: int) -> list[int]:
        if n_probe <= 0:
            return []

        # Reuse persisted probe source IDs when present.
        if self.probe_ids_path is not None and self.probe_ids_path.exists():
            logger.info("Loading persisted probe IDs from %s", self.probe_ids_path)
            payload = json.loads(self.probe_ids_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                loaded_source_ids = [int(v) for v in payload]
                idxs: list[int] = []
                for sid in loaded_source_ids:
                    idx = dataset.index_from_source_id(sid)
                    if idx is not None:
                        idxs.append(int(idx))
                if idxs:
                    logger.info("Resolved %d persisted probe examples", min(len(idxs), n_probe))
                    return idxs[:n_probe]

        # Deterministic random sampling from probe split.
        logger.info("Sampling %d probe examples with seed=%d", n_probe, self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:n_probe]
        return [int(v) for v in indices.tolist()]

    def _persist_probe_source_ids(self, dataset: "TokenizedDataset", probe_indices: list[int]):
        source_ids = [int(dataset.source_id_for_index(i)) for i in probe_indices]
        self.probe_source_ids = source_ids
        logger.info("Resolved %d canonical probe source IDs", len(source_ids))
        if self.persist_probe_ids and self.probe_ids_path is not None:
            self.probe_ids_path.parent.mkdir(parents=True, exist_ok=True)
            self.probe_ids_path.write_text(json.dumps(source_ids, indent=2), encoding="utf-8")
            logger.info("Persisted probe source IDs to %s", self.probe_ids_path)

    def _resolve_train_source_ids(self, train_valid_indices: list[int]) -> list[int]:
        if self.token_budget is None and not (
            self.exclude_probe_from_train and self.probe_split == self.train_split and self.probe_source_ids
        ):
            return [int(v) for v in train_valid_indices]

        cache_path = self._train_subset_cache_path()
        if cache_path.exists():
            logger.info("Train subset cache hit at %s", cache_path)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if self._train_subset_cache_matches(payload):
                return [int(v) for v in payload["train_source_ids"]]
            logger.warning("Rebuilding invalid train subset cache at %s", cache_path)

        logger.info("Train subset cache miss; resolving train source IDs")
        train_source_ids = self._resolve_train_source_ids_uncached(train_valid_indices)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(self._train_subset_cache_payload(train_source_ids)),
            encoding="utf-8",
        )
        logger.info("Cached %d resolved train source IDs at %s", len(train_source_ids), cache_path)
        return [int(v) for v in train_source_ids]

    def _apply_example_limit(
        self,
        source_ids: list[int],
        example_limit: Optional[int],
        split: str,
    ) -> list[int]:
        if example_limit is None:
            return [int(v) for v in source_ids]
        if example_limit <= 0:
            raise ValueError(f"{split}_example_limit must be > 0 when provided")
        limited = [int(v) for v in source_ids[:example_limit]]
        logger.info(
            "Selected %d/%d %s examples after deterministic example-limit filtering",
            len(limited),
            len(source_ids),
            split,
        )
        return limited

    def _exclude_probe_from_train_dataset(self, train_dataset: "TokenizedDataset") -> Dataset:
        if not self.exclude_probe_from_train or self.probe_split != self.train_split:
            return train_dataset
        if not self.probe_source_ids:
            return train_dataset

        probe_source_ids = set(int(v) for v in self.probe_source_ids)
        kept_indices = [
            local_idx
            for local_idx in range(len(train_dataset))
            if int(train_dataset.source_id_for_index(local_idx)) not in probe_source_ids
        ]
        if not kept_indices:
            raise ValueError("Excluding probe examples from train leaves an empty train dataset")
        return Subset(train_dataset, kept_indices)

    def _apply_token_budget_subset(self):
        if self.token_budget is None:
            return
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0 when provided")
        if self.train_example_offset < 0:
            raise ValueError("train_example_offset must be >= 0")

        n_train_target = int(math.ceil(float(self.token_budget) / float(self.max_length)))
        start = min(int(self.train_example_offset), len(self.train_dataset))
        remaining = max(0, len(self.train_dataset) - start)
        n_train_target = min(n_train_target, remaining)
        if n_train_target <= 0:
            raise ValueError(
                "Token-budget subset is empty after applying train_example_offset="
                f"{self.train_example_offset} to dataset of size {len(self.train_dataset)}"
            )
        if n_train_target < len(self.train_dataset):
            generator = torch.Generator().manual_seed(int(self.data_order_seed))
            perm = torch.randperm(len(self.train_dataset), generator=generator).tolist()
            idx = perm[start:start + n_train_target]
            self.train_dataset = Subset(self.train_dataset, idx)

    def _build_tokenize_fn(self):
        def tokenize_fn(examples):
            # Filter empty strings
            texts = [t for t in examples[self.text_field] if t.strip()]
            if not texts:
                return {"input_ids": [], "attention_mask": [], "labels": []}

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if self.lm_objective != "masked_lm":
                tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        return tokenize_fn

    def _setup_streaming(self, stage: Optional[str], tokenize_fn):
        """Streaming setup: avoids mmap-RSS inflation that breaks DDP under
        SLURM cgroups.

        - Probe split: ``shuffle(buffer).take(N)`` materialized to a small
          in-memory list (probe is rank-0 only via the runner's filesystem
          share, so no DDP shard split here).
        - Train split: ``filter`` empty texts, optional ``skip``/``take`` for
          token-budget, ``shuffle(buffer)``, ``split_dataset_by_node`` so each
          DDP rank consumes its own shard, then on-the-fly ``map(tokenize_fn)``.
        - Val/test: same shape, no shard split (each rank evaluates the full
          val set, Lightning aggregates).

        Reproducibility caveats vs the mapped path:
        - probe_source_ids are not meaningful (no canonical index); set to []
        - train_subset cache and split_index cache are skipped — streaming
          mode trades reproducibility-at-the-row-level for memory efficiency
        """
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        load_args = self._dataset_load_args()
        streamed = load_dataset(**load_args)
        text_field = self.text_field

        def _nonempty(ex):
            return ex[text_field].strip() != ""

        # ---- Probe (always materialize; small) -----------------------------
        if stage in {"probe", "probe_only"} or stage is None or stage == "fit":
            probe_n = int(self.probe_n_samples)
            buf = max(probe_n * 4, min(self.streaming_shuffle_buffer, 4_000))
            probe_streamed = (
                streamed[self.probe_split]
                .filter(_nonempty)
                .shuffle(seed=int(self.seed), buffer_size=buf)
                .take(probe_n)
            )
            probe_examples: list[Dict[str, torch.Tensor]] = []
            for ex in probe_streamed:
                tokens = tokenize_fn({text_field: [ex[text_field]]})
                if not tokens.get("input_ids") is None and len(tokens["input_ids"]) == 0:
                    continue
                entry: Dict[str, torch.Tensor] = {
                    "input_ids": tokens["input_ids"][0],
                    "attention_mask": tokens["attention_mask"][0],
                }
                if "labels" in tokens:
                    entry["labels"] = tokens["labels"][0]
                probe_examples.append(entry)
            self.probe_dataset = _ListDataset(probe_examples)
            self.probe_source_ids = []
            logger.info("Streaming probe materialized: %d examples", len(probe_examples))

        if stage in {"probe", "probe_only"}:
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            return

        # ---- Val / test (each rank streams full split) ----------------------
        val_streamed = streamed[self.val_split].filter(_nonempty)
        if self.val_example_limit is not None:
            val_streamed = val_streamed.take(int(self.val_example_limit))
        val_streamed = val_streamed.map(tokenize_fn, batched=True, remove_columns=[text_field])
        self.val_dataset = _StreamingTorchDataset(val_streamed)

        if self.test_split is not None and self.test_split in streamed:
            test_streamed = streamed[self.test_split].filter(_nonempty)
            if self.test_example_limit is not None:
                test_streamed = test_streamed.take(int(self.test_example_limit))
            test_streamed = test_streamed.map(tokenize_fn, batched=True, remove_columns=[text_field])
            self.test_dataset = _StreamingTorchDataset(test_streamed)
        else:
            self.test_dataset = self.val_dataset

        if stage in {"eval", "eval_only"}:
            self.train_dataset = None
            return

        # ---- Train (DDP-sharded streaming) ---------------------------------
        train_streamed = streamed[self.train_split].filter(_nonempty)
        if self.train_example_offset > 0:
            train_streamed = train_streamed.skip(int(self.train_example_offset))
        if self.token_budget is not None:
            if self.token_budget <= 0:
                raise ValueError("token_budget must be > 0 when provided")
            n_train_target = int(math.ceil(float(self.token_budget) / float(self.max_length)))
            train_streamed = train_streamed.take(n_train_target)

        train_streamed = train_streamed.shuffle(
            seed=int(self.data_order_seed),
            buffer_size=int(self.streaming_shuffle_buffer),
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if world > 1:
                train_streamed = split_dataset_by_node(train_streamed, world_size=world, rank=rank)
                logger.info("Streaming train split sharded for DDP: rank=%d/%d", rank, world)

        train_streamed = train_streamed.map(tokenize_fn, batched=True, remove_columns=[text_field])
        self.train_dataset = _StreamingTorchDataset(train_streamed)

    def _setup_probe_only(self, dataset, tokenize_fn):
        logger.info("Running probe-only datamodule setup for split='%s'", self.probe_split)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        probe_source_dataset = TokenizedDataset(
            dataset[self.probe_split],
            tokenize_fn,
            self.max_length,
            text_field=self.text_field,
            valid_indices=self._resolve_valid_indices(dataset[self.probe_split], self.probe_split),
        )
        n_probe = min(self.probe_n_samples, len(probe_source_dataset))
        probe_indices = self._load_or_sample_probe_indices(probe_source_dataset, n_probe=n_probe)
        self._persist_probe_source_ids(probe_source_dataset, probe_indices)
        self.probe_dataset = Subset(probe_source_dataset, probe_indices)

    def _setup_eval_only(self, dataset, tokenize_fn):
        logger.info(
            "Running eval-only datamodule setup for val_split='%s', test_split='%s'",
            self.val_split,
            self.test_split,
        )
        self.train_dataset = None
        self.probe_dataset = None
        self.probe_source_ids = []

        val_valid_indices = self._apply_example_limit(
            self._resolve_valid_indices(dataset[self.val_split], self.val_split),
            self.val_example_limit,
            split=self.val_split,
        )
        self.val_dataset = TokenizedDataset(
            dataset[self.val_split],
            tokenize_fn,
            self.max_length,
            text_field=self.text_field,
            valid_indices=val_valid_indices,
        )
        if self.test_split is not None:
            test_valid_indices = self._apply_example_limit(
                self._resolve_valid_indices(dataset[self.test_split], self.test_split),
                self.test_example_limit,
                split=self.test_split,
            )
            self.test_dataset = TokenizedDataset(
                dataset[self.test_split],
                tokenize_fn,
                self.max_length,
                text_field=self.text_field,
                valid_indices=test_valid_indices,
            )
        else:
            self.test_dataset = self.val_dataset

    def setup(self, stage: Optional[str] = None):
        """Tokenize and prepare datasets."""
        logger.info("TextDataModule.setup(stage=%s) starting", stage)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            local_files_only=_hf_local_files_only(),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.lm_objective == "masked_lm":
            if self.tokenizer.mask_token is None:
                raise ValueError(
                    f"Tokenizer '{self.tokenizer_name}' does not define a mask token required for masked_lm"
                )
            self._mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
            )
        else:
            self._mlm_collator = None

        tokenize_fn = self._build_tokenize_fn()

        if self.streaming:
            logger.info("TextDataModule.setup(stage=%s) using streaming path", stage)
            self._setup_streaming(stage, tokenize_fn)
            logger.info("TextDataModule.setup(stage=%s) finished via streaming path", stage)
            return

        logger.info("Loading raw dataset")
        dataset = self._load_raw_dataset()

        if stage in {"probe", "probe_only"}:
            self._setup_probe_only(dataset, tokenize_fn)
            logger.info("TextDataModule.setup(stage=%s) finished via probe-only path", stage)
            return
        if stage in {"eval", "eval_only"}:
            self._setup_eval_only(dataset, tokenize_fn)
            logger.info("TextDataModule.setup(stage=%s) finished via eval-only path", stage)
            return

        train_valid_indices = self._resolve_valid_indices(dataset[self.train_split], self.train_split)
        val_valid_indices = self._apply_example_limit(
            self._resolve_valid_indices(dataset[self.val_split], self.val_split),
            self.val_example_limit,
            split=self.val_split,
        )
        if self.test_split is not None:
            test_valid_indices = self._apply_example_limit(
                self._resolve_valid_indices(dataset[self.test_split], self.test_split),
                self.test_example_limit,
                split=self.test_split,
            )
        else:
            test_valid_indices = None
        if self.probe_split == self.train_split:
            probe_valid_indices = train_valid_indices
        else:
            # Probe construction must use the full canonical split index, not eval-limited subsets.
            probe_valid_indices = self._resolve_valid_indices(dataset[self.probe_split], self.probe_split)

        self.val_dataset = TokenizedDataset(
            dataset[self.val_split],
            tokenize_fn,
            self.max_length,
            text_field=self.text_field,
            valid_indices=val_valid_indices,
        )
        if self.test_split is not None:
            self.test_dataset = TokenizedDataset(
                dataset[self.test_split],
                tokenize_fn,
                self.max_length,
                text_field=self.text_field,
                valid_indices=test_valid_indices,
            )
        else:
            self.test_dataset = self.val_dataset
        probe_source_dataset = TokenizedDataset(
            dataset[self.probe_split],
            tokenize_fn,
            self.max_length,
            text_field=self.text_field,
            valid_indices=probe_valid_indices,
        )

        # Create fixed probe subset (optionally persisted by source IDs).
        n_probe = min(self.probe_n_samples, len(probe_source_dataset))
        probe_indices = self._load_or_sample_probe_indices(probe_source_dataset, n_probe=n_probe)
        self._persist_probe_source_ids(probe_source_dataset, probe_indices)
        self.probe_dataset = Subset(probe_source_dataset, probe_indices)

        train_source_ids = self._resolve_train_source_ids(train_valid_indices)
        if len(train_source_ids) == len(train_valid_indices):
            logger.info("Using full valid train split without HF subset materialization")
            self.train_dataset = TokenizedDataset(
                dataset[self.train_split],
                tokenize_fn,
                self.max_length,
                text_field=self.text_field,
                valid_indices=train_valid_indices,
            )
        else:
            logger.info("Materializing compact HF train subset with %d rows", len(train_source_ids))
            train_split_dataset = dataset[self.train_split]
            if hasattr(train_split_dataset, "select"):
                train_hf_subset = train_split_dataset.select(train_source_ids)
            else:
                train_hf_subset = [train_split_dataset[idx] for idx in train_source_ids]
            self.train_dataset = TokenizedDataset(
                train_hf_subset,
                tokenize_fn,
                self.max_length,
                text_field=self.text_field,
                valid_indices=list(range(len(train_source_ids))),
            )
        logger.info("TextDataModule.setup(stage=%s) finished", stage)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup() for training first")
        generator = torch.Generator().manual_seed(int(self.data_order_seed))
        # Streaming IterableDataset is already shuffled + DDP-sharded inside
        # the HF stream, so DataLoader must not request shuffle (would error).
        is_streaming = isinstance(self.train_dataset, torch.utils.data.IterableDataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if is_streaming else True,
            num_workers=self.num_workers,
            collate_fn=self._task_collate_fn,
            worker_init_fn=self._seed_worker,
            generator=generator,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized; call setup() for evaluation first")
        generator = torch.Generator().manual_seed(int(self.dataloader_seed))
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._task_collate_fn,
            worker_init_fn=self._seed_worker,
            generator=generator,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized; call setup() for evaluation first")
        generator = torch.Generator().manual_seed(int(self.dataloader_seed))
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._task_collate_fn,
            worker_init_fn=self._seed_worker,
            generator=generator,
        )

    def probe_dataloader(self):
        """Fixed subset for representation probing."""
        if self.probe_dataset is None:
            raise RuntimeError("probe_dataset is not initialized; call setup() for probing first")
        generator = torch.Generator().manual_seed(int(self.dataloader_seed))
        return DataLoader(
            self.probe_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._stack_collate_fn,
            worker_init_fn=self._seed_worker,
            generator=generator,
        )

    def _stack_collate_fn(self, batch):
        """Collate tokenized examples into a deterministic batch."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "labels" in batch[0]:
            payload["labels"] = torch.stack([b["labels"] for b in batch])
        return payload

    def _task_collate_fn(self, batch):
        """Collate task batches, masking tokens for MLM objectives."""
        if self.lm_objective == "masked_lm":
            if self._mlm_collator is None:
                raise RuntimeError("MLM collator is not initialized; call setup() first")
            examples = [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in batch
            ]
            return self._mlm_collator(examples)
        return self._stack_collate_fn(batch)


class TokenizedDataset(Dataset):
    """Lazily tokenized dataset wrapper."""

    def __init__(
        self,
        hf_dataset,
        tokenize_fn,
        max_length: int,
        text_field: str = "text",
        valid_indices: Optional[list[int]] = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.text_field = text_field
        self._cache = {}

        # Pre-filter to non-empty texts, or reuse cached valid indices when available.
        self.valid_indices = [int(v) for v in valid_indices] if valid_indices is not None else [
            i for i, ex in enumerate(hf_dataset)
            if ex[self.text_field].strip()
        ]
        self._source_to_local = {src: i for i, src in enumerate(self.valid_indices)}

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        if real_idx not in self._cache:
            example = self.hf_dataset[real_idx]
            tokenized = self.tokenize_fn({self.text_field: [example[self.text_field]]})
            cached = {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
            }
            if "labels" in tokenized:
                cached["labels"] = tokenized["labels"][0]
            self._cache[real_idx] = cached
        return self._cache[real_idx]

    def source_id_for_index(self, idx: int) -> int:
        return int(self.valid_indices[idx])

    def index_from_source_id(self, source_id: int) -> Optional[int]:
        return self._source_to_local.get(int(source_id))


class _ListDataset(Dataset):
    """Map-style dataset over an in-memory list of pre-tokenized examples.

    Used for the probe split under streaming mode: the probe is small enough
    to materialize, and downstream code (probe_dataloader, snapshot build)
    expects a map-style dataset with ``__len__`` and ``__getitem__``.
    """

    def __init__(self, examples: list[Dict[str, torch.Tensor]]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class _StreamingTorchDataset(torch.utils.data.IterableDataset):
    """Wrap an HF ``IterableDataset`` for use with PyTorch ``DataLoader``.

    Each yielded example is converted to a dict of torch tensors so the
    existing ``_stack_collate_fn`` / ``_task_collate_fn`` paths work without
    modification.
    """

    def __init__(self, hf_iterable):
        self.hf_iterable = hf_iterable

    def __iter__(self):
        for ex in self.hf_iterable:
            entry: Dict[str, torch.Tensor] = {
                "input_ids": torch.as_tensor(ex["input_ids"]),
                "attention_mask": torch.as_tensor(ex["attention_mask"]),
            }
            if "labels" in ex:
                entry["labels"] = torch.as_tensor(ex["labels"])
            yield entry
