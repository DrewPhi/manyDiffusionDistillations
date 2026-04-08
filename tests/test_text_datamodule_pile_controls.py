import json

import torch
from torch.utils.data import Subset

from manylatents.data.text import TextDataModule


class _DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    mask_token = "[MASK]"
    deprecation_warnings = {}

    def __call__(self, texts, truncation=True, max_length=8, padding="max_length", return_tensors="pt"):
        n = len(texts)
        ids = torch.zeros((n, max_length), dtype=torch.long)
        mask = torch.ones((n, max_length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def __len__(self):
        return 30522

    def get_special_tokens_mask(self, val, already_has_special_tokens=True):
        return [0 for _ in val]

    def convert_tokens_to_ids(self, token):
        if token == self.mask_token:
            return 103
        return 0

    def pad(self, encoded_inputs, return_tensors="pt", pad_to_multiple_of=None, **kwargs):
        input_ids = torch.stack([item["input_ids"] for item in encoded_inputs])
        attention_mask = torch.stack([item["attention_mask"] for item in encoded_inputs])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _fake_load_dataset(*_args, **_kwargs):
    train = [{"text": f"train sample {i}"} for i in range(20)]
    validation = [{"text": f"val sample {i}"} for i in range(10)]
    return {
        "train": train,
        "validation": validation,
        "test": validation,
        "probe": validation,
    }


def _source_ids(dataset):
    if dataset is None:
        return []
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [int(base.source_id_for_index(i)) for i in dataset.indices]
    return [int(dataset.source_id_for_index(i)) for i in range(len(dataset))]


def test_text_datamodule_token_budget_subset(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        token_budget=32,  # 32 / 8 = 4 train samples
        probe_n_samples=3,
        seed=42,
    )
    dm.setup()

    assert len(dm.train_dataset) == 4
    assert not isinstance(dm.train_dataset, Subset)
    assert len(dm.train_dataset.hf_dataset) == 4
    assert len(dm.val_dataset) == 10
    assert len(dm.probe_dataset) == 3
    assert len(dm.probe_source_ids) == 3


def test_text_datamodule_persisted_probe_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    probe_ids_path = tmp_path / "probe_ids.json"

    dm_first = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        probe_n_samples=4,
        seed=123,
        probe_ids_path=str(probe_ids_path),
        persist_probe_ids=True,
    )
    dm_first.setup()
    assert probe_ids_path.exists()
    persisted = json.loads(probe_ids_path.read_text(encoding="utf-8"))
    assert len(persisted) == 4

    # New seed, but should load persisted probe IDs instead of resampling.
    dm_second = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        probe_n_samples=4,
        seed=999,
        probe_ids_path=str(probe_ids_path),
        persist_probe_ids=False,
    )
    dm_second.setup()
    assert dm_second.probe_source_ids == persisted


def test_text_datamodule_json_zst_layout_builds_data_files(tmp_path):
    root = tmp_path / "pile_uncopyrighted"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "00.jsonl.zst").write_text("", encoding="utf-8")
    (train_dir / "01.jsonl.zst").write_text("", encoding="utf-8")
    (root / "val.jsonl.zst").write_text("", encoding="utf-8")
    (root / "test.jsonl.zst").write_text("", encoding="utf-8")

    dm = TextDataModule(
        dataset_name="json",
        dataset_config=None,
        dataset_path=str(root),
        train_split="train",
        val_split="validation",
        probe_split="validation",
    )
    args = dm._dataset_load_args()

    assert args["path"] == "json"
    data_files = args["data_files"]
    assert "train" in data_files
    assert len(data_files["train"]) == 2
    assert "validation" in data_files
    assert data_files["validation"][0].endswith("val.jsonl.zst")
    assert "test" in data_files


def test_text_datamodule_probe_only_setup_skips_other_splits(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())

    class _ExplodingSplit:
        def __iter__(self):
            raise AssertionError("non-probe split should not be iterated in probe-only setup")

    def _fake_probe_only_load_dataset(*_args, **_kwargs):
        return {
            "train": _ExplodingSplit(),
            "validation": _ExplodingSplit(),
            "test": _ExplodingSplit(),
            "probe": [{"text": f"probe sample {i}"} for i in range(6)],
        }

    monkeypatch.setattr("datasets.load_dataset", _fake_probe_only_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        test_split="test",
        probe_split="probe",
        max_length=8,
        probe_n_samples=3,
        seed=42,
    )
    dm.setup(stage="probe")

    assert dm.train_dataset is None
    assert dm.val_dataset is None
    assert dm.test_dataset is None
    assert len(dm.probe_dataset) == 3
    assert len(dm.probe_source_ids) == 3


def test_text_datamodule_reuses_cached_split_indices(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())

    class _CountingSplit:
        def __init__(self, rows):
            self._rows = rows
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return iter(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

    first_probe = _CountingSplit([{"text": f"probe sample {i}"} for i in range(6)])
    first_dataset = {
        "train": [{"text": "train sample"}],
        "validation": [{"text": "val sample"}],
        "probe": first_probe,
    }

    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_k: first_dataset)

    dm_first = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        probe_n_samples=3,
        seed=42,
        split_index_cache_dir=str(tmp_path / "split_index_cache"),
    )
    dm_first.setup(stage="probe")
    assert first_probe.iter_calls == 1
    assert len(list((tmp_path / "split_index_cache").glob("*.json"))) == 1

    class _ExplodingSplit:
        def __iter__(self):
            raise AssertionError("cached split indices should avoid rescanning this split")

        def __getitem__(self, idx):
            return {"text": f"probe sample {idx}"}

    second_dataset = {
        "train": [{"text": "train sample"}],
        "validation": [{"text": "val sample"}],
        "probe": _ExplodingSplit(),
    }
    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_k: second_dataset)

    dm_second = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        probe_n_samples=3,
        seed=42,
        split_index_cache_dir=str(tmp_path / "split_index_cache"),
    )
    dm_second.setup(stage="probe")

    assert len(dm_second.probe_dataset) == 3


def test_text_datamodule_masked_lm_train_collator_masks_labels(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        lm_objective="masked_lm",
        max_length=8,
        probe_n_samples=2,
        seed=42,
    )
    dm.setup()

    batch = dm._task_collate_fn([dm.train_dataset[0], dm.train_dataset[1]])
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] != -100).any()


def test_text_datamodule_rebuilds_split_index_cache_on_metadata_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())

    class _CountingSplit:
        def __init__(self, rows):
            self._rows = rows
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return iter(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

    split = _CountingSplit([{"text": f"probe sample {i}"} for i in range(6)])
    dataset = {
        "train": [{"text": "train sample"}],
        "validation": [{"text": "val sample"}],
        "probe": split,
    }
    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_k: dataset)

    cache_dir = tmp_path / "split_index_cache"
    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        max_length=8,
        probe_n_samples=3,
        seed=42,
        split_index_cache_dir=str(cache_dir),
    )
    dm.setup(stage="probe")

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1

    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    payload["text_field"] = "different_text_field"
    cache_files[0].write_text(json.dumps(payload), encoding="utf-8")

    dm.setup(stage="probe")

    assert split.iter_calls == 2


def test_text_datamodule_eval_only_setup_skips_train_and_probe(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())

    class _ExplodingSplit:
        def __iter__(self):
            raise AssertionError("train/probe split should not be iterated in eval-only setup")

        def __getitem__(self, idx):
            return {"text": f"unexpected sample {idx}"}

    def _fake_eval_only_load_dataset(*_args, **_kwargs):
        return {
            "train": _ExplodingSplit(),
            "validation": [{"text": f"val sample {i}"} for i in range(5)],
            "test": [{"text": f"test sample {i}"} for i in range(4)],
            "probe": _ExplodingSplit(),
        }

    monkeypatch.setattr("datasets.load_dataset", _fake_eval_only_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        test_split="test",
        probe_split="probe",
        max_length=8,
        probe_n_samples=3,
        seed=42,
    )
    dm.setup(stage="eval")

    assert dm.train_dataset is None
    assert dm.probe_dataset is None
    assert dm.probe_source_ids == []
    assert len(dm.val_dataset) == 5
    assert len(dm.test_dataset) == 4


def test_text_datamodule_applies_deterministic_eval_example_limits(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="probe",
        test_split=None,
        max_length=8,
        val_example_limit=3,
        seed=42,
    )
    dm.setup(stage="eval")

    assert len(dm.val_dataset) == 3
    assert dm.test_dataset is dm.val_dataset


def test_text_datamodule_keeps_full_probe_pool_when_probe_uses_eval_split(monkeypatch):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    dm = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="validation",
        max_length=8,
        val_example_limit=3,
        probe_n_samples=5,
        seed=42,
    )
    dm.setup()

    assert len(dm.val_dataset) == 3
    assert len(dm.probe_dataset) == 5


def test_text_datamodule_reuses_cached_train_subset_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    cache_dir = tmp_path / "split_index_cache"
    subset_cache_dir = tmp_path / "train_subset_cache"

    dm_first = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="train",
        max_length=8,
        token_budget=32,
        probe_n_samples=3,
        seed=42,
        exclude_probe_from_train=True,
        split_index_cache_dir=str(cache_dir),
        train_subset_cache_dir=str(subset_cache_dir),
    )
    dm_first.setup()
    first_train_ids = _source_ids(dm_first.train_dataset)
    cache_files = list(subset_cache_dir.glob("*.json"))
    assert len(cache_files) == 1

    original = TextDataModule._resolve_train_source_ids_uncached

    def _explode(*_args, **_kwargs):
        raise AssertionError("train subset manifest cache should avoid recomputing train source IDs")

    monkeypatch.setattr(TextDataModule, "_resolve_train_source_ids_uncached", _explode)

    dm_second = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="train",
        max_length=8,
        token_budget=32,
        probe_n_samples=3,
        seed=42,
        exclude_probe_from_train=True,
        split_index_cache_dir=str(cache_dir),
        train_subset_cache_dir=str(subset_cache_dir),
    )
    dm_second.setup()
    second_train_ids = _source_ids(dm_second.train_dataset)
    assert second_train_ids == first_train_ids

    monkeypatch.setattr(TextDataModule, "_resolve_train_source_ids_uncached", original)


def test_text_datamodule_cached_and_uncached_paths_match_memberships(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: _DummyTokenizer())
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    uncached = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="train",
        max_length=8,
        token_budget=40,
        probe_n_samples=4,
        seed=123,
        exclude_probe_from_train=True,
    )
    uncached.setup()
    uncached_probe_ids = list(uncached.probe_source_ids)
    uncached_train_ids = _source_ids(uncached.train_dataset)

    cache_dir = tmp_path / "split_index_cache"
    subset_cache_dir = tmp_path / "train_subset_cache"
    cached_first = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="train",
        max_length=8,
        token_budget=40,
        probe_n_samples=4,
        seed=123,
        exclude_probe_from_train=True,
        split_index_cache_dir=str(cache_dir),
        train_subset_cache_dir=str(subset_cache_dir),
    )
    cached_first.setup()

    cached_second = TextDataModule(
        dataset_name="dummy",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        probe_split="train",
        max_length=8,
        token_budget=40,
        probe_n_samples=4,
        seed=123,
        exclude_probe_from_train=True,
        split_index_cache_dir=str(cache_dir),
        train_subset_cache_dir=str(subset_cache_dir),
    )
    cached_second.setup()

    assert list(cached_first.probe_source_ids) == uncached_probe_ids
    assert list(cached_second.probe_source_ids) == uncached_probe_ids
    assert _source_ids(cached_first.train_dataset) == uncached_train_ids
    assert _source_ids(cached_second.train_dataset) == uncached_train_ids
