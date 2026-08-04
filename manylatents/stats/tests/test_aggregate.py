import json
from manylatents.stats.aggregate import dig, aggregate_paths, aggregate_cell


def _write(tmp_path, seed, cka):
    p = tmp_path / f"discriminant_cka_bert_seed{seed}_2048.json"
    p.write_text(json.dumps({
        "family": "bert", "seed": seed, "tag": "_2048", "knn": 10, "N": 2048,
        "results": {"penultimate": {"po_freeze": {"cka_vs_control": cka}}},
    }))
    return p


def test_dig_nested():
    obj = {"a": {"b": {"c": 3}}}
    assert dig(obj, "a/b/c") == 3


def test_aggregate_cell_across_seeds(tmp_path):
    paths = [_write(tmp_path, 42, 0.90), _write(tmp_path, 43, 0.94),
             _write(tmp_path, 44, 0.92)]
    iv = aggregate_cell(paths, "penultimate", "po_freeze", "cka_vs_control")
    assert iv.n == 3
    assert abs(iv.mean - 0.92) < 1e-9
    assert iv.lo < iv.mean < iv.hi


def test_aggregate_paths_direct(tmp_path):
    paths = [_write(tmp_path, 1, 0.5), _write(tmp_path, 2, 0.7)]
    iv = aggregate_paths(paths, "results/penultimate/po_freeze/cka_vs_control")
    assert iv.n == 2 and abs(iv.mean - 0.6) < 1e-9
