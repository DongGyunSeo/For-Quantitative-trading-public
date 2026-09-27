"""tools/data_audit.py — 감사는 표시만 하고, 대조는 정확해야 한다."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_audit", ROOT / "tools" / "data_audit.py")
DA = importlib.util.module_from_spec(spec)
sys.modules["data_audit"] = DA
spec.loader.exec_module(DA)


def test_longest_run():
    assert DA.longest_run(np.array([], bool)) == 0
    assert DA.longest_run(np.array([False, False])) == 0
    assert DA.longest_run(np.array([True])) == 1
    assert DA.longest_run(np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)) == 3
    assert DA.longest_run(np.array([1, 1, 1, 1], bool)) == 4


def _frame(n=500, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC").as_unit("us")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.concatenate(([c[0]], c[:-1]))
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=rng.uniform(1, 2, n)), index=idx)


def test_compare_identical_subset():
    new = _frame()
    old = new.iloc[50:400].drop(new.index[100])     # 기존은 부분집합 + 한 봉 누락
    r = DA.compare(old, new, None)
    assert r["common"] == len(old)
    assert r["only_old"] == 0
    assert r["only_new"] == len(new) - len(old)
    for k in ("open", "high", "low", "close"):
        assert r[f"{k}_mismatch"] == 0
    assert r["vol_equal_pct"] == 100.0


def test_compare_detects_price_and_volume_diff():
    new = _frame()
    old = new.copy()
    old.iloc[10, old.columns.get_loc("close")] *= 1.0001
    old["volume"] = old["volume"] * 2.0                 # 단위가 다르면 비율로 잡힌다
    r = DA.compare(old, new, None)
    assert r["close_mismatch"] == 1
    assert r["open_mismatch"] == 0
    assert r["vol_equal_pct"] == 0.0
    assert abs(r["vol_ratio_med"] - 0.5) < 1e-12


def test_compare_until_cuts_both():
    new = _frame()
    old = new.copy()
    cut = new.index[199]
    r = DA.compare(old, new, cut)
    assert r["common"] == 200 and r["old_rows"] == 200 and r["new_rows"] == 200


def test_seed_is_process_stable():
    spec2 = importlib.util.spec_from_file_location("gate", ROOT / "examples" / "fvg_gate_stability.py")
    G = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(G)
    # zlib.crc32 는 PYTHONHASHSEED 와 무관 — 고정값
    assert G.seed_of("BTC") == 3176990918
    import zlib
    assert G.seed_of("ETH") == zlib.crc32(b"ETH")
