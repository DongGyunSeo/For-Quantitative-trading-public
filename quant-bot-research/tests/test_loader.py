"""로더 — 스키마 강제와 무결성 리포트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.data.loader import load_ohlcv


def _write(tmp_path, df, name="x.csv"):
    p = tmp_path / name
    out = df.reset_index().rename(columns={"index": "time", "time": "time"})
    out.to_csv(p, index=False)
    return p


def test_roundtrip(tmp_path):
    df = synth_5m(500, seed=1)
    d2, q = load_ohlcv(_write(tmp_path, df))
    assert len(d2) == 500 and q.gaps == 0 and q.dupes == 0
    assert np.allclose(d2["close"].to_numpy(), df["close"].to_numpy())


def test_index_is_microsecond_utc(tmp_path):
    """int64 로 저장했다 되읽을 때 단위가 흔들리면 1970년이 나온다."""
    d2, _ = load_ohlcv(_write(tmp_path, synth_5m(100)))
    assert str(d2.index.dtype) == "datetime64[us, UTC]"
    back = pd.to_datetime(d2.index.view("int64"), unit="us", utc=True)
    assert (back == d2.index).all()


def test_missing_column_raises(tmp_path):
    df = synth_5m(50).drop(columns=["volume"])
    with pytest.raises(ValueError, match="컬럼 누락"):
        load_ohlcv(_write(tmp_path, df))


def test_no_time_column_raises(tmp_path):
    p = tmp_path / "y.csv"
    pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                  "close": [1.0], "volume": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="'time'"):
        load_ohlcv(p)


def test_gap_counted(tmp_path):
    df = synth_5m(200)
    df = df.drop(df.index[50:55])
    _, q = load_ohlcv(_write(tmp_path, df))
    assert q.gaps == 5 and q.max_gap == 5


def test_duplicates_dropped(tmp_path):
    df = synth_5m(100)
    dup = pd.concat([df, df.iloc[40:45]]).sort_index()
    d2, q = load_ohlcv(_write(tmp_path, dup))
    assert q.dupes == 5 and len(d2) == 100


def test_bad_ohlc_detected(tmp_path):
    df = synth_5m(100).copy()
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] - 1.0
    _, q = load_ohlcv(_write(tmp_path, df))
    assert q.bad_ohlc >= 1


def test_symbol_files_finds_csv_and_gz(tmp_path):
    import gzip
    from qbot.data.loader import symbol_files
    (tmp_path / "AAA_5m_futures.csv").write_text("time,open,high,low,close,volume\n")
    with gzip.open(tmp_path / "BBB_5m_futures.csv.gz", "wt") as f:
        f.write("time,open,high,low,close,volume\n")
    got = symbol_files(tmp_path)
    assert set(got) == {"AAA", "BBB"}
    assert got["BBB"].suffix == ".gz"


def test_symbol_files_prefers_gz(tmp_path):
    import gzip
    from qbot.data.loader import symbol_files
    (tmp_path / "CCC_5m_futures.csv").write_text("time,open,high,low,close,volume\n")
    with gzip.open(tmp_path / "CCC_5m_futures.csv.gz", "wt") as f:
        f.write("time,open,high,low,close,volume\n")
    assert symbol_files(tmp_path)["CCC"].suffix == ".gz"


def test_load_ohlcv_reads_gzip(tmp_path):
    import gzip
    from conftest import synth_5m
    from qbot.data.loader import load_ohlcv
    df = synth_5m(300, seed=2)
    p = tmp_path / "g.csv.gz"
    out = df.reset_index().rename(columns={"index": "time"})
    with gzip.open(p, "wt", newline="") as f:
        out.to_csv(f, index=False)
    d2, q = load_ohlcv(p)
    assert len(d2) == 300 and q.gaps == 0
