import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synth_5m(n=20_000, seed=0, vol=0.0025, start="2022-01-01"):
    """같은 스키마의 합성 5m. 실데이터 없이도 배관을 점검한다."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    h = c * (1 + np.abs(rng.normal(0, vol * 0.7, n)))
    l = c * (1 - np.abs(rng.normal(0, vol * 0.7, n)))
    o = np.concatenate(([c[0]], c[:-1]))
    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC").as_unit("us")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.abs(rng.normal(100, 20, n))}, index=idx)


def daily_atr(df, period=14):
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return np.maximum(pd.Series(a).rolling(288, min_periods=48).median().to_numpy(),
                      5e-4 * c)


@pytest.fixture(scope="session")
def df5():
    return synth_5m()


@pytest.fixture(scope="session")
def atr5(df5):
    return daily_atr(df5)
