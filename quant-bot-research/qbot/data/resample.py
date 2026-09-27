"""5m -> HTF 리샘플과 참조 맵 — 룩어헤드가 막히는 유일한 지점.

**절대 규약**: 타임스탬프는 오픈 타임이다. HTF 봉 ``[10:00, 11:00)`` 은 11:00 에
확정되므로 ``ts >= 11:00`` 인 5m 봉부터 참조할 수 있다. ``close_ts`` 컬럼이
그 확정 시각이고, `build_ref_map` 이 이를 강제한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["TF_ALIASES", "resample_htf", "build_ref_map"]

#: pandas 오프셋 별칭. "15m" 은 pandas 가 모른다 — "15min" 이어야 한다.
TF_ALIASES = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1D", "1w": "1W",
}


def _rule(tf: str) -> str:
    return TF_ALIASES.get(tf.strip().lower(), tf)


def resample_htf(df: pd.DataFrame, tf: str, *, min_ltf_bars: int = -1) -> pd.DataFrame:
    """5m -> HTF. ``close_ts`` 와 ``n_ltf`` 를 붙인다.

    Parameters
    ----------
    min_ltf_bars
        구성 5m 봉이 이보다 적은 HTF 봉은 버린다(주말 갭·상장 첫날 등).
        ``-1`` 이면 **관측된 ``n_ltf`` 의 중앙값**을 쓴다.
        예전에는 ``pd.Timedelta(step)`` 으로 기대치를 계산했는데,
        ``pd.tseries.offsets.Day`` 는 Timedelta 로 안 바뀌어 "1d" 에서 터졌다.
    """
    rule = _rule(tf)
    g = df.resample(rule, closed="left", label="left", origin="epoch")
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
    out["n_ltf"] = g.size()
    out = out[out["n_ltf"] > 0].copy()

    # 확정 시각 = 다음 구간의 시작
    idx = out.index
    nxt = idx.to_series().shift(-1)
    step = pd.Series(idx).diff().median()
    nxt = nxt.fillna(idx[-1] + step)
    out["close_ts"] = pd.DatetimeIndex(nxt).as_unit(df.index.unit)

    need = int(np.median(out["n_ltf"])) if min_ltf_bars < 0 else int(min_ltf_bars)
    if need > 1:
        out = out[out["n_ltf"] >= need]
    return out


def build_ref_map(ltf_index: pd.DatetimeIndex, htf: pd.DataFrame) -> np.ndarray:
    """각 5m 봉이 참조할 수 있는 **마지막으로 확정된** HTF 봉의 위치.

    확정 전이면 −1. ``searchsorted(..., "right") - 1`` 이 곧 "close_ts 가
    이 시각 이하인 마지막 HTF 봉"이다.
    """
    ct = htf["close_ts"].to_numpy()
    pos = np.searchsorted(ct, ltf_index.to_numpy(), side="right") - 1
    return pos.astype(np.int64)
