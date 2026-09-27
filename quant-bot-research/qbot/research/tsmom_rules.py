"""시계열 추세에 얹는 규칙 — `claude/시계열추세-뒤집기보류-ATR-사전등록.md`.

뒤집기 보류: 28일 신호가 보유와 반대로 바뀌어도, 이번 주 보유 방향 불리폭 < k·ATR(주봉 14주) 이면 유지.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["weekly_atr", "hold_positions", "book_from_positions"]

FEE = 6e-4
FUND_WK = 0.137 * 3 * 7 / 1e4


def weekly_atr(W: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """주봉(열 = 코인, 행 = 주 시작 월요일)의 high/low/close 로 ATR 단순평균. 그 주 종가까지 포함(인과)."""
    H, L, C = W["high"], W["low"], W["close"]
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    tr = tr.where(pc.notna(), H - L)
    return tr.rolling(period, min_periods=period).mean()


def hold_positions(sig: pd.DataFrame, chg: pd.DataFrame, atr: pd.DataFrame, k: float | None) -> pd.DataFrame:
    """주별 신호 → 보류 규칙을 적용한 포지션(±1).

    sig: 그 주 일요일 종가 기준 28일 부호 · chg: 이번 주 종가 − 지난주 종가(가격) · atr: 그 주 ATR(가격).
    k None 이면 신호 그대로(기준). 신호가 정확히 0 이면 쉰다(0) — 기준(legacy)과 같은 처리. 쉬던 다음 주는 신호 그대로.
    """
    out = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)
    for c in sig.columns:
        s = sig[c].to_numpy(float)
        d = chg[c].reindex(sig.index).to_numpy(float)
        a = atr[c].reindex(sig.index).to_numpy(float)
        p = np.full(len(s), np.nan)
        prev = np.nan
        for i in range(len(s)):
            si = s[i]
            if not np.isfinite(si):
                prev = np.nan
                continue
            if si == 0:
                cur = 0.0
            elif not np.isfinite(prev) or prev == 0 or si == prev:
                cur = si
            elif k is None or not (np.isfinite(a[i]) and np.isfinite(d[i])):
                cur = si
            else:
                adverse = -d[i] if prev > 0 else d[i]          # 보유 방향으로 불리하게 움직인 폭
                cur = prev if adverse < k * a[i] else si
            p[i] = cur
            prev = cur
        out[c] = p
    return out


def book_from_positions(pos: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """포지션 → 주별 동일가중 순수익(수수료 변화분 6bps · 펀딩 롱 지급)."""
    pos = pos.where(fwd.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    gross = pos * fwd
    net = gross - dpos * FEE - pos * FUND_WK
    n = pos.notna().sum(axis=1)
    keep = n > 0
    return pd.DataFrame(dict(ret=net.mean(axis=1), gross=gross.mean(axis=1),
                             fee=(dpos * FEE).where(pos.notna()).mean(axis=1),
                             turnover=dpos.where(pos.notna()).mean(axis=1)))[keep]
