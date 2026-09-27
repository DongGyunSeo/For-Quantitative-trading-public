"""체결 모델 — 지정가 진입 · 지정가 익절 · 스톱 시장가 손절.

이 프로젝트의 표준 셋업(2026-09-18 확정):

    진입   직전 봉 종가에 **지정가**(maker).  ``fill_window`` 봉 안에 닿으면 체결,
           아니면 폐기. 갭으로 더 좋은 가격에 체결될 수 있어도 **지정가로 체결**된
           것으로 본다(보수적).
    익절   진입가 ± ``tp_r`` × 1R 에 **지정가**(maker)
    손절   진입가 ∓ ``sl_r`` × 1R 에 **스톱 시장가**(taker)

비용은 R 단위로 환산한다::

    fee_R = (진입bps + 청산bps) / risk_bps

청산 bps 는 승리면 maker, 패배·미해결이면 taker 다. 미해결도 결국 시장가로
닫아야 하므로 taker 를 문다.

주의 — 세 가지
--------------
1. **체결봉을 레이스에서 빼지 않는다.** 빼면 가장 빠른 손실이 삭제돼 낙관 편향이
   생긴다(원본 FVG 연구의 교훈). 대신 측정해상도 규칙으로 1R 을 봉보다 크게 만든다.
2. **같은 봉에서 TP·SL 이 모두 걸리면 모호**로 보고 0 R 로 처리한다.
3. **미체결은 거래가 아니다.** 체결률을 반드시 함께 보고한다 — 체결률이 낮으면
   남은 표본이 특정 국면에 쏠린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ExecConfig", "limit_entry_trades", "timed_exit_trades", "summarize_trades"]


@dataclass(frozen=True, slots=True)
class ExecConfig:
    #: 진입 지정가 수수료 (maker)
    entry_bps: float = 2.0
    #: 익절 지정가 수수료 (maker)
    tp_bps: float = 2.0
    #: 손절 시장가 수수료 (taker 5 + 슬리피지 1)
    sl_bps: float = 6.0
    #: 지정가 유효 봉 수
    fill_window: int = 12
    tp_r: float = 2.0
    sl_r: float = 1.0
    horizon: int = 576          # 48시간
    #: 체결봉도 레이스에 포함 (보수적). False 로 두지 말 것.
    include_fill_bar: bool = True


def limit_entry_trades(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bars: np.ndarray,
    side: np.ndarray,
    risk: np.ndarray,
    cfg: ExecConfig = ExecConfig(),
    limit_px: np.ndarray | None = None,
) -> pd.DataFrame:
    """신호봉 목록 -> 체결·결과.

    Parameters
    ----------
    bars
        **신호봉** 인덱스. 지정가는 이 봉의 종가에 걸고 다음 봉부터 유효하다.
    limit_px
        지정가를 직접 줄 때. 기본은 신호봉 종가.

    Returns
    -------
    ``filled`` · ``fill_bar`` · ``entry`` · ``outcome`` · ``r_gross`` ·
    ``fee_r`` · ``r_net`` · ``bars_held``
    """
    n = len(high)
    m = len(bars)
    lim = close[bars] if limit_px is None else np.asarray(limit_px, float)

    filled = np.zeros(m, bool)
    fill_bar = np.full(m, -1, np.int64)
    outcome = np.full(m, "미체결", dtype=object)
    r_gross = np.zeros(m)
    fee_r = np.zeros(m)
    held = np.full(m, -1, np.int64)

    for i in range(m):
        b = int(bars[i])
        sd = int(side[i])
        rk = float(risk[i])
        L = float(lim[i])
        if sd == 0 or not np.isfinite(rk) or rk <= 0 or not np.isfinite(L):
            continue
        a0 = b + 1
        a1 = min(n, a0 + cfg.fill_window)
        if a0 >= a1:
            continue
        # 롱 지정가는 저가가 내려와야 체결, 숏은 고가가 올라와야 체결
        hit = (low[a0:a1] <= L) if sd > 0 else (high[a0:a1] >= L)
        if not hit.any():
            continue
        k = a0 + int(hit.argmax())
        filled[i] = True
        fill_bar[i] = k

        s = k if cfg.include_fill_bar else k + 1
        e = min(n, s + cfg.horizon)
        if s >= e:
            outcome[i] = "미해결"
            fee_r[i] = (cfg.entry_bps + cfg.sl_bps) / (rk / L * 1e4)
            continue
        seg_hi = np.maximum.accumulate(high[s:e])
        seg_lo = np.minimum.accumulate(low[s:e])
        tp_px = L + sd * cfg.tp_r * rk
        sl_px = L - sd * cfg.sl_r * rk
        span = e - s
        if sd > 0:
            i_tp = np.searchsorted(seg_hi, tp_px - 1e-12, "left")
            i_sl = np.searchsorted(-seg_lo, -sl_px - 1e-12, "left")
        else:
            i_tp = np.searchsorted(-seg_lo, -tp_px - 1e-12, "left")
            i_sl = np.searchsorted(seg_hi, sl_px - 1e-12, "left")

        risk_bps = rk / L * 1e4
        if i_tp >= span and i_sl >= span:
            outcome[i] = "미해결"
            exit_bps = cfg.sl_bps
            held[i] = span
        elif i_tp == i_sl:
            outcome[i] = "모호"
            exit_bps = cfg.sl_bps
            held[i] = i_tp
        elif i_tp < i_sl:
            outcome[i] = "승"
            r_gross[i] = cfg.tp_r
            exit_bps = cfg.tp_bps
            held[i] = i_tp
        else:
            outcome[i] = "패"
            r_gross[i] = -cfg.sl_r
            exit_bps = cfg.sl_bps
            held[i] = i_sl
        fee_r[i] = (cfg.entry_bps + exit_bps) / risk_bps

    return pd.DataFrame({
        "bar": np.asarray(bars, np.int64), "side": np.asarray(side, np.int8),
        "filled": filled, "fill_bar": fill_bar, "entry": lim,
        "outcome": outcome, "r_gross": r_gross, "fee_r": fee_r,
        "r_net": np.where(filled, r_gross - fee_r, 0.0),
        "bars_held": held,
    })


def summarize_trades(t: pd.DataFrame) -> dict:
    """체결된 거래만 집계. 체결률을 반드시 함께 낸다."""
    n = len(t)
    f = t[t.filled]
    if len(f) == 0:
        return {"n": n, "체결률": 0.0}
    vc = f.outcome.value_counts()
    return {
        "n": n, "체결": len(f), "체결률": len(f) / n,
        "승": int(vc.get("승", 0)), "패": int(vc.get("패", 0)),
        "모호": int(vc.get("모호", 0)), "미해결": int(vc.get("미해결", 0)),
        "승률": float(vc.get("승", 0) / max(vc.get("승", 0) + vc.get("패", 0), 1)),
        "gross": float(f.r_gross.mean()), "수수료R": float(f.fee_r.mean()),
        "net": float(f.r_net.mean()),
        "중앙보유": float(f.loc[f.bars_held >= 0, "bars_held"].median())
        if (f.bars_held >= 0).any() else np.nan,
    }


def timed_exit_trades(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bars: np.ndarray,
    side: np.ndarray,
    hold: int,
    cfg: ExecConfig = ExecConfig(),
    limit_px: np.ndarray | None = None,
    risk: np.ndarray | None = None,
) -> pd.DataFrame:
    """지정가 진입 + **시간 청산**. 배리어가 없다.

    배리어 셋업은 1R 을 봉보다 크게 잡아야 측정이 되는데(측정해상도 규칙),
    잡으려는 드리프트가 그보다 작으면 승부가 잡음으로 결정된다. 그럴 때는
    배리어를 아예 빼고 **드리프트를 bps 로 직접** 재는 편이 정직하다.

    청산은 시각 기준이라 시장가(taker)다. 비용 = 진입 maker + 청산 taker.
    ``risk`` 를 주면 R 단위 손익도 함께 낸다.
    """
    n = len(high)
    m = len(bars)
    lim = close[bars] if limit_px is None else np.asarray(limit_px, float)
    filled = np.zeros(m, bool)
    fill_bar = np.full(m, -1, np.int64)
    gross_bps = np.zeros(m)

    for i in range(m):
        b, sd, L = int(bars[i]), int(side[i]), float(lim[i])
        if sd == 0 or not np.isfinite(L):
            continue
        a0, a1 = b + 1, min(n, b + 1 + cfg.fill_window)
        if a0 >= a1:
            continue
        hit = (low[a0:a1] <= L) if sd > 0 else (high[a0:a1] >= L)
        if not hit.any():
            continue
        k = a0 + int(hit.argmax())
        j = k + hold
        if j >= n:
            continue
        filled[i] = True
        fill_bar[i] = k
        gross_bps[i] = sd * (close[j] - L) / L * 1e4

    fee_bps = cfg.entry_bps + cfg.sl_bps        # 시간 청산은 시장가
    net_bps = np.where(filled, gross_bps - fee_bps, 0.0)
    out = pd.DataFrame({
        "bar": np.asarray(bars, np.int64), "side": np.asarray(side, np.int8),
        "filled": filled, "fill_bar": fill_bar, "entry": lim,
        "gross_bps": gross_bps, "fee_bps": np.where(filled, fee_bps, 0.0),
        "net_bps": net_bps,
    })
    if risk is not None:
        rb = np.asarray(risk, float) / lim * 1e4
        out["risk_bps"] = rb
        out["r_gross"] = np.where(filled & (rb > 0), gross_bps / rb, 0.0)
        out["r_net"] = np.where(filled & (rb > 0), net_bps / rb, 0.0)
    return out
