# scoring.py  v2.0
# ─────────────────────────────────────────────────────────────────
# 포지션 크기 계산 원칙:
#   - 순수 리스크 기반: 거래당 손실이 잔고의 RISK_PER_TRADE를 넘지 않음
#   - scoring(점수)은 size_multiplier로만 반영 (signals.py에서 이미 계산)
#   - 최대 노출가치 캡: 잔고의 MAX_NOTIONAL_RATIO배
#   - compute_total_score는 호환성 유지 (backtest.py가 호출하므로)
#     단 position_pct는 더 이상 포지션 크기에 직접 영향 안 줌
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

# ── 리스크 파라미터
RISK_PER_TRADE     = 0.015   # 거래당 최대 손실 = 잔고의 1.5% (1.5× 증가)
MAX_NOTIONAL_RATIO = 2.0     # 최대 노출가치 = 잔고의 200%
MIN_NOTIONAL       = 10.0    # 최소 노출가치 (너무 작은 포지션 방지)


@dataclass
class ScoreBreakdown:
    base: float
    pd: float
    trend: float
    poi: float
    total: float
    position_pct: float      # 호환성용 — 포지션 크기에 직접 사용 안 함


def pd_score_from_pos(pd_pos: float, direction: str) -> float:
    if pd_pos is None:
        return 0.0
    x = (1.0 - float(pd_pos)) if direction == "long" else float(pd_pos)
    return round(3.0 * max(0.0, min(1.0, x)), 2)


def trend_match_score(htf_state: str, direction: str) -> float:
    if htf_state == "bull" and direction == "long":  return 3.0
    if htf_state == "bear" and direction == "short": return 3.0
    return 0.0


def is_price_in_zone(px: float, zone_low: float, zone_high: float) -> bool:
    return min(zone_low, zone_high) <= px <= max(zone_low, zone_high)


def poi_score(direction: str, htf_state: str, px: float,
              ob_zone: Optional[Dict] = None,
              fvg_zone: Optional[Dict] = None) -> float:
    if trend_match_score(htf_state, direction) <= 0:
        return 0.0
    in_ob  = ob_zone  is not None and is_price_in_zone(px, float(ob_zone["low"]),  float(ob_zone["high"]))
    in_fvg = fvg_zone is not None and is_price_in_zone(px, float(fvg_zone["low"]), float(fvg_zone["high"]))
    return 4.0 if (in_ob or in_fvg) else 0.0


def compute_total_score(direction: str, pd_pos: Optional[float],
                        htf_state: str, px: float,
                        ob_zone_1h: Optional[Dict] = None,
                        fvg_zone_1h: Optional[Dict] = None,
                        base_score: float = 3.0) -> ScoreBreakdown:
    base  = float(base_score)
    pd_s  = pd_score_from_pos(pd_pos, direction) if pd_pos is not None else 0.0
    tr_s  = trend_match_score(htf_state, direction)
    poi_s = poi_score(direction, htf_state, px, ob_zone_1h, fvg_zone_1h)
    total = round(base + pd_s + tr_s + poi_s, 2)
    # position_pct는 호환성용으로만 남김 — calc_risk_capped_position_value에서 무시됨
    return ScoreBreakdown(base=base, pd=pd_s, trend=tr_s, poi=poi_s,
                          total=total, position_pct=total)


def calc_risk_capped_position_value(
    equity_usdt: float,
    position_pct: float,         # 호환성 파라미터 — 더 이상 사용 안 함
    entry: float,
    sl: float,
    leverage: int = 30,
    max_risk_frac: float = 0.05, # 호환성 파라미터 — RISK_PER_TRADE로 대체
    size_multiplier: float = 1.0,
    taker_fee_rate: float = 0.0005,
) -> float:
    """
    리스크 기반 포지션 노출가치 계산.

    공식:
      delta = |entry - sl| / entry
      notional = (equity × RISK_PER_TRADE × size_multiplier) / (delta + fee×2)

    적용 순서:
      1. 리스크 기반 notional 계산
      2. 최대 노출가치 캡 (잔고의 MAX_NOTIONAL_RATIO배) 적용
      3. 최소 노출가치 보장

    반환: 포지션 노출가치 (USDT)
          수량 = 노출가치 / entry
    """
    equity = float(equity_usdt)
    entry  = float(entry)
    sl     = float(sl)
    sm     = float(size_multiplier)

    if equity <= 0 or entry <= 0:
        return 0.0

    delta = abs(entry - sl) / entry
    if delta <= 1e-9:
        return 0.0

    fee_both = float(taker_fee_rate) * 2.0

    # 리스크 기반 notional
    notional = (equity * RISK_PER_TRADE * sm) / (delta + fee_both)

    # 최대 노출 캡
    max_notional = equity * MAX_NOTIONAL_RATIO
    notional = min(notional, max_notional)

    # 최소 보장
    notional = max(notional, MIN_NOTIONAL)

    return round(notional, 2)