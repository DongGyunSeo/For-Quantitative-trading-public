# liquidity.py  v3.1  —  세션 기반 유동성 레벨 시스템
# ─────────────────────────────────────────────────────────────────
# 세션 정의 (UTC):
#   Asia:       00:00~05:00  → EQL/EQH 업데이트, 진입 X
#   Pre-London: 05:00~07:00  → 진입 O
#   London:     07:00~10:00  → London H/L 생성(10:00), 진입 O
#   Pre-NY:     10:00~12:00  → 진입 O
#   New York:   12:00~20:00  → NY H/L 생성(20:00), 진입 O
#   NY After:   20:00~24:00  → 진입 X
#
# 레벨 종류 & 강도:
#   PWH/PWL:           3.0  (지난 완성 주 월~금)
#   PDH/PDL:           2.5  (어제 00:00~24:00)
#   EQH/EQL:           2.0  (5영업일 15m pivot, 1H마다 갱신)
#   Asia H/L:          2.0  (당일 아시아 세션)
#   London H/L:        2.0  (당일 런던 세션)
#   NY H/L:            2.0
#
# 레벨 생명주기:
#   EQL/EQH: sweep → S/R flip, 타임아웃 → 삭제
#   PDH/PDL/PWH/PWL/Session H/L: 유효기간 내 무조건 유지
#   겹침(0.1% 이내) → 강도 합산(max 5.0), 대표가격 = 강도 높은 것
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from swings import find_swings
from timeframes import resample_5m_to_15m_1h

_LONDON_TZ = ZoneInfo("Europe/London")
_NY_TZ     = ZoneInfo("America/New_York")
_UTC_TZ    = ZoneInfo("UTC")

# ── 세션 경계 (UTC 시) — 서머타임 미적용 기본값 (런던 겨울 기준)
ASIA_START       = 0
ASIA_END         = 5
DAY_END          = 24   # PDH/PDL 유효 종료 (00:00~24:00)

def _get_session_hours(ts: pd.Timestamp) -> dict:
    """
    런던 + 뉴욕 서머타임을 반영한 세션 경계 (UTC 시) 반환.
    런던: Europe/London (GMT=UTC+0, BST=UTC+1)
    뉴욕: America/New_York (EST=UTC-5, EDT=UTC-4)
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    dt = ts.to_pydatetime()

    lo = int(dt.astimezone(_LONDON_TZ).utcoffset().total_seconds() / 3600)
    no = int(dt.astimezone(_NY_TZ).utcoffset().total_seconds() / 3600)

    # 런던 로컬 시각 기준:
    #   London open:  07:00 local → UTC = 07:00 - lo
    #   London close: 10:00 local → UTC = 10:00 - lo
    #   겨울(GMT, lo=0): UTC 07~10
    #   여름(BST, lo=1): UTC 06~09
    london_open  = 7  - lo
    london_close = 10 - lo
    # 뉴욕 로컬 시각: 07:00 ~ 15:00 (EST 기준)
    # 겨울(no=-5): 7-(-5)=12, 15-(-5)=20 → UTC 12~20
    # 여름(no=-4): 7-(-4)=11, 15-(-4)=19 → UTC 11~19
    ny_open      = 7 - no
    ny_close_raw = 15 - no
    # 24 이상이면 다음날로 넘어감 → 당일 기준 ny_after 없음 (24로 설정)
    ny_close     = min(ny_close_raw, 24)

    return {
        "asia_start":    ASIA_START,
        "asia_end":      ASIA_END,
        "pre_lon_end":   london_open,
        "london_start":  london_open,
        "london_end":    london_close,
        "pre_ny_start":  london_close,
        "ny_start":      ny_open,
        "ny_end":        ny_close,   # 겨울=20, 여름=19
        "day_end":       DAY_END,
    }

# 기본값 (겨울 기준, 동적 계산 전 초기값용)
LONDON_START     = 7
LONDON_END       = 10
NY_START         = 12
NY_END           = 20

# ── 레벨 강도
STRENGTH = {
    "pwh": 3.0, "pwl": 3.0,
    "pdh": 2.5, "pdl": 2.5,
    "eqh": 2.0, "eql": 2.0,
    "asia_high": 2.0, "asia_low": 2.0,
    "london_high": 2.0, "london_low": 2.0,
    "idm_high": 2.0, "idm_low": 2.0,         # ← IDM 추가
}
MAX_STRENGTH     = 5.0

# ── Asia range 필터
ASIA_RANGE_ATR_MULT = 1.0   # Asia range < ATR × 1.0 이면 H/L 레벨 생성 안 함
ASIA_ATR_LOOKBACK   = 20    # ATR lookback (5m봉)

def _calc_atr(df_5m: pd.DataFrame, lookback: int = ASIA_ATR_LOOKBACK) -> float:
    """True Range 기반 ATR. 데이터 부족 시 0.0 반환."""
    if df_5m is None or len(df_5m) < lookback + 1:
        return 0.0
    tail = df_5m.iloc[-(lookback+1):].copy()
    hi = tail["high"].to_numpy(dtype=float)
    lo = tail["low"].to_numpy(dtype=float)
    cl = tail["close"].to_numpy(dtype=float)
    tr = [max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
          for i in range(1, len(hi))]
    return float(sum(tr)/len(tr)) if tr else 0.0

# ── 파라미터
EQ_BAND_PCT      = 0.003   # (deprecated, fallback only) EQL/EQH zone ±0.3%
MERGE_PCT        = 0.001   # 레벨 병합 기준 0.1%
MIN_TOUCHES_EQ   = 2       # EQL/EQH 최소 터치

# ── EQH/EQL DBSCAN 파라미터
EQ_EPS_ATR_RATIO    = 0.15  # ε = ATR × 0.15 (변동성 적응형)
EQ_MAX_SPREAD_RATIO = 1.5   # cluster diameter > ε × 1.5 시 chain으로 간주, median 재축약
EQ_FALLBACK_PCT     = 0.003 # ATR 미가용 시 fallback ε = price × 0.3%


# ══════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════

def _is_weekday(ts: pd.Timestamp) -> bool:
    return ts.weekday() < 5  # 0=월, 4=금


def _last_n_business_days(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """ts 기준 n 영업일 전 날짜 반환"""
    count = 0
    cur = ts.normalize()
    while count < n:
        cur -= pd.Timedelta(days=1)
        if _is_weekday(cur):
            count += 1
    return cur


def _session_of(ts: pd.Timestamp) -> str:
    """서머타임 반영한 세션 분류"""
    h  = ts.hour
    sh = _get_session_hours(ts)
    if sh["asia_start"] <= h < sh["asia_end"]:
        return "asia"
    elif h < sh["london_start"]:
        return "pre_london"
    elif h < sh["london_end"]:
        return "london"
    elif h < sh["ny_start"]:
        return "pre_ny"
    elif h < sh["ny_end"]:
        return "newyork"
    else:
        return "ny_after"


def _can_enter(ts: pd.Timestamp) -> bool:
    """진입 가능 세션 여부 (아시아, NY After 제외)"""
    s = _session_of(ts)
    return s not in ("asia", "ny_after")


def _make_zone(price: float, pct: float) -> Tuple[float, float]:
    band = abs(price) * pct
    return price - band, price + band


def _level(level_type: str, price: float, zone_pct: float,
           ts_start: pd.Timestamp, ts_end: Optional[pd.Timestamp] = None,
           flipped: bool = False, touches: int = 1,
           slope: float = 0.0, intercept: float = 0.0) -> dict:
    zl, zh = _make_zone(price, zone_pct)
    strength = STRENGTH.get(level_type, 2.0)
    return {
        "level_type":    level_type,
        "price_center":  price,
        "zone_low":      zl,
        "zone_high":     zh,
        "zone_pct":      zone_pct,
        "strength":      strength,
        "touches":       touches,
        "ts_start":      ts_start,
        "ts_end":        ts_end,       # None = 영구
        "flipped":       flipped,
        "slope":         slope,
        "intercept":     intercept,
        "swept":         False,
    }


# ══════════════════════════════════════════════════════════════════
# S/R flip 확인 조건
# ══════════════════════════════════════════════════════════════════

def _sr_flip_confirmed(lv: dict, df_5m: pd.DataFrame,
                       vol_mult: float = 1.5, body_mult: float = 1.5,
                       lookback: int = 120) -> bool:
    """
    S/R flip 확인 조건 (모두 충족해야 flip):
      - close가 레벨을 완전히 돌파 (high레벨: close > zone_high, low레벨: close < zone_low)
      - volume > 직전 lookback봉 중위값 × vol_mult
      - body_size(|close-open|) > 직전 lookback봉 평균 body × body_mult
    데이터 부족 시 True 반환 (통과)
    """
    if df_5m is None or df_5m.empty:
        return True

    required = {"close", "open"}
    if not required.issubset(df_5m.columns):
        return True

    n = len(df_5m)
    if n < 5:
        return True

    last = df_5m.iloc[-1]
    c    = float(last["close"])
    o    = float(last["open"])
    body = abs(c - o)

    zh = float(lv.get("zone_high", lv.get("price_center", 0)))
    zl = float(lv.get("zone_low",  lv.get("price_center", 0)))
    lt = lv.get("level_type", "")

    # 방향 확인
    if lt in ("eqh", "pdh", "pwh", "asia_high", "london_high"):
        if c <= zh:
            return False
    elif lt in ("eql", "pdl", "pwl", "asia_low", "london_low"):
        if c >= zl:
            return False

    start = max(0, n - lookback - 1)
    hist  = df_5m.iloc[start:-1]  # 현재 봉 제외

    if len(hist) < 10:
        return True

    # 거래량 조건
    if "volume" in df_5m.columns and "volume" in hist.columns:
        vol = float(last.get("volume", 0))
        vol_median = float(hist["volume"].median())
        if vol_median > 0 and vol < vol_median * vol_mult:
            return False

    # body 조건
    if "close" in hist.columns and "open" in hist.columns:
        avg_body = float((hist["close"] - hist["open"]).abs().mean())
        if avg_body > 0 and body < avg_body * body_mult:
            return False

    return True


# ══════════════════════════════════════════════════════════════════
# (1) EQL / EQH
# ══════════════════════════════════════════════════════════════════

def build_eq_levels(pivots: pd.DataFrame,
                    kind: str,
                    atr: float = 0.0,
                    eps_atr_ratio: float = EQ_EPS_ATR_RATIO,
                    max_spread_ratio: float = EQ_MAX_SPREAD_RATIO,
                    min_touches: int = MIN_TOUCHES_EQ,
                    fallback_pct: float = EQ_FALLBACK_PCT) -> List[dict]:
    """
    DBSCAN-style 밀도 기반 EQH/EQL 감지.

    핵심 설계:
      1. ε = ATR × eps_atr_ratio (변동성 적응형)
         ATR=0이면 fallback_pct × median(prices) 사용
      2. DBSCAN density-reachable 클러스터링 (drift 없음)
      3. cluster diameter > ε × max_spread_ratio 시 → median ± ε로 재축약
         (chain 효과 차단)
      4. center = median (mean의 outlier 영향 제거)
      5. **Sweep 트리거 = cluster extreme**:
         EQH → zone_high = max(member_prices)
         EQL → zone_low  = min(member_prices)
         반대편 zone은 멤버 분포의 hull (분포 정보 보존, 거리 계산용)

    Parameters
    ----------
    pivots : DataFrame  (columns: kind, price, idx, time)
    kind   : 'H' → eqh, 'L' → eql
    atr    : 5m ATR (가격 단위). 0이면 fallback 사용.
    """
    level_type = "eqh" if kind == "H" else "eql"
    p = pivots[pivots["kind"].astype(str) == kind].copy()
    if len(p) < min_touches:
        return []

    p = p.sort_values("idx").reset_index(drop=True)
    prices = p["price"].to_numpy(dtype=float)
    times  = p["time"].values if "time" in p.columns else [None] * len(p)
    n      = len(p)

    # ── ε 결정: ATR 기반 우선, 없으면 percentage fallback
    if atr > 0:
        eps = atr * eps_atr_ratio
    else:
        eps = float(np.median(prices)) * fallback_pct

    if eps <= 0:
        return []

    # ── DBSCAN (1D): density-reachable 클러스터링
    visited = [False] * n
    cluster_id = [-1] * n  # -1 = noise (cluster 미할당)
    cur_cid = 0

    def neighbors(i: int) -> List[int]:
        """i의 ε-이웃 인덱스 (자기 자신 포함)"""
        return [j for j in range(n) if abs(prices[j] - prices[i]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if len(nb) < min_touches:
            continue  # core point 아님 → noise

        cluster_id[i] = cur_cid
        # density-reachable 확장 (BFS)
        seed = list(nb)
        k = 0
        while k < len(seed):
            j = seed[k]
            if not visited[j]:
                visited[j] = True
                nb_j = neighbors(j)
                if len(nb_j) >= min_touches:
                    for x in nb_j:
                        if x not in seed:
                            seed.append(x)
            if cluster_id[j] == -1:
                cluster_id[j] = cur_cid
            k += 1
        cur_cid += 1

    # ── 클러스터별 검증 + 레벨 생성
    levels = []
    for cid in range(cur_cid):
        members = [i for i in range(n) if cluster_id[i] == cid]
        if len(members) < min_touches:
            continue

        member_prices = prices[members]
        spread = float(member_prices.max() - member_prices.min())

        # ── chain 차단: spread가 ε×ratio를 넘으면 median ± ε로 재축약
        if spread > eps * max_spread_ratio:
            med = float(np.median(member_prices))
            tight = [i for i in members if abs(prices[i] - med) <= eps]
            if len(tight) < min_touches:
                continue
            members = tight
            member_prices = prices[members]

        # ── center = median (drift-free, outlier-robust)
        center = float(np.median(member_prices))

        # ── Sweep 트리거 = cluster extreme (의미적 핵심)
        cluster_max = float(member_prices.max())
        cluster_min = float(member_prices.min())

        if level_type == "eqh":
            # EQH: 가격이 cluster_max를 넘어야 sweep
            zone_high = cluster_max
            zone_low  = cluster_min  # 분포 하한 (거리 계산용 정보)
        else:  # eql
            # EQL: 가격이 cluster_min 아래로 가야 sweep
            zone_low  = cluster_min
            zone_high = cluster_max  # 분포 상한 (거리 계산용 정보)

        # ── ts 정보
        # members는 idx 정렬되어 있어 첫/마지막이 시간 순서대로 매핑됨
        members_sorted_by_idx = sorted(members)
        first_t = times[members_sorted_by_idx[0]]
        last_t  = times[members_sorted_by_idx[-1]]
        ts_start = pd.Timestamp(first_t) if first_t is not None else pd.Timestamp.now(tz="UTC")
        ts_last  = pd.Timestamp(last_t)  if last_t  is not None else ts_start
        ts_end   = ts_last + pd.Timedelta(days=7)

        # zone_pct: 호환성 유지 (effective band, center 기준)
        zone_pct_eff = (zone_high - zone_low) / (2 * center) if center > 0 else 0.0

        levels.append({
            "level_type":   level_type,
            "price_center": center,
            "zone_low":     zone_low,
            "zone_high":    zone_high,
            "zone_pct":     zone_pct_eff,
            "strength":     STRENGTH[level_type],
            "touches":      len(members),
            "ts_start":     ts_start,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })

    return levels


# ══════════════════════════════════════════════════════════════════
# (2) Session High/Low (Asia / London / NY)
# ══════════════════════════════════════════════════════════════════

def _extract_swing_liquidity(seg: pd.DataFrame,
                             lookback: int = 3,
                             neighbor_window: int = 0) -> dict:
    """세션/일/주 구간 seg에서 swing pivot 기반 유동성 추출.

    단순 high.max()/low.min()(신기루) 대신 swing pivot으로 확정된 것만.
    + 사후 침범 체크: 극값 pivot이 '마지막 pivot 이후~구간끝' 가격에 침범당했으면
      이미 소비된 유동성으로 보고 제외 (동적 추적의 사후 근사, 서동균 설계).

    반환:
      {
        'high': {'price': float, 'neighbors': int} or None,
        'low':  {'price': float, 'neighbors': int} or None,
      }
    """
    if seg is None or len(seg) < 7:  # find_swings 최소 (2*lookback+1)
        return {"high": None, "low": None}
    sw = find_swings(seg, lookback=lookback).reset_index(drop=True)
    highs = sw[sw["swing_high"]]
    lows  = sw[sw["swing_low"]]
    n = len(sw)
    out = {"high": None, "low": None}

    if len(highs) > 0:
        top_pos = highs["high"].idxmax()          # 극값 high pivot의 행 위치
        top_price = float(sw.at[top_pos, "high"])
        last_high_pos = highs.index.max()         # 마지막 high pivot 위치
        # 마지막 pivot 이후 ~ 구간끝 가격이 극값 high를 침범(>= top)했나
        after = sw.iloc[last_high_pos + 1:]
        violated = (len(after) > 0) and bool((after["high"] >= top_price).any())
        if not violated:
            out["high"] = {"price": top_price, "neighbors": int(len(highs))}

    if len(lows) > 0:
        bot_pos = lows["low"].idxmin()
        bot_price = float(sw.at[bot_pos, "low"])
        last_low_pos = lows.index.max()
        after = sw.iloc[last_low_pos + 1:]
        violated = (len(after) > 0) and bool((after["low"] <= bot_price).any())
        if not violated:
            out["low"] = {"price": bot_price, "neighbors": int(len(lows))}

    return out


def build_session_hl(df_5m: pd.DataFrame,
                     session: str,
                     date: pd.Timestamp,
                     asia_range_filter: bool = True) -> List[dict]:
    """
    특정 날짜의 세션 고저 레벨 생성.
    session: 'asia' | 'london' | 'newyork'
    date: 날짜 (timezone-aware UTC)
    asia_range_filter: Asia range < ATR×ASIA_RANGE_ATR_MULT 이면 빈 리스트 반환
    """
    # 서머타임 반영: 해당 날짜의 세션 경계 계산
    sh = _get_session_hours(date)
    session_map = {
        "asia":    (sh["asia_start"],   sh["asia_end"]),
        "london":  (sh["london_start"], sh["london_end"]),
        "newyork": (sh["ny_start"],     sh["ny_end"]),
    }
    if session not in session_map:
        return []

    h_start, h_end = session_map[session]
    day = date.normalize()
    t0  = day + pd.Timedelta(hours=h_start)
    t1  = day + pd.Timedelta(hours=h_end)

    # tz 불일치 방어: df_5m["time"] tz-naive이면 localize
    df_time = pd.to_datetime(df_5m["time"], utc=True)
    mask = (df_time >= t0) & (df_time < t1)
    seg  = df_5m[mask]
    if seg.empty:
        return []

    # ── swing pivot 기반 유동성 추출 (신기루 차단: 단순 극값 아닌 확정 swing)
    sw_liq = _extract_swing_liquidity(seg, lookback=3)
    high_info = sw_liq["high"]
    low_info  = sw_liq["low"]
    # swing이 하나도 없으면 그 방향 유동성 없음 (구간이 추세적이거나 너무 짧음)
    if high_info is None and low_info is None:
        return []
    high_price = high_info["price"] if high_info else None
    low_price  = low_info["price"]  if low_info  else None

    # Asia range 필터: 저변동성 날 가짜 sweep 방지 (swing 양쪽 있을 때만 의미)
    if session == "asia" and asia_range_filter and high_price is not None and low_price is not None:
        asia_range = high_price - low_price
        atr = _calc_atr(df_5m)
        if atr > 0 and asia_range < atr * ASIA_RANGE_ATR_MULT:
            return []   # range 너무 좁음 → H/L 레벨 생성 안 함

    hl_type = f"{session}_high"
    ll_type = f"{session}_low"

    # 유효기간
    ts_end_map = {
        "asia":    day + pd.Timedelta(hours=sh["london_end"]),  # 당일 런던 마감(10:00)까지
        "london":  day + pd.Timedelta(hours=sh["day_end"]),     # 당일 24:00까지 유지
        "newyork": day + pd.Timedelta(days=1, hours=4),         # 다음날 04:00까지 유지
    }
    ts_created = t1  # 세션 종료 시 생성
    ts_end     = ts_end_map[session]

    levels = []
    _specs = []
    if high_price is not None:
        _specs.append((high_price, hl_type, high_info["neighbors"]))
    if low_price is not None:
        _specs.append((low_price, ll_type, low_info["neighbors"]))
    for price, lt, neighbors in _specs:
        levels.append({
            "level_type":   lt,
            "price_center": price,
            "zone_low":     price,
            "zone_high":    price,
            "zone_pct":     0.0,
            "strength":     STRENGTH.get(lt, 2.0),
            "touches":      1,
            "pivot_neighbors": neighbors,   # 같은 방향 swing 개수 (raw)
            "ts_start":     ts_created,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })
    return levels


# ══════════════════════════════════════════════════════════════════
# (4) PDH / PDL
# ══════════════════════════════════════════════════════════════════

def build_pdhl(df_5m: pd.DataFrame, today: pd.Timestamp) -> List[dict]:
    """
    어제 UTC 00:00~24:00 고저.
    월요일이면 금요일 기준 PDH/PDL을 사용.
    주말(토~일)에 해당 가격을 돌파한 적 있으면 해당 레벨 삭제.
    """
    dow = today.weekday()  # 0=월
    if dow == 0:
        # 월요일 → 금요일 기준
        friday = today - pd.Timedelta(days=3)
    else:
        friday = today - pd.Timedelta(days=1)

    t0 = friday
    t1 = friday + pd.Timedelta(days=1)

    mask = (df_5m["time"] >= t0) & (df_5m["time"] < t1)
    seg  = df_5m[mask]
    if seg.empty:
        return []

    # ── 15m swing pivot 기반 (명세: previous day = 15m pivot)
    seg15, _ = resample_5m_to_15m_1h(seg, time_col="time", tz="UTC",
                                     drop_incomplete=True)
    sw_liq = _extract_swing_liquidity(seg15, lookback=3)
    pdh = sw_liq["high"]["price"] if sw_liq["high"] else None
    pdl = sw_liq["low"]["price"]  if sw_liq["low"]  else None
    pdh_nb = sw_liq["high"]["neighbors"] if sw_liq["high"] else 0
    pdl_nb = sw_liq["low"]["neighbors"]  if sw_liq["low"]  else 0
    if pdh is None and pdl is None:
        return []

    # 월요일: 주말(토~일) 동안 PDH/PDL을 돌파했는지 확인
    weekend_swept_high = False
    weekend_swept_low  = False
    if dow == 0:
        sat_start = today - pd.Timedelta(days=2)
        wk_mask = (df_5m["time"] >= sat_start) & (df_5m["time"] < today)
        wk_seg  = df_5m[wk_mask]
        if not wk_seg.empty:
            if pdh is not None and float(wk_seg["high"].max()) > pdh:
                weekend_swept_high = True
            if pdl is not None and float(wk_seg["low"].min()) < pdl:
                weekend_swept_low = True

    ts_end = today + pd.Timedelta(hours=DAY_END)

    levels = []
    if pdh is not None and not weekend_swept_high:
        levels.append({
            "level_type":   "pdh",
            "price_center": pdh,
            "zone_low":     pdh,
            "zone_high":    pdh,
            "zone_pct":     0.0,
            "strength":     STRENGTH["pdh"],
            "touches":      1,
            "pivot_neighbors": pdh_nb,
            "ts_start":     today,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })
    if pdl is not None and not weekend_swept_low:
        levels.append({
            "level_type":   "pdl",
            "price_center": pdl,
            "zone_low":     pdl,
            "zone_high":    pdl,
            "zone_pct":     0.0,
            "strength":     STRENGTH["pdl"],
            "touches":      1,
            "pivot_neighbors": pdl_nb,
            "ts_start":     today,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })
    return levels


# ══════════════════════════════════════════════════════════════════
# (5) PWH / PWL
# ══════════════════════════════════════════════════════════════════

def build_pwhl(df_5m: pd.DataFrame, today: pd.Timestamp) -> List[dict]:
    """
    지난 완성 주(월~금) 고저.
    월요일 호출 시: 주말(토~일)에 터치된 레벨은 제외.
    """
    # 지난 월요일 찾기
    dow = today.weekday()  # 0=월
    # 지난 주 금요일
    last_fri = today - pd.Timedelta(days=dow + 3)
    last_mon = last_fri - pd.Timedelta(days=4)

    t0 = last_mon
    t1 = last_fri + pd.Timedelta(hours=21)  # 금요일 21:00

    mask = (df_5m["time"] >= t0) & (df_5m["time"] < t1)
    seg  = df_5m[mask]
    if seg.empty:
        return []

    # ── 1H swing pivot 기반 (명세: previous week = 1H pivot)
    _, seg1h = resample_5m_to_15m_1h(seg, time_col="time", tz="UTC",
                                     drop_incomplete=True)
    sw_liq = _extract_swing_liquidity(seg1h, lookback=3)
    pwh = sw_liq["high"]["price"] if sw_liq["high"] else None
    pwl = sw_liq["low"]["price"]  if sw_liq["low"]  else None
    pwh_nb = sw_liq["high"]["neighbors"] if sw_liq["high"] else 0
    pwl_nb = sw_liq["low"]["neighbors"]  if sw_liq["low"]  else 0
    if pwh is None and pwl is None:
        return []

    # 주말(토~일) 동안 PWH/PWL 터치 확인
    weekend_swept_high = False
    weekend_swept_low  = False
    sat_start = last_fri + pd.Timedelta(days=1)  # 토요일
    wk_mask = (df_5m["time"] >= sat_start) & (df_5m["time"] < today)
    wk_seg  = df_5m[wk_mask]
    if not wk_seg.empty:
        if pwh is not None and float(wk_seg["high"].max()) > pwh:
            weekend_swept_high = True
        if pwl is not None and float(wk_seg["low"].min()) < pwl:
            weekend_swept_low = True

    # 이번 주 금요일 21:00까지 유효
    this_fri = today + pd.Timedelta(days=(4 - dow))
    ts_end   = this_fri + pd.Timedelta(hours=21)

    levels = []
    if pwh is not None and not weekend_swept_high:
        levels.append({
            "level_type":   "pwh",
            "price_center": pwh,
            "zone_low":     pwh,
            "zone_high":    pwh,
            "zone_pct":     0.0,
            "strength":     STRENGTH["pwh"],
            "touches":      1,
            "pivot_neighbors": pwh_nb,
            "ts_start":     today,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })
    if pwl is not None and not weekend_swept_low:
        levels.append({
            "level_type":   "pwl",
            "price_center": pwl,
            "zone_low":     pwl,
            "zone_high":    pwl,
            "zone_pct":     0.0,
            "strength":     STRENGTH["pwl"],
            "touches":      1,
            "pivot_neighbors": pwl_nb,
            "ts_start":     today,
            "ts_end":       ts_end,
            "flipped":      False,
            "slope":        0.0,
            "intercept":    0.0,
            "swept":        False,
        })
    return levels


# ══════════════════════════════════════════════════════════════════
# (6) 레벨 병합
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# (7) 레벨 병합
# ══════════════════════════════════════════════════════════════════

def merge_levels(levels: List[dict], merge_pct: float = MERGE_PCT) -> List[dict]:
    """
    0.1% 이내 겹치는 레벨 병합.
    대표가격 = 강도 높은 레벨의 가격.
    강도 합산 (max 5.0).
    level_type = 강도 높은 것.
    """
    if not levels:
        return []

    sorted_lvs = sorted(levels, key=lambda x: x["price_center"])
    merged = []
    used   = [False] * len(sorted_lvs)

    for i, lv in enumerate(sorted_lvs):
        if used[i]:
            continue
        group = [lv]
        for j in range(i + 1, len(sorted_lvs)):
            if used[j]:
                continue
            ref = sorted_lvs[j]["price_center"]
            if abs(ref - lv["price_center"]) / lv["price_center"] <= merge_pct:
                group.append(sorted_lvs[j])
                used[j] = True

        if len(group) == 1:
            merged.append(lv.copy())
        else:
            # 강도 가장 높은 것이 대표
            rep   = max(group, key=lambda x: x["strength"])
            total = min(sum(x["strength"] for x in group), MAX_STRENGTH)
            m     = rep.copy()
            m["strength"] = total
            m["touches"]  = sum(x["touches"] for x in group)
            merged.append(m)

        used[i] = True

    return merged


# ══════════════════════════════════════════════════════════════════
# (8) SessionLiquidityPool — 메인 관리 클래스
# ══════════════════════════════════════════════════════════════════

class SessionLiquidityPool:
    """
    세션 기반 유동성 레벨 풀.
    backtest의 IncrementalState에서 매 5m 봉마다 tick()을 호출.
    세션 전환 시 자동으로 레벨 갱신.
    """

    def __init__(self,
                 eq_band_pct: float   = EQ_BAND_PCT,
                 merge_pct: float     = MERGE_PCT,
                 min_touches_eq: int  = MIN_TOUCHES_EQ,
                 lookback_days: int   = 5):

        self.eq_band_pct    = eq_band_pct
        self.merge_pct      = merge_pct
        self.min_touches_eq = min_touches_eq
        self.lookback_days  = lookback_days

        # 레벨 저장소
        self._levels: List[dict] = []       # 활성 레벨
        self._flipped: List[dict] = []      # S/R flip 레벨

        # 세션 전환 추적
        self._last_session: Optional[str]        = None
        self._last_date:    Optional[pd.Timestamp] = None
        self._last_week:    Optional[pd.Timestamp] = None

        # 누적 데이터 (rolling window)
        self._df_5m_history: Optional[pd.DataFrame] = None

        # 진단 카운터
        self.stat_sweeps  = 0
        self.stat_flips   = 0
        self.stat_expired = 0
        self._bar_count           = 0   # 전체 봉 카운터
        self._flip_cooldown_bars  = 12  # flip 후 1h(12봉) 쿨다운

        # Asia/London H/L 지연 저장용
        self._pending_session_levels: List[dict] = []
        self._pending_session_cooldown: int = 0

    # ── 데이터 누적
    def feed_bar(self, bar: dict):
        """5m 봉 하나를 히스토리에 추가"""
        row = pd.DataFrame([bar])
        # time 컬럼 tz-aware UTC 보장 (mask 비교 오류 방지)
        if "time" in row.columns:
            row["time"] = pd.to_datetime(row["time"], utc=True)
        if self._df_5m_history is None:
            self._df_5m_history = row
        else:
            self._df_5m_history = pd.concat(
                [self._df_5m_history, row], ignore_index=True
            )
        # 최대 lookback_days * 2 + 7일치만 유지 (메모리 절약)
        keep_bars = (self.lookback_days * 2 + 7) * 24 * 12  # 5m bars
        if len(self._df_5m_history) > keep_bars:
            self._df_5m_history = self._df_5m_history.iloc[-keep_bars:].reset_index(drop=True)

    # ── 메인 tick
    def tick(self, bar: dict, pivots_15m: pd.DataFrame,
             current_bar_idx: float = 0,
             df_15m_struct: Optional[pd.DataFrame] = None,
             ) -> List[str]:
        """
        매 5m 봉마다 호출.
        반환: 변경 이벤트 로그 리스트
        """
        self._bar_count += 1
        self.feed_bar(bar)
        ts  = pd.Timestamp(bar["time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        notify = []
        today  = ts.normalize().tz_localize("UTC") if ts.tzinfo is None else ts.normalize()
        cur_session = _session_of(ts)

        # ── 세션 전환 감지
        if cur_session != self._last_session:
            notify += self._on_session_change(
                cur_session, self._last_session, ts, today, pivots_15m
            )
            self._last_session = cur_session

        # ── 날짜 전환 감지 (00:00)
        if self._last_date is None or today > self._last_date:
            notify += self._on_day_change(today)
            self._last_date = today

        # ── 주 전환 감지 (월요일 00:00)
        if ts.weekday() == 0 and (self._last_week is None or today > self._last_week):
            notify += self._on_week_change(today)
            self._last_week = today

        # ── 만료 레벨 처리
        notify += self._expire_levels(ts)

        # ── Asia/London H/L 지연 저장 (쿨다운 후 가격 필터 적용)
        if self._pending_session_cooldown > 0:
            self._pending_session_cooldown -= 1
            if self._pending_session_cooldown == 0 and self._pending_session_levels:
                _cur_px = float(bar.get("close", 0))
                for lv in self._pending_session_levels:
                    lt = lv["level_type"]
                    pc = lv["price_center"]
                    # 가격 필터: low 레벨은 현재가 위에 있을 때만, high 레벨은 현재가 아래에 있을 때만
                    if "low" in lt and _cur_px > pc:
                        self._levels.append(lv)
                        notify.append(f"[LIQ] 유동성 추가: {lt}({pc:,.1f})")
                    elif "high" in lt and _cur_px < pc:
                        self._levels.append(lv)
                        notify.append(f"[LIQ] 유동성 추가: {lt}({pc:,.1f})")
                    else:
                        notify.append(f"[LIQ] {lt} @ {pc:.1f} 가격 필터 탈락 (px={_cur_px:.1f})")
                self._pending_session_levels = []

        # ── 주말: 터치된 유동성 삭제 (거래는 안 하지만 레벨 정리)
        if ts.weekday() >= 5:  # 토/일
            notify += self._weekend_cleanup(bar)
        else:
            # ── IDM: 완전 제거 확정 (생성 자체 차단).
            #   BOS 못 만든 스윙(OB 없는 swing) = inducement. 전체 셋업의 54% 차지하나
            #   entry 신호로 약함(SL회피 56% < 강한레벨 63%), FVG 백킹으로도 정제 불가
            #   (단변량 0.6233, 구분 안 됨), trail은 설계상 무관. → entry 모집단서 제거.
            # (IDM 생성 블록 삭제)

            # ── sweep 감지 (진입 가능 세션만)
            if _can_enter(ts):
                notify += self._detect_sweeps(bar)

        return notify

    # ── 세션 전환 처리
    def _on_session_change(self, new_session: str, old_session: Optional[str],
                           ts: pd.Timestamp, today: pd.Timestamp,
                           pivots_15m: pd.DataFrame) -> List[str]:
        notify = []

        # 아시아 종료(05:00 pre_london) → EQL/EQH + Asia H/L 예약 (6봉 후 저장)
        if new_session == "pre_london":
            notify += self._rebuild_eq_levels(pivots_15m, today)
            if self._df_5m_history is not None:
                self._remove_by_type(["asia_high", "asia_low"])
                lv = build_session_hl(self._df_5m_history, "asia", today)
                if lv:
                    self._pending_session_levels = lv
                    self._pending_session_cooldown = 6  # 6봉 쿨다운
                    msg = " / ".join(f"{l['level_type']}({l['price_center']:.1f})" for l in lv)
                    notify.append(f"[LIQ] Asia H/L 예약(6봉 쿨다운): {msg}")

        # London H/L: 완전 제거 확정 (생성 자체 차단).
        #   crypto 24h 시장에서 변별 미미 + 노이즈. backtest 레벨 차단 → 근본 제거로 이관.
        # (London 생성 블록 삭제)

        # (v2.1: NY H/L 삭제 — IDM으로 대체)

        return notify

    # ── 날짜 전환 처리 (00:00)
    def _on_day_change(self, today: pd.Timestamp) -> List[str]:
        notify = []
        if self._df_5m_history is None:
            return notify

        # PDH/PDL 갱신
        self._remove_by_type(["pdh", "pdl"])
        lv = build_pdhl(self._df_5m_history, today)
        for l in lv:
            self._levels.append(l)
        if lv:
            for l in lv:
                notify.append(f"[LIQ] 유동성 추가: {l['level_type']}({l['price_center']:,.1f})")

        # Asia/London H/L: 00:00 시점엔 당일 아시아 세션이 아직 진행 중
        # → 전날 잔재 제거만 수행. 당일 H/L은 세션 종료 시 생성
        self._remove_by_type(["asia_high", "asia_low",
                              "london_high", "london_low"])

        return notify


    # ── 주 전환 처리 (월요일)
    def _on_week_change(self, today: pd.Timestamp) -> List[str]:
        notify = []
        if self._df_5m_history is None:
            return notify

        self._remove_by_type(["pwh", "pwl"])
        lv = build_pwhl(self._df_5m_history, today)
        for l in lv:
            self._levels.append(l)
        if lv:
            for l in lv:
                notify.append(f"[LIQ] 유동성 추가: {l['level_type']}({l['price_center']:,.1f})")

        return notify

    # ── EQL/EQH 재계산
    def _rebuild_eq_levels(self, pivots_15m: pd.DataFrame,
                           today: pd.Timestamp,
                           current_ts: Optional[pd.Timestamp] = None) -> List[str]:
        notify = []

        # 기존 EQL/EQH 제거 (flipped 아닌 것만)
        self._remove_by_type(
            ["eql", "eqh"],
            keep_flipped=True
        )

        if pivots_15m.empty:
            return notify

        # 5영업일 필터 + 미래 피봇 제외
        cutoff = _last_n_business_days(today, self.lookback_days)
        upper = current_ts if current_ts is not None else today + pd.Timedelta(hours=24)
        if "time" in pivots_15m.columns:
            t_col = pd.to_datetime(pivots_15m["time"], utc=True)
            piv   = pivots_15m[(t_col >= cutoff) & (t_col <= upper)].copy()
        else:
            piv = pivots_15m.copy()

        # 주말(토~일) 가격 범위 (월요일일 때만)
        _wk_high, _wk_low = None, None
        if today.weekday() == 0 and self._df_5m_history is not None:
            sat_start = today - pd.Timedelta(days=2)
            wk_mask = (self._df_5m_history["time"] >= sat_start) & \
                      (self._df_5m_history["time"] < today)
            wk_seg = self._df_5m_history[wk_mask]
            if not wk_seg.empty:
                _wk_high = float(wk_seg["high"].max())
                _wk_low  = float(wk_seg["low"].min())

        # EQL/EQH 생성 후 방향별 정렬 필터 적용
        # ATR 계산: 5m history에서 (DBSCAN ε 산출용)
        _atr_eq = _calc_atr(self._df_5m_history) if self._df_5m_history is not None else 0.0

        for kind in ("H", "L"):
            lvs = build_eq_levels(piv, kind, atr=_atr_eq,
                                  min_touches=self.min_touches_eq)
            if not lvs:
                continue
            lt = "eqh" if kind == "H" else "eql"

            # 주말 터치 필터: EQH가 주말 고점보다 낮으면 삭제, EQL이 주말 저점보다 높으면 삭제
            if _wk_high is not None and _wk_low is not None:
                _before = len(lvs)
                if lt == "eqh":
                    lvs = [lv for lv in lvs if lv["price_center"] > _wk_high]
                else:  # eql
                    lvs = [lv for lv in lvs if lv["price_center"] < _wk_low]
                _removed = _before - len(lvs)
                if _removed > 0:
                    notify.append(f"[LIQ] 주말 터치로 {lt.upper()} {_removed}개 필터링")
            if not lvs:
                continue

            # 시간순 정렬
            lvs_sorted = sorted(lvs, key=lambda x: x["ts_start"])

            if lt == "eql":
                filtered: List[dict] = []
                for lv in lvs_sorted:
                    pc = lv["price_center"]
                    filtered = [x for x in filtered if x["price_center"] <= pc]
                    filtered.append(lv)
                filtered = sorted(filtered, key=lambda x: x["price_center"])

            else:  # eqh
                filtered = []
                for lv in lvs_sorted:
                    pc = lv["price_center"]
                    filtered = [x for x in filtered if x["price_center"] >= pc]
                    filtered.append(lv)
                filtered = sorted(filtered, key=lambda x: x["price_center"], reverse=True)

            for l in filtered:
                self._levels.append(l)

        eq_count = sum(1 for l in self._levels if l["level_type"] in ("eql", "eqh"))
        notify.append(f"[LIQ] EQL/EQH {eq_count}개 갱신")
        return notify

    # ── 1H 봉 마감 시 EQL/EQH 갱신 (외부 호출용)
    def refresh_eq_levels(self, pivots_15m: pd.DataFrame,
                          today: pd.Timestamp,
                          current_ts: Optional[pd.Timestamp] = None) -> List[str]:
        """backtest/engine에서 1H 봉 마감 시 호출."""
        return self._rebuild_eq_levels(pivots_15m, today, current_ts=current_ts)

    # ── timeout된 레벨 비활성화 (외부 호출용)
    def disable_level(self, level_type: str, price_center: float,
                      tolerance_pct: float = 0.005) -> bool:
        """
        signals.py의 STANDBY timeout 시 호출.
        해당 레벨을 _levels에서 제거.
        반환: True=제거됨, False=없음
        """
        before = len(self._levels)
        self._levels = [
            lv for lv in self._levels
            if not (lv["level_type"] == level_type and
                    abs(lv["price_center"] - price_center) / max(price_center, 1)
                    < tolerance_pct)
        ]
        removed = before - len(self._levels)
        return removed > 0

    # ── 주말 유동성 정리 (터치된 레벨 삭제)
    def _weekend_cleanup(self, bar: dict) -> List[str]:
        """주말에 가격이 레벨을 터치/돌파하면 해당 레벨 삭제."""
        notify = []
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        cleanup_types = {"pdh", "pdl", "pwh", "pwl", "eqh", "eql"}

        remaining = []
        for lv in self._levels:
            lt = lv["level_type"]
            if lt not in cleanup_types:
                remaining.append(lv)
                continue
            zh = float(lv["zone_high"])
            zl = float(lv["zone_low"])
            touched = False
            if lt in ("pdh", "pwh", "eqh") and h > zh:
                touched = True
            elif lt in ("pdl", "pwl", "eql") and l < zl:
                touched = True
            if touched:
                notify.append(f"[LIQ] 주말 터치 삭제: {lt} @ {lv['price_center']:.1f}")
            else:
                remaining.append(lv)
        self._levels = remaining
        return notify

    # ── 만료 레벨 처리
    def _expire_levels(self, ts: pd.Timestamp) -> List[str]:
        notify = []
        remaining = []

        for lv in self._levels:
            ts_end = lv.get("ts_end")
            if ts_end is None:
                remaining.append(lv)
                continue

            ts_end_ts = pd.Timestamp(ts_end)
            if ts_end_ts.tzinfo is None:
                ts_end_ts = ts_end_ts.tz_localize("UTC")

            if ts >= ts_end_ts:
                lt = lv["level_type"]
                # EQL/EQH 만료 → 단순 삭제
                if lt in ("eql", "eqh"):
                    notify.append(f"[LIQ] 만료 삭제: {lt} @ {lv['price_center']:.1f}")
                    self.stat_expired += 1
                # Asia/London/Session H/L 만료 → 단순 삭제
                elif lt in ("asia_high", "asia_low", "london_high", "london_low",
                            "idm_high", "idm_low"):
                    notify.append(f"[LIQ] 만료 삭제: {lt} @ {lv['price_center']:.1f}")
                    self.stat_expired += 1
                else:
                    self.stat_expired += 1
            else:
                remaining.append(lv)

        self._levels = remaining

        # flipped 레벨도 만료 체크
        remaining_flip = []
        for lv in self._flipped:
            ts_end = lv.get("ts_end")
            if ts_end is None:
                remaining_flip.append(lv)
                continue
            ts_end_ts = pd.Timestamp(ts_end)
            if ts_end_ts.tzinfo is None:
                ts_end_ts = ts_end_ts.tz_localize("UTC")
            if ts < ts_end_ts:
                remaining_flip.append(lv)
            else:
                self.stat_expired += 1
        self._flipped = remaining_flip

        return notify

    # ── sweep 감지 (레벨 삭제 처리)
    def _detect_sweeps(self, bar: dict) -> List[str]:
        notify = []
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))

        remaining = []
        for lv in self._levels:
            lt = lv["level_type"]
            # PDH/PDL/PWH/PWL/Session H/L/IDM → 유지(sweep 후에도) + swept 마킹
            if lt in ("pdh", "pdl", "pwh", "pwl",
                      "asia_high", "asia_low",
                      "london_high", "london_low",
                      "idm_high", "idm_low"):
                # 미회수 장부용: 침범 시 swept=True 마킹 (제거는 안 함, 유효기간 유지)
                if not lv.get("swept", False):
                    is_high = lt.endswith("_high") or lt in ("pdh", "pwh")
                    px = float(lv["price_center"])
                    if is_high and h > px:
                        lv["swept"] = True
                    elif (not is_high) and l < px:
                        lv["swept"] = True
                remaining.append(lv)
                continue

            # EQL/EQH → sweep 시 S/R flip
            zh = float(lv["zone_high"])
            zl = float(lv["zone_low"])
            swept = False
            if h > zh and lt == "eqh":
                swept = True
            elif l < zl and lt == "eql":
                swept = True

            if swept:
                self.stat_sweeps += 1
                flip_map = {"eql": "eqh", "eqh": "eql"}
                if lt in flip_map:
                    confirmed = _sr_flip_confirmed(lv, self._df_5m_history)
                    if confirmed:
                        flipped = lv.copy()
                        flipped["flipped"]      = True
                        flipped["level_type"]   = flip_map[lt]
                        flipped["flip_bar_idx"] = self._bar_count
                        self._flipped.append(flipped)
                        self.stat_flips += 1
                        notify.append(f"[LIQ] swept→flip(확인): {lt} @ {lv['price_center']:.1f}")
                    else:
                        notify.append(f"[LIQ] swept→flip 조건 미달: {lt} @ {lv['price_center']:.1f} → 삭제")
            else:
                remaining.append(lv)

        # flipped 레벨 sweep 처리
        remaining_flip = []
        for lv in self._flipped:
            lt   = lv["level_type"]
            zh   = float(lv["zone_high"])
            zl   = float(lv["zone_low"])
            swept = False
            if h > zh and lt == "eqh":
                swept = True
            elif l < zl and lt == "eql":
                swept = True
            if swept:
                self.stat_sweeps += 1
                notify.append(f"[LIQ] swept (flipped): {lt} @ {lv['price_center']:.1f}")
            else:
                remaining_flip.append(lv)

        self._levels  = remaining
        self._flipped = remaining_flip
        return notify

    # ── 유틸: 타입별 레벨 제거
    def _remove_by_type(self, types: List[str], keep_flipped: bool = False):
        self._levels = [l for l in self._levels
                        if l["level_type"] not in types]
        if not keep_flipped:
            self._flipped = [l for l in self._flipped
                             if l["level_type"] not in types]

    def unswept_counts(self, price: float) -> dict:
        """미회수(swept=False) swing pivot 유동성 장부 (명세 3번).
        현재가 기준 위/아래 미회수 유동성 개수와 쏠림(imbalance) 반환.
        draw-on-liquidity feature 토대: 미회수 유동성이 많은 쪽으로 draw.
        """
        above = 0  # 현재가 위 미회수 high 유동성
        below = 0  # 현재가 아래 미회수 low 유동성
        for lv in self._levels:
            if lv.get("swept", False):
                continue
            lt = lv["level_type"]
            is_high = lt.endswith("_high") or lt in ("pdh", "pwh", "eqh")
            is_low  = lt.endswith("_low")  or lt in ("pdl", "pwl", "eql")
            px = float(lv["price_center"])
            if is_high and px > price:
                above += 1
            elif is_low and px < price:
                below += 1
        total = above + below
        imbalance = (above - below) / total if total > 0 else 0.0
        return {
            "unswept_above": float(above),
            "unswept_below": float(below),
            "unswept_imbalance": imbalance,  # +면 위쪽 유동성 많음(draw 위로)
        }

    # ── 활성 레벨 반환 (병합 포함)
    def get_levels(self, current_bar_idx: float = 0,
                   do_merge: bool = True) -> List[dict]:
        """현재 활성 레벨 반환."""
        # flip 쿨다운 필터: flip 후 12봉(1h)은 sweep 감지 제외
        active_flipped = [
            lv for lv in self._flipped
            if (self._bar_count - lv.get("flip_bar_idx", 0)) >= self._flip_cooldown_bars
        ]
        all_lvs = [lv.copy() for lv in self._levels + active_flipped]

        if do_merge:
            all_lvs = merge_levels(all_lvs, self.merge_pct)

        return sorted(all_lvs, key=lambda x: x["price_center"])

    # ── 진단 출력
    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for lv in self._levels + self._flipped:
            lt = lv["level_type"]
            counts[lt] = counts.get(lt, 0) + 1
        lines = [f"[LIQ] 활성 레벨 {len(self._levels) + len(self._flipped)}개: {counts}",
                 f"      sweeps={self.stat_sweeps} flips={self.stat_flips} expired={self.stat_expired}"]
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # IDM (Inducement) — 15m 구조 기반
    # ══════════════════════════════════════════════════════════════
    #
    # Long IDM 생성:
    #   bull 상태에서 BOS(HH) → HL → LH 구조가 연속 형성될 때의 HL
    #   즉, BOS로 HH 확정 후 pullback(HL)이 생기고, 그 후 LH가 형성되면
    #   해당 HL이 IDM (유동성 타겟)
    #
    # Short IDM 생성:
    #   bear 상태에서 BOS(LL) → LH → HL 구조가 연속 형성될 때의 LH
    #
    # 삭제 조건:
    #   1) sweep 없이 새 HH(long) 또는 LL(short) 구조 확정 → 레벨 무효화
    #   2) IDM sweep → standby 진입 → 포지션 실패 → LL(long)/HH(short) 구조 파괴
    #      (구조 파괴 = 반대 CHOCH 발생)
    # ══════════════════════════════════════════════════════════════

    def _update_idm_levels(self, df_15m_struct: pd.DataFrame,
                           pivots_15m: pd.DataFrame,
                           ts: pd.Timestamp) -> List[str]:
        """15m 구조에서 IDM 레벨 감지 및 무효화."""
        notify = []

        if pivots_15m is None or pivots_15m.empty or len(df_15m_struct) < 3:
            return notify

        # 최근 15m 구조 상태
        last_row = df_15m_struct.iloc[-1]
        state = str(last_row.get("structure_state", "unknown"))

        # ── 1. 기존 IDM 무효화 체크
        remaining = []
        for lv in self._levels:
            lt = lv["level_type"]
            if lt not in ("idm_high", "idm_low"):
                remaining.append(lv)
                continue

            invalidated = False

            if lt == "idm_low":
                # long IDM: sweep 없이 새 HH 확정 → 무효화
                # 또는 bear CHOCH 발생 → 구조 파괴
                _created_idx = lv.get("_created_struct_idx", 0)
                # 생성 이후 bos_bull (새 HH) 발생했는지 확인
                recent = df_15m_struct.iloc[max(0, _created_idx - len(df_15m_struct)):] if _created_idx > 0 else df_15m_struct.tail(20)
                if len(recent) > 0:
                    # sweep 여부: 이 IDM 레벨이 sweep 된 적 있는지
                    was_swept = lv.get("_swept", False)
                    if not was_swept:
                        # sweep 안 됐는데 새 HH (bos_bull) → 가격이 올라가서 IDM 의미 없어짐
                        if recent["bos_bull"].any():
                            invalidated = True
                            notify.append(f"[IDM] 무효화(새HH): idm_low @ {lv['price_center']:.1f}")
                    # CHOCH bear → 구조 파괴
                    if recent["choch_bear"].any():
                        invalidated = True
                        notify.append(f"[IDM] 무효화(CHOCH): idm_low @ {lv['price_center']:.1f}")

            elif lt == "idm_high":
                # short IDM: sweep 없이 새 LL 확정 → 무효화
                _created_idx = lv.get("_created_struct_idx", 0)
                recent = df_15m_struct.iloc[max(0, _created_idx - len(df_15m_struct)):] if _created_idx > 0 else df_15m_struct.tail(20)
                if len(recent) > 0:
                    was_swept = lv.get("_swept", False)
                    if not was_swept:
                        if recent["bos_bear"].any():
                            invalidated = True
                            notify.append(f"[IDM] 무효화(새LL): idm_high @ {lv['price_center']:.1f}")
                    if recent["choch_bull"].any():
                        invalidated = True
                        notify.append(f"[IDM] 무효화(CHOCH): idm_high @ {lv['price_center']:.1f}")

            if not invalidated:
                remaining.append(lv)

        self._levels = remaining

        # ── 2. 새 IDM 감지
        # 최근 피봇에서 BOS 후 pullback 패턴 탐색
        p = pivots_15m.sort_values("idx").reset_index(drop=True)
        if len(p) < 4:
            return notify

        # 최근 5개 피봇에서 패턴 탐색
        recent_pivots = p.tail(5)
        kinds = recent_pivots["kind"].tolist()
        prices = recent_pivots["price"].astype(float).tolist()
        idxs = recent_pivots["idx"].astype(int).tolist()
        times_col = recent_pivots["time"].tolist() if "time" in recent_pivots.columns else [None] * len(recent_pivots)

        # 이미 등록된 IDM 가격 (중복 방지)
        existing_idm_prices = set()
        for lv in self._levels:
            if lv["level_type"] in ("idm_high", "idm_low"):
                existing_idm_prices.add(round(lv["price_center"], 2))

        # Long IDM: H-L-H-L-H 또는 부분 패턴에서 BOS(HH) 후 HL 찾기
        # 핵심: bull state에서 마지막 3개 피봇이 H(HH확정) → L(HL) → H(LH) 이면 그 L이 IDM
        if state == "bull" and len(kinds) >= 3:
            for i in range(len(kinds) - 2):
                if kinds[i] == "H" and kinds[i+1] == "L" and kinds[i+2] == "H":
                    h1, l1, h2 = prices[i], prices[i+1], prices[i+2]
                    # HH 확정 (h1이 이전 H보다 높은지는 struct_state=bull로 보장)
                    # h2 < h1 → LH (pullback 후 lower high)
                    if h2 < h1:
                        idm_price = l1
                        if round(idm_price, 2) not in existing_idm_prices:
                            _ts = pd.Timestamp(times_col[i+1]) if times_col[i+1] is not None else ts
                            if _ts.tzinfo is None:
                                _ts = _ts.tz_localize("UTC")
                            band = self.eq_band_pct
                            zl, zh = idm_price * (1 - band), idm_price * (1 + band)
                            lv = {
                                "level_type": "idm_low",
                                "price_center": idm_price,
                                "zone_low": zl,
                                "zone_high": zh,
                                "zone_pct": band,
                                "strength": STRENGTH.get("idm_low", 2.0),
                                "touches": 1,
                                "ts_start": _ts,
                                "ts_end": _ts + pd.Timedelta(days=3),
                                "_created_struct_idx": idxs[i+2],
                                "_swept": False,
                            }
                            self._levels.append(lv)
                            notify.append(f"[IDM] 생성: idm_low @ {idm_price:.1f}")

        # Short IDM: bear state에서 L → H(LH) → L(HL) 이면 그 H가 IDM
        if state == "bear" and len(kinds) >= 3:
            for i in range(len(kinds) - 2):
                if kinds[i] == "L" and kinds[i+1] == "H" and kinds[i+2] == "L":
                    l1, h1, l2 = prices[i], prices[i+1], prices[i+2]
                    # l2 > l1 → HL (pullback 후 higher low)
                    if l2 > l1:
                        idm_price = h1
                        if round(idm_price, 2) not in existing_idm_prices:
                            _ts = pd.Timestamp(times_col[i+1]) if times_col[i+1] is not None else ts
                            if _ts.tzinfo is None:
                                _ts = _ts.tz_localize("UTC")
                            band = self.eq_band_pct
                            zl, zh = idm_price * (1 - band), idm_price * (1 + band)
                            lv = {
                                "level_type": "idm_high",
                                "price_center": idm_price,
                                "zone_low": zl,
                                "zone_high": zh,
                                "zone_pct": band,
                                "strength": STRENGTH.get("idm_high", 2.0),
                                "touches": 1,
                                "ts_start": _ts,
                                "ts_end": _ts + pd.Timedelta(days=3),
                                "_created_struct_idx": idxs[i+2],
                                "_swept": False,
                            }
                            self._levels.append(lv)
                            notify.append(f"[IDM] 생성: idm_high @ {idm_price:.1f}")

        return notify