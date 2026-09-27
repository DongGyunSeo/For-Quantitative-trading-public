# order_blocks.py
from __future__ import annotations
import pandas as pd


def build_order_blocks_from_bos(
    df: pd.DataFrame,
    lookback_candles: int = 30,
    zone_mode: str = "conservative",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    TF 제한 없이 BOS 기반 OB 리스트 생성.

    필요 컬럼:
      - open, high, low, close
      - bos_bull, bos_bear
      - time_col(기본 "time")는 있으면 사용, 없으면 df.index 사용

    zone_mode:
      - "conservative"(추천)
          bull_ob: [low, open]   (마지막 bearish candle)
          bear_ob: [open, high]  (마지막 bullish candle)
      - "full"
          bull_ob: [low, high]
          bear_ob: [low, high]

    반환 columns:
      - ob_type: "bull_ob" / "bear_ob"
      - formed_at: BOS 발생 시점(time or index)
      - ob_time: OB 캔들 시점(time or index)
      - ob_candle_idx: df.index 값
      - zone_low, zone_high
      - invalidated: False (몸통으로 완전히 지워지면 True)
    """
    if zone_mode not in ("conservative", "full"):
        raise ValueError("zone_mode must be 'conservative' or 'full'")

    required = {"open", "high", "low", "close", "bos_bull", "bos_bear"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    times = df[time_col].to_list() if time_col in df.columns else df.index.to_list()
    idx_list = df.index.to_list()

    obs = []
    n = len(df)

    for i in range(n):
        # bullish OB: bullish BOS 직전 마지막 bearish candle
        if bool(df["bos_bull"].iloc[i]):
            start = max(0, i - lookback_candles)
            ob_i = None
            for k in range(i - 1, start - 1, -1):
                if float(df["close"].iloc[k]) < float(df["open"].iloc[k]):
                    ob_i = k
                    break
            if ob_i is not None:
                o = float(df["open"].iloc[ob_i])
                h = float(df["high"].iloc[ob_i])
                l = float(df["low"].iloc[ob_i])

                if zone_mode == "conservative":
                    zone_low, zone_high = l, o
                else:
                    zone_low, zone_high = l, h

                zone_low, zone_high = min(zone_low, zone_high), max(zone_low, zone_high)
                obs.append({
                    "ob_type": "bull_ob",
                    "formed_at": times[i],
                    "ob_time": times[ob_i],
                    "ob_candle_idx": idx_list[ob_i],
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "invalidated": False,
                    "invalidated_at": None,
                })

        # bearish OB: bearish BOS 직전 마지막 bullish candle
        if bool(df["bos_bear"].iloc[i]):
            start = max(0, i - lookback_candles)
            ob_i = None
            for k in range(i - 1, start - 1, -1):
                if float(df["close"].iloc[k]) > float(df["open"].iloc[k]):
                    ob_i = k
                    break
            if ob_i is not None:
                o = float(df["open"].iloc[ob_i])
                h = float(df["high"].iloc[ob_i])
                l = float(df["low"].iloc[ob_i])

                if zone_mode == "conservative":
                    zone_low, zone_high = o, h
                else:
                    zone_low, zone_high = l, h

                zone_low, zone_high = min(zone_low, zone_high), max(zone_low, zone_high)
                obs.append({
                    "ob_type": "bear_ob",
                    "formed_at": times[i],
                    "ob_time": times[ob_i],
                    "ob_candle_idx": idx_list[ob_i],
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "invalidated": False,
                    "invalidated_at": None,
                })

    return pd.DataFrame(obs)


def mark_invalidation_by_close(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    '터치'로는 폐기하지 않고,
    '몸통(close)으로 완전히 지워버리면' invalidated=True 처리.

    invalidation rule (v0.1, close-only):
      - bull_ob: close < zone_low  이면 완전 무효화(하단을 종가로 깔고 내려감)
      - bear_ob: close > zone_high 이면 완전 무효화(상단을 종가로 깔고 올라감)
    """
    if obs is None or obs.empty:
        return obs.copy()

    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")

    out = obs.copy()

    times = df[time_col].to_list() if time_col in df.columns else df.index.to_list()
    pos = {t: i for i, t in enumerate(times)}
    closes = df["close"].astype(float).to_list()

    for r_i in range(len(out)):
        ob_time = out["ob_time"].iloc[r_i]
        if ob_time not in pos:
            continue

        start = pos[ob_time] + 1
        zone_low = float(out["zone_low"].iloc[r_i])
        zone_high = float(out["zone_high"].iloc[r_i])
        ob_type = out["ob_type"].iloc[r_i]

        invalid = False
        invalidated_at = None
        for i in range(start, len(times)):
            c = closes[i]
            if ob_type == "bull_ob":
                if c < zone_low:
                    invalid = True
                    invalidated_at = times[i]
                    break
            else:  # bear_ob
                if c > zone_high:
                    invalid = True
                    invalidated_at = times[i]
                    break

        out.at[out.index[r_i], "invalidated"] = invalid
        out.at[out.index[r_i], "invalidated_at"] = invalidated_at

    return out


def select_latest_two_obs(
    obs: pd.DataFrame,
    t_now,
    include_invalidated: bool = False,
) -> dict:
    """
    현재 시점(t_now) 기준 '가장 최근 bull_ob 1개' + '가장 최근 bear_ob 1개' 반환.
    (상승/하락 OB 공존 → 2개 관리)

    include_invalidated=False면 invalidated=True는 제외.
    """
    if obs is None or obs.empty:
        return {"bull_ob": None, "bear_ob": None}

    o = obs[obs["formed_at"] <= t_now].copy()
    if o.empty:
        return {"bull_ob": None, "bear_ob": None}

    if not include_invalidated:
        if "invalidated_at" in o.columns:
            # 타임스탬프가 지정되지 않았거나(깨지지 않음), 현재 시점 이후에 깨진 경우만 유효
            o = o[o["invalidated_at"].isna() | (o["invalidated_at"] > t_now)]
        elif "invalidated" in o.columns:
            o = o[o["invalidated"] == False]

    result = {"bull_ob": None, "bear_ob": None}
    for typ in ("bull_ob", "bear_ob"):
        sub = o[o["ob_type"] == typ]
        if sub.empty:
            continue
        row = sub.sort_values("formed_at").iloc[-1]
        result[typ] = row.to_dict()

    return result


def ob_stop_take_from_ob(
    direction: str,
    entry: float,
    ob: dict,
    buffer: float = 0.0,
    rr: float = 1.5,
) -> dict:
    """
    OB zone을 기반으로 SL/TP를 잡는 단순 유틸 (5m에서 쓰기 좋음)
    - direction: "long" / "short"
    - entry: 진입가
    - ob: select_latest_two_obs()로 얻은 bull_ob 또는 bear_ob
    - buffer: SL 여유
    - rr: 목표 RR

    반환: {"sl":..., "tp":..., "risk":..., "reward":...}
    """
    if ob is None:
        return {"sl": None, "tp": None, "risk": None, "reward": None}

    zlow = float(ob["zone_low"])
    zhigh = float(ob["zone_high"])

    if direction == "long":
        sl = zlow - buffer
        risk = entry - sl
        if risk <= 0:
            return {"sl": sl, "tp": None, "risk": risk, "reward": None}
        tp = entry + rr * risk
        return {"sl": sl, "tp": tp, "risk": risk, "reward": rr * risk}

    if direction == "short":
        sl = zhigh + buffer
        risk = sl - entry
        if risk <= 0:
            return {"sl": sl, "tp": None, "risk": risk, "reward": None}
        tp = entry - rr * risk
        return {"sl": sl, "tp": tp, "risk": risk, "reward": rr * risk}

    raise ValueError("direction must be 'long' or 'short'")