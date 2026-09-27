# fvg.py
from __future__ import annotations
import pandas as pd


def detect_fvgs_lookback(
    df: pd.DataFrame,
    lookback_bars: int = 200,
    fill_mode: str = "wick",      # "wick" or "close"
    remove_filled: bool = True,
    min_size: float = 0.0,
) -> pd.DataFrame:
    """
    최근 lookback_bars 만큼 되돌아보며 생성된 FVG(3-candle imbalance)를 탐지한다.

    필요 컬럼: open, high, low, close
    반환 컬럼:
      - fvg_type: "bull" / "bear"
      - created_idx: df.index 값
      - zone_low, zone_high
      - size
      - filled (remove_filled=False일 때 의미 있음)

    FVG 정의:
      - bull FVG at i: low[i] > high[i-2]
        zone = [high[i-2], low[i]]
      - bear FVG at i: high[i] < low[i-2]
        zone = [high[i], low[i-2]]

    filled 판정(생성 이후 구간에서):
      - fill_mode="wick":
          bull: low <= zone_low   -> filled
          bear: high >= zone_high -> filled
      - fill_mode="close":
          bull: close <= zone_low -> filled
          bear: close >= zone_high-> filled
    """
    if fill_mode not in ("wick", "close"):
        raise ValueError("fill_mode must be 'wick' or 'close'")

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    if lookback_bars < 3:
        return pd.DataFrame(columns=["fvg_type","created_idx","created_time","zone_low","zone_high","size","filled"])

    x = df.tail(lookback_bars).copy()
    idxs = x.index.to_list()

    highs = x["high"].astype(float).to_list()
    lows = x["low"].astype(float).to_list()
    closes = x["close"].astype(float).to_list()
    times = x["time"].to_list() if "time" in x.columns else [None] * len(x)

    fvgs = []

    # 1) 생성: 최근 lookback_bars 구간에서 생긴 모든 FVG
    for i in range(2, len(x)):
        # bull
        if lows[i] > highs[i - 2]:
            z_low = highs[i - 2]
            z_high = lows[i]
            size = z_high - z_low
            if size >= min_size:
                fvgs.append({
                    "fvg_type": "bull",
                    "created_idx": idxs[i],
                    "created_time": times[i],
                    "zone_low": float(z_low),
                    "zone_high": float(z_high),
                    "size": float(size),
                    "filled": False,
                })

        # bear
        if highs[i] < lows[i - 2]:
            z_low = highs[i]
            z_high = lows[i - 2]
            size = z_high - z_low
            if size >= min_size:
                fvgs.append({
                    "fvg_type": "bear",
                    "created_idx": idxs[i],
                    "created_time": times[i],
                    "zone_low": float(z_low),
                    "zone_high": float(z_high),
                    "size": float(size),
                    "filled": False,
                })

    if not fvgs:
        return pd.DataFrame(columns=["fvg_type","created_idx","created_time","zone_low","zone_high","size","filled"])

    fvg_df = pd.DataFrame(fvgs)

    # 2) filled 제거 옵션: 생성 이후 구간을 보며 채워졌는지 판정
    if remove_filled:
        pos = {idx: i for i, idx in enumerate(idxs)}

        for r_i in range(len(fvg_df)):
            created_idx = fvg_df.loc[r_i, "created_idx"]
            start = pos.get(created_idx, None)
            if start is None:
                continue

            z_low = float(fvg_df.loc[r_i, "zone_low"])
            z_high = float(fvg_df.loc[r_i, "zone_high"])
            typ = fvg_df.loc[r_i, "fvg_type"]

            filled = False
            for j in range(start + 1, len(x)):
                if typ == "bull":
                    if fill_mode == "wick":
                        if lows[j] <= z_low:
                            filled = True
                            break
                    else:
                        if closes[j] <= z_low:
                            filled = True
                            break
                else:  # bear
                    if fill_mode == "wick":
                        if highs[j] >= z_high:
                            filled = True
                            break
                    else:
                        if closes[j] >= z_high:
                            filled = True
                            break

            fvg_df.loc[r_i, "filled"] = filled

        fvg_df = fvg_df[fvg_df["filled"] == False].copy()

    # 최근 생성된 것이 위로 오게 정렬
    fvg_df = fvg_df.sort_values("created_idx", ascending=False).reset_index(drop=True)
    return fvg_df