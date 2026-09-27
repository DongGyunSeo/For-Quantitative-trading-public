# pivots.py
from __future__ import annotations

import pandas as pd


def build_pivots(df_with_swings: pd.DataFrame) -> pd.DataFrame:
    """
    find_swings() 결과(swing_high / swing_low 컬럼 포함)를
    liquidity.py / struct_event.py 가 기대하는 형태의 pivot 테이블로 변환.

    반환 columns:
      - idx   : df_with_swings 의 positional index (0-based int) — 연속 계산에 사용
      - time  : 원본 타임스탬프 (time 컬럼이 없으면 df.index 사용)
      - kind  : "H" (swing_high) / "L" (swing_low)
      - price : swing_high → high 값 / swing_low → low 값

    한 봉이 동시에 swing_high + swing_low 이면 두 행을 만든다
    (극히 드문 케이스지만 일관성을 위해 명시).
    """
    required = {"swing_high", "swing_low"}
    missing = required - set(df_with_swings.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)} — run find_swings() first.")

    if "high" not in df_with_swings.columns or "low" not in df_with_swings.columns:
        raise ValueError("df must contain 'high' and 'low' columns.")

    has_time = "time" in df_with_swings.columns

    rows = []
    for pos_i, (idx_label, row) in enumerate(df_with_swings.iterrows()):
        t = row["time"] if has_time else idx_label
        if bool(row["swing_high"]):
            rows.append({
                "idx":   pos_i,
                "time":  t,
                "kind":  "H",
                "price": float(row["high"]),
            })
        if bool(row["swing_low"]):
            rows.append({
                "idx":   pos_i,
                "time":  t,
                "kind":  "L",
                "price": float(row["low"]),
            })

    if not rows:
        return pd.DataFrame(columns=["idx", "time", "kind", "price"])

    return pd.DataFrame(rows).reset_index(drop=True)


def compress_pivots(
    pivots_raw: pd.DataFrame,
    mode: str = "zigzag",
) -> pd.DataFrame:
    """
    연속으로 같은 kind(H/H/H 또는 L/L/L)가 나열될 때 중복을 제거해
    H-L-H-L … 지그재그 형태로 만든다.

    mode:
      - "zigzag" (기본, 권장)
          H 연속 → 가장 높은 값(최대) 하나만 남김
          L 연속 → 가장 낮은 값(최소) 하나만 남김

      - "last"
          연속 구간에서 마지막 것만 남김
          (빠르지만 극단값 보존 안 됨)

    반환: build_pivots() 와 동일한 columns (idx, time, kind, price)
    단, idx는 원본 positional 값 그대로 유지(재정렬 안 함).
    """
    if mode not in ("zigzag", "last"):
        raise ValueError("mode must be 'zigzag' or 'last'")

    required = {"idx", "kind", "price"}
    missing = required - set(pivots_raw.columns)
    if missing:
        raise ValueError(f"pivots_raw missing columns: {sorted(missing)}")

    if pivots_raw.empty:
        return pivots_raw.copy()

    # 안정 정렬을 보장하여 동일 idx 내에서 H/L 순서 셔플에 의한 구조(Structure) 플리커링 방지
    p = pivots_raw.sort_values(["idx", "kind"]).reset_index(drop=True)

    # 연속 same-kind 그룹 번호 부여
    p["_grp"] = (p["kind"] != p["kind"].shift()).cumsum()

    kept_rows = []

    for _, grp in p.groupby("_grp", sort=True):
        kind = grp["kind"].iloc[0]

        if mode == "zigzag":
            if kind == "H":
                best_i = grp["price"].idxmax()
            else:
                best_i = grp["price"].idxmin()
            kept_rows.append(grp.loc[best_i])
        else:  # "last"
            kept_rows.append(grp.iloc[-1])

    out = pd.DataFrame(kept_rows).drop(columns=["_grp"]).reset_index(drop=True)
    return out