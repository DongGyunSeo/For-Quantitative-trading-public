#struct_event.py
import pandas as pd


def detect_structure_events_close_protected(df: pd.DataFrame, pivots: pd.DataFrame) -> pd.DataFrame:
    """
    Close 기준으로만 BOS/CHOCH를 판정하면서,
    구조 방어선(protected_low / protected_high)을 함께 추적한다.

    입력
      - df: OHLCV 원본(시간순, index가 시간 또는 정렬된 인덱스)
      - pivots: pivot 리스트 (columns: idx, kind("H"/"L"), price)

    출력(df에 컬럼 추가)
      - bos_bull, bos_bear, choch_bull, choch_bear (bool)
      - structure_state: "bull" / "bear" / "unknown"
      - last_swing_high, last_swing_low
      - protected_low, protected_high

    protected 정의(v0.1, 단순/명확)
      - bull 상태에서 protected_low = bull 상태에서 확정된 "마지막 pivot low"
      - bear 상태에서 protected_high = bear 상태에서 확정된 "마지막 pivot high"
      - bull에서 close < protected_low  -> CHOCH bear
      - bear에서 close > protected_high -> CHOCH bull

    참고:
      - 이 버전은 "보호 레벨"을 구조 상태에 따라 1개씩 유지하는 형태라 안정적임.
      - 더 ICT스럽게(마지막 HL/LH 엄격 채택) 하려면 이후 조건을 추가하면 됨.
    """
    out = df.copy()

    # 이벤트 컬럼 초기화
    for c in ["bos_bull", "bos_bear", "choch_bull", "choch_bear"]:
        out[c] = False
    out["structure_state"] = "unknown"

    out["last_swing_high"] = pd.NA
    out["last_swing_low"] = pd.NA
    out["protected_low"] = pd.NA
    out["protected_high"] = pd.NA

    # pivot (idx -> (kind, price)) 매핑
    pivot_by_idx = {}
    for _, r in pivots.iterrows():
        pivot_by_idx[r["idx"]] = (r["kind"], float(r["price"]))

    last_H = None
    last_L = None

    protected_low = None    # bull에서 의미
    protected_high = None   # bear에서 의미

    state = "unknown"

    for idx, row in out.iterrows():
        # pivot 업데이트
        if idx in pivot_by_idx:
            kind, price = pivot_by_idx[idx]
            if kind == "H":
                last_H = price
                # bear 상태에서는 "방어선 high"를 최신 pivot high로 갱신
                if state == "bear":
                    protected_high = price
            else:
                last_L = price
                # bull 상태에서는 "방어선 low"를 최신 pivot low로 갱신
                if state == "bull":
                    protected_low = price

        close = float(row["close"])

        # --- break 판정 (close-only) ---
        up_break_lastH = (last_H is not None) and (close > last_H)
        dn_break_lastL = (last_L is not None) and (close < last_L)

        # protected break (CHOCH 트리거)
        dn_break_protL = (protected_low is not None) and (close < protected_low)
        up_break_protH = (protected_high is not None) and (close > protected_high)

        # --- 상태머신 ---
        if state == "unknown":
            # 최초 방향 결정은 last swing break로 BOS 처리
            if up_break_lastH:
                out.at[idx, "bos_bull"] = True
                state = "bull"
                # bull로 들어가는 순간: 보호선 low는 "현재까지의 마지막 L"로 초기화
                if protected_low is None and last_L is not None:
                    protected_low = last_L
                # 반대쪽 보호선은 의미 없으니 유지/무시
            elif dn_break_lastL:
                out.at[idx, "bos_bear"] = True
                state = "bear"
                if protected_high is None and last_H is not None:
                    protected_high = last_H

        elif state == "bull":
            # bull -> bear 전환: protected_low가 깨지면 CHOCH
            if dn_break_protL:
                out.at[idx, "choch_bear"] = True
                state = "bear"
                # bear로 들어가며 보호선 high 초기화
                if protected_high is None and last_H is not None:
                    protected_high = last_H
            else:
                # bull 지속: last_H 돌파는 BOS(컨티뉴에이션)
                if up_break_lastH:
                    out.at[idx, "bos_bull"] = True

        else:  # state == "bear"
            # bear -> bull 전환: protected_high가 깨지면 CHOCH
            if up_break_protH:
                out.at[idx, "choch_bull"] = True
                state = "bull"
                # bull로 들어가며 보호선 low 초기화
                if protected_low is None and last_L is not None:
                    protected_low = last_L
            else:
                # bear 지속: last_L 이탈은 BOS
                if dn_break_lastL:
                    out.at[idx, "bos_bear"] = True

        # 기록
        out.at[idx, "structure_state"] = state
        out.at[idx, "last_swing_high"] = last_H
        out.at[idx, "last_swing_low"] = last_L
        out.at[idx, "protected_low"] = protected_low
        out.at[idx, "protected_high"] = protected_high

    return out