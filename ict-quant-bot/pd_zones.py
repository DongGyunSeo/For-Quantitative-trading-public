# pd_zones.py  v2.0 — ICT Dealing Range (BOS origin + IDM sweep confirmation)
# ═══════════════════════════════════════════════════════════════════════════
#
# 핵심 원칙:
#   1. PD 레인지는 impulse leg 기반으로 고정 (매 봉 갱신 금지)
#   2. BOS → 기원점(dealing_low/high) 확정
#   3. IDM sweep → 풀백 확정 → temp_extreme을 dealing_high/low로 lock
#   4. ChoCh → 방향 전환, 새 사이클 시작
#   5. 클램핑 금지 — pd_pos < 0 (SSL sweep), > 1 (BSL sweep)은 유효한 신호
#
# 갱신 트리거:
#   ① BOS 발생 → impulse 기원점 = dealing_low(bull) / dealing_high(bear)
#                 temp_extreme 추적 시작, range unlocked
#   ② IDM sweep  → temp_extreme을 dealing_high(bull) / dealing_low(bear)로 확정
#                   range locked (다음 BOS/ChoCh까지 유지)
#   ③ ChoCh 발생 → 방향 전환, 새 사이클
#
# 출력:
#   dealing_low, dealing_high: PD 레인지 경계 (고정)
#   pd_pos: (price - dealing_low) / (dealing_high - dealing_low)
#           클램핑 없음 → sweep 시 음수/1 초과 가능
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
from typing import Optional, Tuple


class DealingRangeTracker:
    """
    1H 구조 이벤트(BOS/ChoCh) + 피봇 기반으로 Dealing Range를 추적.

    사용법:
        tracker = DealingRangeTracker()
        # 1H 봉 마감마다 호출
        tracker.update(df_1h_struct, pivots_1h)
        # 진입 시 pd_pos 계산
        pd_pos = tracker.get_pd_pos(current_price)
    """

    def __init__(self):
        self.dealing_low: Optional[float] = None
        self.dealing_high: Optional[float] = None
        self.trend_dir: int = 0          # 1=bull, -1=bear, 0=unknown
        self.is_range_locked: bool = False

        # 내부 추적 변수
        self._temp_extreme: Optional[float] = None    # BOS 후 고점/저점 추적
        self._last_bos_idx: int = -1                   # 마지막 BOS 처리한 struct idx
        self._last_choch_idx: int = -1                 # 마지막 ChoCh 처리한 struct idx
        self._idm_level: Optional[float] = None        # IDM(미끼) 레벨
        self._idm_set: bool = False                    # IDM이 설정되었는지
        self._prev_struct_len: int = 0                 # 이전 업데이트까지의 struct 길이

    def update(self, df_1h_struct: pd.DataFrame, pivots_1h: pd.DataFrame):
        """
        1H 봉 마감마다 호출. 새로 추가된 구조 이벤트만 처리.

        Parameters
        ----------
        df_1h_struct : pd.DataFrame
            1H 구조 DataFrame (bos_bull, bos_bear, choch_bull, choch_bear,
                               structure_state, last_swing_high, last_swing_low 포함)
        pivots_1h : pd.DataFrame
            1H 피봇 (idx, kind, price)
        """
        if df_1h_struct is None or df_1h_struct.empty:
            return

        n = len(df_1h_struct)
        # 새로 추가된 행만 처리
        start = max(0, self._prev_struct_len)
        self._prev_struct_len = n

        for i in range(start, n):
            row = df_1h_struct.iloc[i]
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # ── temp_extreme 추적 (BOS 이후 range unlock 상태에서)
            if not self.is_range_locked and self._temp_extreme is not None:
                if self.trend_dir == 1:
                    self._temp_extreme = max(self._temp_extreme, high)
                elif self.trend_dir == -1:
                    self._temp_extreme = min(self._temp_extreme, low)

            # ── IDM sweep 감지 → 풀백 확정 → range lock
            if (not self.is_range_locked and self._idm_set
                    and self._idm_level is not None
                    and self._temp_extreme is not None):
                if self.trend_dir == 1 and close < self._idm_level:
                    # Bull: IDM(minor swing low) 하방 sweep → 풀백 확정
                    self.dealing_high = self._temp_extreme
                    self.is_range_locked = True
                    self._idm_set = False
                elif self.trend_dir == -1 and close > self._idm_level:
                    # Bear: IDM(minor swing high) 상방 sweep → 풀백 확정
                    self.dealing_low = self._temp_extreme
                    self.is_range_locked = True
                    self._idm_set = False

            # ── IDM 설정: BOS 후 첫 번째 반대 피봇이 IDM
            if (not self.is_range_locked and not self._idm_set
                    and self._temp_extreme is not None):
                # 현재 행에 피봇이 있는지 확인
                if pivots_1h is not None and not pivots_1h.empty:
                    pivot_at_i = pivots_1h[pivots_1h["idx"] == i]
                    if not pivot_at_i.empty:
                        pk = str(pivot_at_i.iloc[0]["kind"])
                        pp = float(pivot_at_i.iloc[0]["price"])
                        if self.trend_dir == 1 and pk == "L":
                            # Bull에서 minor low = IDM
                            self._idm_level = pp
                            self._idm_set = True
                        elif self.trend_dir == -1 and pk == "H":
                            # Bear에서 minor high = IDM
                            self._idm_level = pp
                            self._idm_set = True

            # ── ChoCh 감지 → 방향 전환
            if row.get("choch_bull", False) and self.trend_dir != 1:
                self._start_new_cycle(1, df_1h_struct, pivots_1h, i, close)
                continue
            if row.get("choch_bear", False) and self.trend_dir != -1:
                self._start_new_cycle(-1, df_1h_struct, pivots_1h, i, close)
                continue

            # ── BOS 감지 → 기원점 확정 + temp_extreme 추적 시작
            if row.get("bos_bull", False) and self.trend_dir == 1:
                self._handle_bos(1, df_1h_struct, pivots_1h, i, close)
            elif row.get("bos_bear", False) and self.trend_dir == -1:
                self._handle_bos(-1, df_1h_struct, pivots_1h, i, close)

            # ── 초기 방향 설정 (unknown 상태)
            if self.trend_dir == 0:
                if row.get("bos_bull", False):
                    self._start_new_cycle(1, df_1h_struct, pivots_1h, i, close)
                elif row.get("bos_bear", False):
                    self._start_new_cycle(-1, df_1h_struct, pivots_1h, i, close)

    def _start_new_cycle(self, direction: int,
                         df_struct: pd.DataFrame, pivots: pd.DataFrame,
                         current_idx: int, close: float):
        """ChoCh 또는 초기 BOS 시 새 사이클 시작."""
        self.trend_dir = direction
        self.is_range_locked = False
        self._idm_set = False
        self._idm_level = None

        # 기원점 찾기: BOS/ChoCh 직전의 가장 깊은 눌림목
        origin = self._find_impulse_origin(direction, df_struct, pivots, current_idx)

        if direction == 1:
            self.dealing_low = origin
            self.dealing_high = None  # 아직 미확정
            self._temp_extreme = close  # 고점 추적 시작
        else:
            self.dealing_high = origin
            self.dealing_low = None  # 아직 미확정
            self._temp_extreme = close  # 저점 추적 시작

    def _handle_bos(self, direction: int,
                    df_struct: pd.DataFrame, pivots: pd.DataFrame,
                    current_idx: int, close: float):
        """
        BOS(continuation) 발생 시:
        - 기존 range가 locked이면 → 새 impulse 시작, range unlock
        - unlocked이면 → 기원점만 업데이트
        """
        origin = self._find_impulse_origin(direction, df_struct, pivots, current_idx)

        if direction == 1:
            self.dealing_low = origin
            self._temp_extreme = close
        else:
            self.dealing_high = origin
            self._temp_extreme = close

        self.is_range_locked = False
        self._idm_set = False
        self._idm_level = None

    def _find_impulse_origin(self, direction: int,
                             df_struct: pd.DataFrame, pivots: pd.DataFrame,
                             current_idx: int) -> Optional[float]:
        """
        BOS/ChoCh를 만든 impulse leg의 기원점(가장 깊은 눌림목)을 찾는다.

        Bull BOS: 직전 구간에서 가장 낮은 pivot low
        Bear BOS: 직전 구간에서 가장 높은 pivot high
        """
        if pivots is None or pivots.empty:
            return None

        # 최근 10개 피봇 중 current_idx 이전의 것들
        p = pivots[pivots["idx"] <= current_idx].sort_values("idx")
        if p.empty:
            return None

        # 최근 10개로 제한
        p = p.tail(10)

        if direction == 1:
            # Bull: 가장 낮은 pivot low = impulse 기원점
            lows = p[p["kind"] == "L"]
            if lows.empty:
                return float(df_struct.iloc[max(0, current_idx - 5):current_idx + 1]["low"].min())
            return float(lows["price"].min())
        else:
            # Bear: 가장 높은 pivot high = impulse 기원점
            highs = p[p["kind"] == "H"]
            if highs.empty:
                return float(df_struct.iloc[max(0, current_idx - 5):current_idx + 1]["high"].max())
            return float(highs["price"].max())

    def get_pd_pos(self, price: float) -> Optional[float]:
        """
        현재 가격의 PD position 계산.

        반환:
          - 0.0~1.0: 레인지 내부
          - < 0: dealing_low 하방 이탈 (SSL sweep 신호)
          - > 1: dealing_high 상방 이탈 (BSL sweep 신호)
          - None: 레인지 미확정

        ⚠️ 클램핑 없음 — 이탈 강도가 그대로 보존됨
        """
        if self.dealing_low is None or self.dealing_high is None:
            return None

        rng = self.dealing_high - self.dealing_low
        if rng <= 0:
            return None

        return round((price - self.dealing_low) / rng, 4)

    def get_dealing_range(self) -> Tuple[Optional[float], Optional[float]]:
        """현재 dealing_low, dealing_high 반환."""
        return self.dealing_low, self.dealing_high


# ══════════════════════════════════════════════════════════════════
# 하위 호환용 함수 (기존 backtest.py 인터페이스)
# ══════════════════════════════════════════════════════════════════

def pd_pos_2dp(px: float, pd_low: float, pd_high: float):
    """
    PD position 계산 (v2: 클램핑 제거).

    반환:
      - float: (px - pd_low) / (pd_high - pd_low)
      - 클램핑 없음 → 범위 밖 값 허용
      - pd_low >= pd_high이면 None
    """
    if pd.isna(pd_low) or pd.isna(pd_high) or pd.isna(px):
        return None
    rng = pd_high - pd_low
    if rng <= 0:
        return None
    return round((px - pd_low) / rng, 4)


def add_pd_bounds_from_strong_weak(
    df: pd.DataFrame,
    fallback_to_last_swing: bool = True,
) -> pd.DataFrame:
    """
    [Legacy] strong/weak 레벨로 PD 레인지 계산.
    DealingRangeTracker로 전환 전 호환용으로 유지.
    """
    out = df.copy()

    if "structure_state" not in out.columns:
        raise ValueError("df must contain 'structure_state'")

    if "strong_low" not in out.columns:
        out["strong_low"] = out.get("protected_low", pd.NA)
    if "strong_high" not in out.columns:
        out["strong_high"] = out.get("protected_high", pd.NA)
    if "weak_high" not in out.columns:
        out["weak_high"] = out.get("last_swing_high", pd.NA)
    if "weak_low" not in out.columns:
        out["weak_low"] = out.get("last_swing_low", pd.NA)

    if fallback_to_last_swing:
        if "last_swing_high" not in out.columns or "last_swing_low" not in out.columns:
            raise ValueError("fallback_to_last_swing=True requires last_swing_high/low columns")

    states = out["structure_state"].astype(str)
    pd_low = []
    pd_high = []

    for i in range(len(out)):
        st = states.iat[i]
        if st == "bull":
            lo = out["strong_low"].iat[i]
            hi = out["weak_high"].iat[i]
        elif st == "bear":
            hi = out["strong_high"].iat[i]
            lo = out["weak_low"].iat[i]
        else:
            lo = out["last_swing_low"].iat[i] if fallback_to_last_swing else pd.NA
            hi = out["last_swing_high"].iat[i] if fallback_to_last_swing else pd.NA

        if fallback_to_last_swing:
            if pd.isna(lo):
                lo = out["last_swing_low"].iat[i]
            if pd.isna(hi):
                hi = out["last_swing_high"].iat[i]

        pd_low.append(lo)
        pd_high.append(hi)

    out["pd_low"] = pd.to_numeric(pd_low, errors="coerce")
    out["pd_high"] = pd.to_numeric(pd_high, errors="coerce")
    return out
