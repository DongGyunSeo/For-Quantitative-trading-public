"""
fvg_lifecycle.py — 4H/1H/15m FVG 생애주기 상태머신 Tracker

FVG_LIFECYCLE_SPEC.md 구현. TF 독립적 (tf 파라미터로 4h/1h/15m 재사용).

상태: PENDING → ARMED → TOUCHED → USED(SOFT_DEAD) / IFVG / DEAD / EXPIRED
핵심:
- "충분히 떠남"은 봉수만 (횡보 FVG = noise 배제)
- fill_ratio 최대침투 기준 (단조 비가역)
- 50%+ filled → 진입 불가 (SOFT_DEAD)
- edge/mid 각 최초 1회만 timeout 리셋 (무한연장 방지)
- iFVG 변환: 관통+반전 단일 트리거 (진입 전/후 공통)
- USED = SOFT_DEAD (제거 안 함, iFVG 변환 후보로 추적 지속)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


# 상태 상수
PENDING = "PENDING"
ARMED = "ARMED"
TOUCHED = "TOUCHED"
SOFT_DEAD = "SOFT_DEAD"   # USED 또는 50%+ filled. 진입 불가, iFVG 감시 지속
IFVG = "IFVG"             # 관통+반전으로 변환됨 (변환 직후 ARMED 로직 재적용)
EXPIRED = "EXPIRED"       # timeout/queue 제거 대상


@dataclass
class FVGState:
    fvg_id: int
    fvg_type: str            # 'bull' / 'bear'
    created_idx: int         # 생성 TF봉 인덱스 (timeout 기준, 터치 리셋 시 갱신)
    origin_idx: int          # 최초 생성 인덱스 (불변, 진단용)
    zone_low: float
    zone_high: float
    size: float
    state: str = PENDING
    fill_ratio: float = 0.0  # 최대침투 (단조)
    touched_idx: int = -1    # 최초 ARMED-후 터치 봉 (진입맥락용)
    first_touch_idx: int = -1  # 최초 zone 접촉 봉 (상태 무관, fill 경로 포함; diag용)
    touch_count: int = 0
    touched_edge_once: bool = False  # edge 첫 터치 timeout 리셋 사용 여부
    touched_mid_once: bool = False   # mid 첫 터치 timeout 리셋 사용 여부
    used: bool = False       # 진입에 사용됨
    ifvg_generation: int = 0 # iFVG 변환 횟수 (디버그)

    @property
    def mid(self) -> float:
        return (self.zone_low + self.zone_high) / 2.0


class FVGLifecycleTracker:
    def __init__(self, tf: str = "4h",
                 n_depart: int = 5, n_depart_arm: int = 4,
                 timeout_bars: int = 360, fill_dead: float = 0.5,
                 min_size_atr: float = 0.3, max_slots: int = 20,
                 ifvg_require_close_beyond: bool = True):
        self.tf = tf
        self.n_depart = n_depart
        self.n_depart_arm = n_depart_arm
        self.timeout_bars = timeout_bars
        self.fill_dead = fill_dead
        self.min_size_atr = min_size_atr
        self.max_slots = max_slots
        self.ifvg_require_close_beyond = ifvg_require_close_beyond
        self._fvgs: List[FVGState] = []
        self._next_id = 0
        self._known_origins = set()  # 중복 생성 방지 (origin_idx + type)

    # ──────────────────────────────────────────────────────────
    def add_new_fvgs(self, new_fvgs, atr: float, cur_idx: int):
        """detect_fvgs_lookback 결과(DataFrame 또는 dict 리스트)에서
        아직 추적 안 된 FVG를 min_size 필터 후 PENDING으로 추가."""
        if new_fvgs is None or atr <= 0:
            return
        rows = new_fvgs.to_dict("records") if hasattr(new_fvgs, "to_dict") else new_fvgs
        min_size = self.min_size_atr * atr
        for r in rows:
            cidx = int(r["created_idx"])
            ftype = r["fvg_type"]
            key = (cidx, ftype)
            if key in self._known_origins:
                continue
            size = float(r["size"])
            if size < min_size:
                continue
            self._known_origins.add(key)
            self._fvgs.append(FVGState(
                fvg_id=self._next_id, fvg_type=ftype,
                created_idx=cidx, origin_idx=cidx,
                zone_low=float(r["zone_low"]), zone_high=float(r["zone_high"]),
                size=size, state=PENDING,
            ))
            self._next_id += 1

    # ──────────────────────────────────────────────────────────
    def update(self, cur_idx: int, bar_low: float, bar_high: float,
               bar_close: float):
        """매 TF봉 close마다 호출: fill 갱신 → 상태 전이 → 만료/queue."""
        for f in self._fvgs:
            if f.state in (EXPIRED,):
                continue
            self._update_fill(f, bar_low, bar_high)
            # first_touch_idx: 상태 전이 무관 최초 zone 접촉 (50%+fill→SOFT_DEAD
            # 직행 경로 포함). 기존 touched_idx는 ARMED후만 찍혀 99.5% -1이었음.
            if f.first_touch_idx < 0 and self._touches(f, bar_low, bar_high):
                f.first_touch_idx = cur_idx
            self._transition(f, cur_idx, bar_low, bar_high, bar_close)
        self._expire(cur_idx)
        self._enforce_slots()

    def _update_fill(self, f: FVGState, bar_low: float, bar_high: float):
        if f.size <= 0:
            return
        if f.fvg_type == "bull":
            pen = (f.zone_high - bar_low) / f.size
        else:
            pen = (bar_high - f.zone_low) / f.size
        pen = max(0.0, min(1.0, pen))
        if pen > f.fill_ratio:
            f.fill_ratio = pen

    def _touches(self, f: FVGState, bar_low: float, bar_high: float) -> bool:
        # edge 터치 인정: 봉 범위가 zone과 겹침
        return (bar_high >= f.zone_low) and (bar_low <= f.zone_high)

    def _crosses_inverted(self, f: FVGState, bar_close: float) -> bool:
        # 관통+반전: bull FVG는 zone_low 아래 종가, bear는 zone_high 위 종가
        if f.fvg_type == "bull":
            return bar_close < f.zone_low
        else:
            return bar_close > f.zone_high

    def _transition(self, f: FVGState, cur_idx: int,
                    bar_low: float, bar_high: float, bar_close: float):
        # 1) iFVG 변환은 모든 살아있는 상태에서 우선 검사 (관통+반전 단일 트리거)
        if f.state in (ARMED, TOUCHED, SOFT_DEAD):
            if self._crosses_inverted(f, bar_close):
                self._convert_ifvg(f, cur_idx)
                return

        # 2) 50%+ filled → SOFT_DEAD (진입 불가, iFVG 감시 지속)
        if f.state in (PENDING, ARMED, TOUCHED) and f.fill_ratio >= self.fill_dead:
            f.state = SOFT_DEAD
            return

        # 3) PENDING → ARMED (떠남 봉수)
        if f.state == PENDING:
            age = cur_idx - f.created_idx
            if age >= self.n_depart_arm:
                f.state = ARMED
            # ARMED 전 터치는 무효 → 별도 처리 안 함 (그냥 PENDING 유지)
            return

        # 4) ARMED → TOUCHED (+ edge/mid 최초 터치 timeout 리셋)
        if f.state == ARMED:
            if self._touches(f, bar_low, bar_high):
                if f.touched_idx < 0:
                    f.touched_idx = cur_idx
                f.touch_count += 1
                f.state = TOUCHED
                self._maybe_reset_timeout(f, cur_idx, bar_low, bar_high)
            return

        # 5) TOUCHED: 재터치 시 timeout 리셋 가능 (edge/mid 최초 1회)
        if f.state == TOUCHED:
            if self._touches(f, bar_low, bar_high):
                f.touch_count += 1
                self._maybe_reset_timeout(f, cur_idx, bar_low, bar_high)
            # TOUCHED → USED 는 외부 mark_used()로만. 여기선 유지.
            return

    def _maybe_reset_timeout(self, f: FVGState, cur_idx: int,
                             bar_low: float, bar_high: float):
        """edge 최초 터치 1회 + mid 최초 터치 1회만 created_idx 갱신."""
        # edge 터치 판정: zone 경계 근처 도달 (이미 _touches True인 상태)
        if not f.touched_edge_once:
            f.touched_edge_once = True
            f.created_idx = cur_idx  # timeout 연장
            return
        # mid 터치 판정: 봉이 mid를 관통
        if not f.touched_mid_once:
            if bar_low <= f.mid <= bar_high:
                f.touched_mid_once = True
                f.created_idx = cur_idx  # timeout 연장

    def _convert_ifvg(self, f: FVGState, cur_idx: int):
        """관통+반전 → 반대 방향 FVG로 변환, ARMED 로직 재적용."""
        f.fvg_type = "bear" if f.fvg_type == "bull" else "bull"
        # zone 유지 (관통된 구간이 반대 의미). 필요시 재정의 가능.
        f.state = PENDING          # 봉수 count 리셋 → 다시 ARMED 절차
        f.created_idx = cur_idx
        f.origin_idx = cur_idx
        f.fill_ratio = 0.0
        f.touched_idx = -1
        f.first_touch_idx = -1
        f.touch_count = 0
        f.touched_edge_once = False
        f.touched_mid_once = False
        f.used = False
        f.ifvg_generation += 1

    def _expire(self, cur_idx: int):
        for f in self._fvgs:
            if f.state == EXPIRED:
                continue
            if cur_idx - f.created_idx >= self.timeout_bars:
                f.state = EXPIRED
        # EXPIRED 제거
        self._fvgs = [f for f in self._fvgs if f.state != EXPIRED]

    def _enforce_slots(self):
        # 살아있는(추적 중) FVG가 max_slots 초과 시 가장 오래된 것 제거
        alive = [f for f in self._fvgs if f.state != EXPIRED]
        if len(alive) > self.max_slots:
            # created_idx 오래된 순 정렬, 초과분 제거
            alive.sort(key=lambda x: x.created_idx)
            keep = set(id(f) for f in alive[-self.max_slots:])
            self._fvgs = [f for f in self._fvgs if id(f) in keep]

    # ──────────────────────────────────────────────────────────
    def mark_used(self, fvg_id: int):
        """5m 진입이 이 FVG를 사용 → SOFT_DEAD (제거 안 함, iFVG 감시 지속)."""
        for f in self._fvgs:
            if f.fvg_id == fvg_id:
                f.used = True
                f.state = SOFT_DEAD
                return

    def get_active(self) -> List[FVGState]:
        """진입 맥락 판단용: ARMED/TOUCHED 상태 (진입 가능) FVG."""
        return [f for f in self._fvgs if f.state in (ARMED, TOUCHED)]

    def context_at(self, price: float, direction: str,
                   atr: float) -> Dict[str, float]:
        """price/direction에 대한 4H FVG 맥락 feature dict."""
        active = self.get_active()
        out = {
            "in_4h_fvg": 0.0,
            "in_4h_fvg_aligned": 0.0,
            "nearest_4h_fvg_dist_atr": np.nan,
            "active_4h_fvg_fill": np.nan,
            "n_active_4h_fvg": float(len(active)),
        }
        if not active:
            return out
        best_dist = np.inf
        for f in active:
            inside = f.zone_low <= price <= f.zone_high
            if inside:
                out["in_4h_fvg"] = 1.0
                out["active_4h_fvg_fill"] = f.fill_ratio
                if (f.fvg_type == "bull" and direction == "long") or \
                   (f.fvg_type == "bear" and direction == "short"):
                    out["in_4h_fvg_aligned"] = 1.0
            # 가장 가까운 edge 거리
            d = min(abs(price - f.zone_low), abs(price - f.zone_high))
            if inside:
                d = 0.0
            if d < best_dist:
                best_dist = d
        if np.isfinite(best_dist) and atr > 0:
            out["nearest_4h_fvg_dist_atr"] = best_dist / atr
        return out

    def diagnose_at(self, price: float, direction: str, atr: float,
                    cur_idx: int, sweep_extreme: float = None) -> Dict[str, float]:
        """진단용: 진입가에 대한 같은방향 FVG 상세 (거리/상태/터치후봉수/침투fib/zone위치).
        가장 가까운 같은방향(살아있는) FVG 기준. 4H FVG 약한 원인 분석용.
        반환: 모든 살아있는 FVG 중 같은방향만 보고, 가장 가까운 것의 상세.
        - dist_atr: edge 거리 (zone 안이면 0)
        - state_code: 0=없음 1=PENDING 2=ARMED 3=TOUCHED 4=SOFT_DEAD 5=IFVG
        - bars_since_touch: cur_idx - touched_idx (터치 후 경과 4H봉; -1=미터치)
        - fill_ratio: 최대 침투 (extreme fib, 0~1+)
        - zone_pos: 진입가의 zone 내 위치 (0=먼edge, 1=깊은edge; zone밖이면 <0 or >1)
        - size_atr: FVG 크기 / atr
        - n_same_dir_alive: 같은방향 살아있는 FVG 수
        """
        want = "bull" if direction == "long" else "bear"
        alive = [f for f in self._fvgs
                 if f.state != EXPIRED and f.fvg_type == want]
        out = {
            "diag_dist_atr": np.nan, "diag_state_code": 0.0,
            "diag_bars_since_touch": -1.0,
            "diag_bars_since_first_touch": -1.0,
            "diag_fill_ratio": np.nan,
            "diag_zone_pos": np.nan, "diag_size_atr": np.nan,
            "diag_n_same_dir_alive": float(len(alive)),
            "diag_sweep_zone_pos": np.nan,
            "diag_sweep_dist_atr": np.nan,
            "diag_sweep_in_zone": np.nan,
        }
        if not alive or atr <= 0:
            return out
        _code = {PENDING: 1.0, ARMED: 2.0, TOUCHED: 3.0, SOFT_DEAD: 4.0, IFVG: 5.0}
        best = None; best_d = np.inf
        for f in alive:
            inside = f.zone_low <= price <= f.zone_high
            d = 0.0 if inside else min(abs(price - f.zone_low), abs(price - f.zone_high))
            if d < best_d:
                best_d = d; best = f
        if best is None:
            return out
        rng = max(best.zone_high - best.zone_low, 1e-9)
        # zone_pos: long이면 zone_low가 깊은쪽(되돌림 깊음). 방향별 정규화.
        if direction == "long":
            zpos = (price - best.zone_low) / rng       # 0=high(얕음) 1=low(깊음)... 뒤집어:
            zpos = 1.0 - zpos                          # 1=깊은 침투(zone_low쪽), 0=얕은(zone_high)
        else:
            zpos = (price - best.zone_low) / rng       # short: 1=깊은(zone_high쪽)
        out["diag_dist_atr"] = best_d / atr
        out["diag_state_code"] = _code.get(best.state, 0.0)
        out["diag_bars_since_touch"] = float(cur_idx - best.touched_idx) if best.touched_idx >= 0 else -1.0
        out["diag_bars_since_first_touch"] = (
            float(cur_idx - best.first_touch_idx) if best.first_touch_idx >= 0 else -1.0)
        out["diag_fill_ratio"] = best.fill_ratio
        out["diag_zone_pos"] = zpos
        out["diag_size_atr"] = best.size / atr

        # ── sweep extreme 기준 (ICT 명제의 올바른 측정: 진입가 아닌 sweep wick이
        #    FVG를 찔렀는가). entry는 reclaim 후라 FVG 위로 떠 80% zone밖 착시 가능.
        if sweep_extreme is not None:
            s_best = None; s_bd = np.inf
            for f2 in alive:
                s_in = f2.zone_low <= sweep_extreme <= f2.zone_high
                sd = 0.0 if s_in else min(abs(sweep_extreme - f2.zone_low),
                                          abs(sweep_extreme - f2.zone_high))
                if sd < s_bd:
                    s_bd = sd; s_best = f2
            if s_best is not None:
                s_rng = max(s_best.zone_high - s_best.zone_low, 1e-9)
                if direction == "long":
                    s_zpos = 1.0 - (sweep_extreme - s_best.zone_low) / s_rng
                else:
                    s_zpos = (sweep_extreme - s_best.zone_low) / s_rng
                out["diag_sweep_zone_pos"] = s_zpos
                out["diag_sweep_dist_atr"] = s_bd / atr
                out["diag_sweep_in_zone"] = 1.0 if (s_best.zone_low <= sweep_extreme
                                                    <= s_best.zone_high) else 0.0
        return out

    def confluence_with(self, liq_price: float, liq_dir: str,
                        atr: float = 0.0, max_dist_atr: float = 0.0):
        """기존 liquidity 가격이 같은 방향 ARMED FVG zone에 실제로 닿는가(포함).
        long: bull FVG zone 안에 liq(SSL) 가격 / short: bear FVG zone 안에 liq(BSL).
        '근처(거리 허용)'가 아니라 zone 포함만 결합으로 인정 — noise 방지.
        atr/max_dist_atr 인자는 호환성 위해 남기되 미사용.
        결합되는 FVG 반환 (zone 폭 가장 작은 = 가장 타이트한 것 우선, 없으면 None)."""
        best = None
        best_size = np.inf
        for f in self.get_active():
            # 방향 정합
            if liq_dir == "long" and f.fvg_type != "bull":
                continue
            if liq_dir == "short" and f.fvg_type != "bear":
                continue
            # liquidity 가격이 FVG zone에 실제로 포함되는가 (닿음)
            if f.zone_low <= liq_price <= f.zone_high:
                if f.size < best_size:
                    best_size = f.size
                    best = f
        return best
