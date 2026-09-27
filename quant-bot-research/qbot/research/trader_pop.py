"""합성 트레이더 집단 — 지표마다 자기 논리의 손절/청산을 갖는 가상 참가자들.

왜 만드는가
-----------
"사람들이 어디서 물려 있는가"를 재려면 **포지션 상태**가 필요하다. 상태 없는
가격 함수(롤링 극단 돌파 같은 것)로는 경로 의존성을 담을 수 없다. 지표마다
진입·손절·청산 규칙을 주고 5m 축에서 동시에 돌린 뒤, 봉마다 집단 상태를 집계한다.

설계 원칙
---------
1. **손절은 지표 고유 논리로.** GitHub 16개 구현을 열어 확인한 결과 오픈소스는
   거의 전부 (a) ATR 배수 (b) 고정 % (c) 무손절 셋 중 하나였다. "신호를 만든
   구조가 깨지면 손절"은 0개였다 — 그래서 구조 손절만은 자체 설계다.
2. **HTF 신호는 확정된 뒤에만 쓴다.** `build_ref_map` 이 강제한다.
3. **에피소드 단위로 벡터화.** 세그먼트 오프셋 트릭으로 O(n) 에 구간별 누적
   최소/최대를 낸다. 65명 × 55만봉이 10초대.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..data.resample import build_ref_map, resample_htf

__all__ = [
    "StopSpec", "ExitSpec", "Archetype", "PopConfig", "TraderState", "PainResult",
    "CANON", "TFS", "LEVELS", "EXIT_REASONS", "PAIN_CAP_R",
    "signal_series", "run_trader", "population_state", "canon_archetypes",
    "archetypes_for_level", "segment_cummin", "segment_cummax", "segment_cumor",
    "first_hit_per_segment", "positions_frame", "Position",
]

EXIT_REASONS = ("손절", "신호반전", "지표청산", "시간초과", "미청산")
#: 고통 포화 지점. 분해 결과 실제 정보는 0.25~0.75R 에만 있었다(문서 참조).
PAIN_CAP_R = 3.0
TFS = ("5m", "15m", "1h", "4h", "1d")
#: NEP 50 — 파이썬 int 와 int64 를 섞으면 오버플로가 조용히 생긴다.
BIG = np.int64(1) << np.int64(40)


# ---------------------------------------------------------------- 세그먼트 원자


def _seg_offsets(n: int, starts: np.ndarray, scale: float) -> np.ndarray:
    """세그먼트마다 단조 증가하는 오프셋. 길이 n."""
    sid = np.zeros(n, np.int64)
    if len(starts):
        sid[starts] = 1
        sid = np.cumsum(sid) - 1
    return sid.astype(np.float64) * scale


def _scale_for(x: np.ndarray) -> float:
    fin = x[np.isfinite(x)]
    if len(fin) == 0:
        return 1.0
    span = float(fin.max() - fin.min())
    return (span + abs(float(fin.max())) + abs(float(fin.min())) + 1.0) * 4.0


def segment_cummin(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """구간별 누적 최소. `starts` 는 x 자신의 인덱스 공간.

    구간마다 **증가하는** 오프셋을 빼면 뒤 구간 값이 앞 구간보다 통째로 낮아져
    전역 `minimum.accumulate` 가 구간 경계에서 저절로 리셋된다.
    """
    off = _seg_offsets(len(x), starts, _scale_for(x))
    return np.minimum.accumulate(x - off) + off


def segment_cummax(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """구간별 누적 최대. 부호가 cummin 과 반대다(+off / −off)."""
    off = _seg_offsets(len(x), starts, _scale_for(x))
    return np.maximum.accumulate(x + off) - off


def segment_cumor(mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """구간별 누적 OR. 비단조 조건을 단조화해 같은 경로로 태우기 위한 것."""
    return segment_cummax(np.asarray(mask, np.float64), starts) > 0.5


def first_hit_per_segment(mask: np.ndarray, starts: np.ndarray,
                          ends: np.ndarray) -> np.ndarray:
    """구간마다 mask 가 처음 True 인 **절대 위치**. 없으면 −1."""
    n = len(mask)
    ar = np.arange(n, dtype=np.int64)
    cand = np.where(mask, ar, BIG)
    if len(starts) == 0:
        return np.empty(0, np.int64)
    out = np.minimum.reduceat(cand, starts)
    # reduceat 은 마지막 구간을 배열 끝까지 본다 — 구간 끝을 넘은 히트는 무효
    bad = (out >= BIG) | (out >= ends)
    return np.where(bad, -1, out)


def _ranges(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """여러 [s,e) 구간을 이어 붙인 절대 위치 배열과, 그 안에서의 구간 시작 오프셋."""
    lens = (ends - starts).astype(np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    off = np.cumsum(lens) - lens
    pos = np.repeat(starts, lens) + (np.arange(tot, dtype=np.int64) - np.repeat(off, lens))
    return pos, off


def _neutral(x: np.ndarray, *, for_max: bool) -> np.ndarray:
    """NaN 을 "절대 이기지 못하는 **유한**값" 으로 바꾼다.

    ±inf 를 쓰면 세그먼트 오프셋 트릭에서 ``inf − inf = nan`` 이 되어 전부 망가진다.
    """
    fin = np.isfinite(x)
    if not fin.any():
        return np.zeros_like(x)
    lo_, hi_ = float(x[fin].min()), float(x[fin].max())
    span = (hi_ - lo_) + abs(hi_) + abs(lo_) + 1.0
    fill = lo_ - span if for_max else hi_ + span
    return np.where(fin, x, fill)


# ---------------------------------------------------------------- 지표


def _sma(x: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(x).rolling(p, min_periods=p).mean().to_numpy()


def _ema(x: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(x).ewm(span=p, adjust=False).mean().to_numpy()


def _rsi(x: np.ndarray, p: int) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    up = pd.Series(np.maximum(d, 0.0)).ewm(alpha=1 / p, adjust=False).mean().to_numpy()
    dn = pd.Series(np.maximum(-d, 0.0)).ewm(alpha=1 / p, adjust=False).mean().to_numpy()
    rs = np.where(dn > 0, up / np.where(dn > 0, dn, 1.0), np.inf)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[:p] = np.nan
    return out


def _sign(x: np.ndarray) -> np.ndarray:
    s = np.zeros(len(x), np.int8)
    s[x > 0] = 1
    s[x < 0] = -1
    s[~np.isfinite(x)] = 0
    return s


def _atr(high, low, close, p: int) -> np.ndarray:
    pc = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).ewm(alpha=1 / p, adjust=False).mean().to_numpy()


def _pivot_levels(high: np.ndarray, low: np.ndarray, lb: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """각 시점에서 **이미 확정된** 마지막 스윙 고/저 가격.

    프랙탈 스윙은 뒤 ``lb`` 개 봉이 있어야 확정되므로, i 번 봉의 스윙은
    ``i + lb`` 시점부터만 알 수 있다. shift 로 그 지연을 구조적으로 넣는다.
    """
    from .choch import swing_points

    is_sh, is_sl = swing_points(high, low, lb)
    hv = pd.Series(np.where(is_sh, high, np.nan)).shift(lb).ffill().to_numpy()
    lv = pd.Series(np.where(is_sl, low, np.nan)).shift(lb).ffill().to_numpy()
    return hv, lv


def signal_series(htf: pd.DataFrame, kind: str, params: tuple) -> np.ndarray:
    """HTF 봉 종가 기준 신호 ∈ {-1,0,+1}. 길이 = len(htf)."""
    c = htf["close"].to_numpy(dtype=np.float64)
    h = htf["high"].to_numpy(dtype=np.float64)
    low = htf["low"].to_numpy(dtype=np.float64)

    if kind == "ma_cross":
        f, s = params
        return _sign(_sma(c, f) - _sma(c, s))
    if kind == "ema_cross":
        f, s = params
        out = _sign(_ema(c, f) - _ema(c, s))
        out[: max(f, s)] = 0
        return out
    if kind == "ma_side":
        (p,) = params
        return _sign(c - _sma(c, p))
    if kind == "rsi50":
        (p,) = params
        return _sign(_rsi(c, p) - 50.0)
    if kind == "macd_osc":
        f, s, sg = params
        macd = _ema(c, f) - _ema(c, s)
        hist = macd - _ema(macd, sg)
        out = _sign(hist)
        out[: s + sg] = 0
        return out
    if kind == "donchian":
        (p,) = params
        hh = pd.Series(h).rolling(p, min_periods=p).max().shift(1).to_numpy()
        ll = pd.Series(low).rolling(p, min_periods=p).min().shift(1).to_numpy()
        raw = np.zeros(len(c), dtype=np.float64)
        raw[c > hh] = 1.0
        raw[c < ll] = -1.0
        raw[raw == 0] = np.nan
        held = pd.Series(raw).ffill().to_numpy().copy()
        held[: p + 1] = np.nan
        return _sign(held)
    if kind == "rsi_band":
        # RSI 평균회귀: 하단 밴드에서 롱, 상단 밴드에서 숏. 밴드 밖에서는 유지(히스테리시스)
        pr, blo, bhi = params
        r = _rsi(c, pr)
        raw = np.full(len(c), np.nan)
        raw[r < blo] = 1.0
        raw[r > bhi] = -1.0
        held = pd.Series(raw).ffill().to_numpy().copy()
        held[:pr] = np.nan
        return _sign(held)
    if kind == "rsi2_sma":
        # Connors RSI(2): 장기 MA 위에서 RSI(2) 과매도만 롱
        pr, blo, bhi, ma = params
        r = _rsi(c, pr)
        trend = _sma(c, ma)
        raw = np.full(len(c), np.nan)
        raw[(r < blo) & (c > trend)] = 1.0
        raw[(r > bhi) & (c < trend)] = -1.0
        held = pd.Series(raw).ffill().to_numpy().copy()
        held[:ma] = np.nan
        return _sign(held)
    if kind == "pivot_break":
        # 확정 스윙(프랙탈) 레벨 돌파. 확정은 i+lb 시점이므로 lb 만큼 뒤로 민다.
        (lb,) = params
        piv_hi, piv_lo = _pivot_levels(h, low, lb)
        raw = np.full(len(c), np.nan)
        raw[c > piv_hi] = 1.0
        raw[c < piv_lo] = -1.0
        held = pd.Series(raw).ffill().to_numpy().copy()
        held[: 2 * lb + 1] = np.nan
        return _sign(held)
    raise ValueError(f"알 수 없는 지표: {kind}")


# ---------------------------------------------------------------- 명세


@dataclass(frozen=True, slots=True)
class StopSpec:
    """손절 규칙.

    kind
        ``struct``   진입 시점의 직전 ``lookback`` 봉 극단 (HTF 축)
        ``pivot``    확정 스윙 ∪ 그 이후 레그 극단 중 **깊은 쪽**
        ``ma``       느린 MA 레벨 (봉마다 갱신되는 동적 손절)
        ``atr``      진입가 ∓ ``mult`` × ATR(``period``)
        ``atr_trail`` 종가 ∓ ``mult`` × ATR 의 래칫(되돌아가지 않는다)
        ``chandelier`` 에피소드 최고가 ∓ ``mult`` × ATR
        ``pct``      진입가 × (1 ∓ ``pct``)
        ``none``     손절 없음
    """
    kind: str = "struct"
    lookback: int = 10
    mult: float = 2.0
    period: int = 14
    ma: int = 60
    pct: float = 0.10


@dataclass(frozen=True, slots=True)
class ExitSpec:
    """청산 규칙.

    kind
        ``flip``      신호 반전
        ``channel``   ``lookback`` 봉 반대 채널 돌파 (터틀)
        ``ma_touch``  종가가 SMA(``ma``) 를 되찾음 (Connors)
        ``none``      청산 없음 — 손절로만 끝난다
    time_bars
        HTF 봉 단위 시간 손절. 0 이면 없음.
    deadband
        신호 반전을 인정하는 최소 폭 (Lean 데드밴드).
    """
    kind: str = "flip"
    lookback: int = 10
    ma: int = 5
    time_bars: int = 0
    deadband: float = 0.0


@dataclass(frozen=True, slots=True)
class Archetype:
    kind: str
    params: tuple
    tf: str = "1h"
    sl_lookback: int = 10
    stop: StopSpec = field(default_factory=StopSpec)
    exit: ExitSpec = field(default_factory=ExitSpec)
    label: str = ""
    #: 합성 트레이더 — 여러 신호의 다수결
    combine: tuple = ()
    vote: int = 1

    @property
    def name(self) -> str:
        return f"{self.label or self.kind}@{self.tf}"


@dataclass(frozen=True, slots=True)
class PopConfig:
    sl_lookback: int = 10
    #: 진입 봉에서의 손절 히트도 인정할지 (True = 보수적)
    stop_on_entry_bar: bool = True
    #: 손절 없음(`none`) 일 때 강제 종료 지평 (5m 봉)
    max_hold: int = 20_000


#: 정본 카탈로그 — 라벨 · 지표 · 파라미터 · 손절 · 청산 · 출처
CANON: tuple[tuple[str, str, tuple, StopSpec, ExitSpec, str], ...] = (
    ("turtle20", "donchian", (20,),
     StopSpec("atr", mult=2.0, period=20), ExitSpec("channel", lookback=10),
     "Faith 원문 2N + jesse TurtleRules + pplonski/turtle-trading-python"),
    ("turtle55", "donchian", (55,),
     StopSpec("atr", mult=2.0, period=20), ExitSpec("channel", lookback=20),
     "Faith 원문 System 2"),
    ("macd_atr3", "macd_osc", (12, 26, 9),
     StopSpec("atr_trail", mult=3.0, period=14), ExitSpec("none"),
     "backtrader samples/macd-settings: pstop=max(pstop, close-3*ATR), 청산은 손절뿐"),
    ("macd_flip", "macd_osc", (12, 26, 9),
     StopSpec("none"), ExitSpec("flip"),
     "QuantConnect Lean MACDTrendAlgorithm: 손절 없음, 히스토그램 부호 반전"),
    ("ma_cross_struct", "ma_cross", (20, 60),
     StopSpec("ma", ma=60), ExitSpec("flip"),
     "손절=느린 MA 이탈(신호 구조 파괴). 청산=데드크로스 (Lean/jesse)"),
    ("ma_cross_bare", "ma_cross", (20, 60),
     StopSpec("none"), ExitSpec("flip"),
     "Lean MovingAverageCrossAlgorithm / jesse SMACrossover: 손절 없음"),
    ("ma_cross_pct", "ma_cross", (20, 60),
     StopSpec("pct", pct=0.10), ExitSpec("flip"),
     "freqtrade Strategy001: stoploss=-0.10 고정"),
    ("rsi_band", "rsi_band", (14, 30, 70),
     StopSpec("pivot", lookback=3), ExitSpec("flip", time_bars=14),
     "청산=반대밴드(freqtrade BbandRsi) / 손절=직전 확정 스윙(자체) / 시간손절=Lean RsiAlphaModel"),
    ("rsi_band_pct", "rsi_band", (14, 30, 70),
     StopSpec("pct", pct=0.25), ExitSpec("flip"),
     "freqtrade BbandRsi: stoploss=-0.25, RSI>70 청산"),
    ("rsi2_connors", "rsi2_sma", (2, 10, 90, 200),
     StopSpec("none"), ExitSpec("ma_touch", ma=5),
     "jesse RSI2 / Connors: 손절 없음, close>SMA(5) 청산"),
    ("rsi50_chand", "rsi50", (14,),
     StopSpec("chandelier", mult=3.0, period=14), ExitSpec("flip"),
     "freqtrade hlhb(RSI50 크로스 + 트레일) 를 샹들리에로 구현"),
    ("ma120_side", "ma_side", (120,),
     StopSpec("atr", mult=2.0, period=14), ExitSpec("flip"),
     "MA120 추세추종 + 2xATR 손절. 청산은 MA 되돌파"),
    ("pivot_break", "pivot_break", (3,),
     StopSpec("pivot", lookback=3), ExitSpec("flip"),
     "지지저항 돌파 -> 돌파한 레벨 아래 손절 (구조 파괴). 오픈소스 선례 없음, 자체 설계"),
)

LEVELS: dict[str, tuple] = {
    "L1": (("ma_cross", (20, 60)),),
    "L2": (("ma_cross", (20, 60)), ("rsi50", (14,))),
    "L3": (("ma_cross", (20, 60)), ("rsi50", (14,)), ("macd_osc", (12, 26, 9))),
    "L4": (("ma_cross", (20, 60)), ("rsi50", (14,)), ("macd_osc", (12, 26, 9)),
           ("donchian", (20,)), ("ma_side", (120,))),
}


def canon_archetypes(tfs: tuple[str, ...] = TFS,
                     only: tuple[str, ...] | None = None) -> list[Archetype]:
    """정본 카탈로그 × 타임프레임 -> 아키타입 목록."""
    out = []
    for label, kind, params, stop, exit_, _src in CANON:
        if only and label not in only:
            continue
        for tf in tfs:
            out.append(Archetype(kind, params, tf, stop=stop, exit=exit_, label=label))
    return out


def archetypes_for_level(level: str, tfs: tuple[str, ...] = TFS) -> list[Archetype]:
    """레벨 이름 -> 아키타입 목록. L5 는 L4 + 합성 트레이더."""
    if level == "L5":
        out = archetypes_for_level("L4", tfs)
        for tf in tfs:
            out.append(Archetype("ma_cross", (20, 60), tf,
                                 combine=(Archetype("rsi50", (14,), tf),
                                          Archetype("macd_osc", (12, 26, 9), tf)),
                                 vote=2))
        for tf, up in (("5m", "1h"), ("15m", "4h"), ("1h", "1d")):
            if tf in tfs:
                out.append(Archetype("ma_cross", (20, 60), tf,
                                     combine=(Archetype("ma_cross", (20, 60), up),),
                                     vote=2))
        return out
    specs = LEVELS[level]
    return [Archetype(k, p, tf) for tf in tfs for k, p in specs]


# ---------------------------------------------------------------- 실행


@dataclass(slots=True)
class TraderState:
    """한 트레이더의 결과. 에피소드 배열 + 5m 축 확장 배열."""
    #: 에피소드 시작 / 끝(exclusive) / 방향 / 진입가 / 최초 손절가 / 청산 사유 코드
    ep_start: np.ndarray
    ep_end: np.ndarray
    ep_side: np.ndarray
    ep_entry: np.ndarray
    ep_stop0: np.ndarray
    ep_exit: np.ndarray
    #: 5m 축: 살아 있는 방향 (+1/−1/0)
    side5: np.ndarray
    #: 5m 축: 그 시점의 손절가 (죽어 있으면 NaN)
    stop_px5: np.ndarray

    def __len__(self) -> int:
        return len(self.ep_start)


def _expand(vals: np.ndarray, starts: np.ndarray, ends: np.ndarray, n: int,
            fill: float = np.nan) -> np.ndarray:
    """에피소드 단위 값을 5m 축으로 펼친다."""
    out = np.full(n, fill, np.float64)
    pos, _ = _ranges(starts, ends)
    if len(pos):
        out[pos] = np.repeat(np.asarray(vals, np.float64), (ends - starts))
    return out


def _tf_ratio(tf: str) -> int:
    m = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "1d": 288}
    return m.get(tf.strip().lower(), 1)


def _map5(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """HTF 축 배열을 5m 축으로. ``ref < 0`` (확정 전) 은 NaN."""
    out = np.full(len(ref), np.nan)
    ok = ref >= 0
    out[ok] = np.asarray(x, np.float64)[ref[ok]]
    return out


def _as_htf(ltf: pd.DataFrame) -> pd.DataFrame:
    """5m 자체를 HTF 로 취급 (close_ts/n_ltf 컬럼만 붙인다)."""
    out = ltf[["open", "high", "low", "close", "volume"]].copy()
    step = ltf.index[1] - ltf.index[0] if len(ltf) > 1 else pd.Timedelta("5min")
    out["close_ts"] = ltf.index + step
    out["n_ltf"] = 1
    return out


def _htf_ref(ltf: pd.DataFrame, tf: str, cache: dict) -> tuple[pd.DataFrame, np.ndarray]:
    key = ("htf", tf)
    if key not in cache:
        htf = _as_htf(ltf) if tf.strip().lower() == "5m" else resample_htf(ltf, tf)
        cache[key] = (htf, build_ref_map(ltf.index, htf))
    return cache[key]


def _signal_on_5m(ltf: pd.DataFrame, arch: Archetype, cache: dict
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(5m 축 신호, 5m 축 롱손절가 후보, 5m 축 숏손절가 후보).

    손절가 후보는 "그 5m 시점에 확정된 마지막 HTF 봉까지의 직전 lookback 봉 극단".
    """
    htf, ref = _htf_ref(ltf, arch.tf, cache)
    sig = signal_series(htf, arch.kind, arch.params)
    lb = arch.stop.lookback if arch.stop.kind == "struct" else arch.sl_lookback
    ll = pd.Series(htf["low"].to_numpy()).rolling(lb, min_periods=1).min().to_numpy()
    hh = pd.Series(htf["high"].to_numpy()).rolling(lb, min_periods=1).max().to_numpy()
    s5 = np.nan_to_num(_map5(sig.astype(np.float64), ref)).astype(np.int8)
    return s5, _map5(ll, ref), _map5(hh, ref)


def _exit_condition(ltf: pd.DataFrame, arch: Archetype, cache: dict) -> np.ndarray | None:
    """5m 축 청산 조건 (롱/숏 각각). None 이면 flip/none 만 쓴다."""
    k = arch.exit.kind
    if k in ("flip", "none"):
        return None
    htf, ref = _htf_ref(ltf, arch.tf, cache)
    c = htf["close"].to_numpy(np.float64)
    if k == "channel":
        p = arch.exit.lookback
        ll = pd.Series(htf["low"].to_numpy()).rolling(p, min_periods=p).min().shift(1).to_numpy()
        hh = pd.Series(htf["high"].to_numpy()).rolling(p, min_periods=p).max().shift(1).to_numpy()
        long_out = c < ll
        short_out = c > hh
    elif k == "ma_touch":
        m = _sma(c, arch.exit.ma)
        long_out = c > m
        short_out = c < m
    else:
        raise ValueError(f"알 수 없는 청산: {k}")
    lo5 = np.nan_to_num(_map5(long_out.astype(float), ref)) > 0.5
    sh5 = np.nan_to_num(_map5(short_out.astype(float), ref)) > 0.5
    return np.stack([lo5, sh5])


def _stop_series(arch, starts_all, ends_all, side_all, o, struct_lo, struct_hi,
                 aux, n) -> tuple[np.ndarray, np.ndarray]:
    """봉마다의 손절가(압축 축). 고정형이든 동적/트레일이든 같은 모양으로 낸다."""
    lens = ends_all - starts_all
    sd = np.repeat(side_all.astype(np.int64), lens)
    k = arch.stop
    if k.kind == "none":
        return np.where(sd > 0, -np.inf, np.inf), sd

    def at_entry(arr):
        """진입 시점 값을 구간 전체로 편다. ``arr`` 은 **전체 길이** 배열이어야 한다.

        압축 축 배열(``close5`` 등)과 전체 길이 배열(``atr_full`` 등)을 섞지 말 것 —
        재구축 중 여기서 IndexError 로 한 번 걸렸다.
        """
        return np.repeat(np.asarray(arr)[starts_all], lens)

    _, off = _ranges(starts_all, ends_all)
    ent = at_entry(o)
    if k.kind == "struct":
        base = np.where(sd > 0, at_entry(struct_lo), at_entry(struct_hi))
    elif k.kind == "pivot":
        base = np.where(sd > 0, at_entry(aux["piv_lo5"]), at_entry(aux["piv_hi5"]))
    elif k.kind == "ma":
        base = aux["ma5"]
    elif k.kind == "atr":
        a = at_entry(aux["atr_full"])          # 진입 시점 ATR 로 고정
        base = np.where(sd > 0, ent - k.mult * a, ent + k.mult * a)
    elif k.kind == "pct":
        base = np.where(sd > 0, ent * (1.0 - k.pct), ent * (1.0 + k.pct))
    elif k.kind == "atr_trail":
        up_raw = aux["close5"] - k.mult * aux["atr5"]
        dn_raw = aux["close5"] + k.mult * aux["atr5"]
        base = np.where(sd > 0,
                        segment_cummax(_neutral(up_raw, for_max=True), off),
                        segment_cummin(_neutral(dn_raw, for_max=False), off))
    elif k.kind == "chandelier":
        hi_r = segment_cummax(_neutral(aux["high5"], for_max=True), off)
        lo_r = segment_cummin(_neutral(aux["low5"], for_max=False), off)
        a = aux["atr5"]
        base = np.where(sd > 0, hi_r - k.mult * a, lo_r + k.mult * a)
    else:
        raise ValueError(f"알 수 없는 손절: {k.kind}")

    return base, sd


def _raw_episodes(s5: np.ndarray, cond: np.ndarray | None, tb5: int, n: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """신호 런 -> (시작, 끝exclusive, 방향, 종료사유코드).

    종료 사유는 아직 손절을 보지 않은 **예비** 값이다 (1=신호반전, 2=지표청산,
    3=시간초과, 4=미청산). 손절은 나중에 덮어쓴다.
    """
    prev = np.concatenate(([np.int8(0)], s5[:-1]))
    starts = np.flatnonzero((s5 != prev) & (s5 != 0))
    if len(starts) == 0:
        z = np.empty(0, np.int64)
        return z, z, np.empty(0, np.int8), z
    side = s5[starts]
    # 신호가 바뀌는 다음 지점
    chg = np.flatnonzero(s5 != prev)
    nxt = np.searchsorted(chg, starts, side="right")
    flip_end = np.where(nxt < len(chg), chg[np.minimum(nxt, len(chg) - 1)], n)
    flip_end = np.where(nxt < len(chg), flip_end, n).astype(np.int64)
    ends = flip_end.copy()
    why = np.ones(len(starts), np.int64)          # 신호반전
    why[ends >= n] = 4                            # 미청산

    if cond is not None:
        ar = np.arange(n, dtype=np.int64)
        for k, s in enumerate(starts):
            row = 0 if side[k] > 0 else 1
            e = int(ends[k])
            seg = cond[row, s:e]
            if seg.any():
                ends[k] = s + int(seg.argmax()) + 1
                why[k] = 2
    if tb5 > 0:
        cap = starts + tb5
        hit = cap < ends
        ends = np.where(hit, cap, ends)
        why = np.where(hit, 3, why)
    return starts.astype(np.int64), ends.astype(np.int64), side, why



def _stop_aux(ltf: pd.DataFrame, arch: Archetype, cache: dict) -> dict:
    """손절 재료(ATR · MA · 피벗)를 **그 트레이더의 TF 에서** 만들어 5m 축으로 편다.

    이걸 5m 에서 만들면 1h 트레이더의 손절이 5분봉에 달라붙어 즉사한다 —
    재구축 중 실제로 그렇게 됐다(1h ma_cross_struct 가동률 39% → 2.5%).
    ATR·MA·스윙은 전부 **그 지표가 보는 봉** 위에서 정의되어야 한다.
    """
    k = ("stopaux", arch.tf, arch.stop.period, arch.stop.ma, arch.stop.lookback)
    if k in cache:
        return cache[k]
    htf, ref = _htf_ref(ltf, arch.tf, cache)
    hh = htf["high"].to_numpy(np.float64)
    ll = htf["low"].to_numpy(np.float64)
    cc = htf["close"].to_numpy(np.float64)
    ph, pl = _pivot_levels(hh, ll, max(arch.stop.lookback, 1))
    # 확정 스윙과 **그 이후 레그 극단 중 깊은 쪽**.
    # 하락 중 RSI 롱 진입이면 확정 스윙이 진입가보다 위에 있어 즉사한다 —
    # 레그 극단을 섞어 구조 파괴 지점을 진입가 아래로 내린다.
    # 레그는 반드시 **확정된 HTF 봉까지만** 본다. 5m 축에서 진행형 극단을 쓰면
    # 진입봉의 저가가 곧 손절가가 되어 전원 1봉 만에 털린다(재구축 중 실측 96%).
    pl = np.fmin(pl, segment_cummin(_neutral(ll, for_max=False),
                                    np.flatnonzero(np.diff(pl, prepend=np.nan) != 0)))
    ph = np.fmax(ph, segment_cummax(_neutral(hh, for_max=True),
                                    np.flatnonzero(np.diff(ph, prepend=np.nan) != 0)))
    out = dict(atr_full=_map5(_atr(hh, ll, cc, arch.stop.period), ref),
               ma_full=_map5(_sma(cc, arch.stop.ma), ref),
               piv=(_map5(ph, ref), _map5(pl, ref)))
    cache[k] = out
    return out


def run_trader(ltf: pd.DataFrame, arch: Archetype, cache: dict | None = None,
               cfg: PopConfig = PopConfig()) -> TraderState:
    """아키타입 하나를 5m 축에서 돌린다."""
    cache = {} if cache is None else cache
    n = len(ltf)
    o = ltf["open"].to_numpy(np.float64)
    h = ltf["high"].to_numpy(np.float64)
    l = ltf["low"].to_numpy(np.float64)
    c = ltf["close"].to_numpy(np.float64)

    s5, sl_lo, sl_hi = _signal_on_5m(ltf, arch, cache)
    if arch.combine:
        votes = np.abs(s5).astype(np.int64) * 0
        acc = s5.astype(np.int64)
        for sub in arch.combine:
            t5, _, _ = _signal_on_5m(ltf, replace(sub, stop=arch.stop, exit=arch.exit), cache)
            acc = acc + t5.astype(np.int64)
        need = max(int(arch.vote), 1)
        s5 = np.where(acc >= need, 1, np.where(acc <= -need, -1, 0)).astype(np.int8)
        del votes

    aux_full = _stop_aux(ltf, arch, cache)
    tb5 = arch.exit.time_bars * _tf_ratio(arch.tf)
    cond = _exit_condition(ltf, arch, cache)

    if arch.exit.kind == "none":
        return _run_hold_through(ltf, arch, s5, sl_lo, sl_hi, aux_full, cfg)

    starts, ends, side, why = _raw_episodes(s5, cond, tb5, n)
    if len(starts) == 0:
        z = np.empty(0, np.int64)
        return TraderState(z, z, np.empty(0, np.int8), np.empty(0), np.empty(0), z,
                           np.zeros(n, np.int8), np.full(n, np.nan))

    pos, off = _ranges(starts, ends)
    # 압축 축(구간을 이어 붙인 것)과 전체 길이 배열을 이름으로 구분한다
    aux = dict(close5=c[pos], high5=h[pos], low5=l[pos],
               atr5=aux_full["atr_full"][pos], ma5=aux_full["ma_full"][pos],
               atr_full=aux_full["atr_full"],
               piv_hi5=aux_full["piv"][0], piv_lo5=aux_full["piv"][1])
    stop_c, sd_c = _stop_series(arch, starts, ends, side, o, sl_lo, sl_hi, aux, n)

    lo_c, hi_c = l[pos], h[pos]
    hit = np.where(sd_c > 0, lo_c <= stop_c, hi_c >= stop_c)
    if not cfg.stop_on_entry_bar:
        first_of_seg = np.zeros(len(pos), bool)
        first_of_seg[off] = True
        hit &= ~first_of_seg
    ends_c = off + (ends - starts)
    hit_at = first_hit_per_segment(hit, off, ends_c)      # 압축 축 위치
    got = hit_at >= 0
    new_end = np.where(got, starts + (hit_at - off) + 1, ends)
    why = np.where(got, 0, why)                            # 0 = 손절

    entry = o[starts]
    stop0 = np.full(len(starts), np.nan)
    ok = off < len(stop_c)
    stop0[ok] = stop_c[off[ok]]

    side5 = np.zeros(n, np.int8)
    stop_px5 = np.full(n, np.nan)
    pos2, off2 = _ranges(starts, new_end)
    if len(pos2):
        side5[pos2] = np.repeat(side, new_end - starts)
        keep = (np.repeat(off, new_end - starts)
                + (np.arange(len(pos2), dtype=np.int64)
                   - np.repeat(off2, new_end - starts)))
        stop_px5[pos2] = stop_c[keep]
    return TraderState(starts, new_end, side, entry, stop0, why, side5, stop_px5)


def _run_hold_through(ltf, arch, s5, sl_lo, sl_hi, aux_full, cfg) -> TraderState:
    """``exit="none"`` — 신호가 뒤집혀도 손절 전까지 버틴다.

    에피소드 경계가 손절 결과에 의존하므로 벡터화가 안 된다. 순차 루프.
    """
    n = len(ltf)
    o = ltf["open"].to_numpy(np.float64)
    h = ltf["high"].to_numpy(np.float64)
    l = ltf["low"].to_numpy(np.float64)
    c = ltf["close"].to_numpy(np.float64)
    atr = aux_full["atr_full"]
    ma = aux_full["ma_full"]
    piv_hi, piv_lo = aux_full["piv"]
    k = arch.stop

    starts, ends, sides, entries, stop0s, whys = [], [], [], [], [], []
    side5 = np.zeros(n, np.int8)
    stop_px5 = np.full(n, np.nan)
    t = 0
    prev = np.int8(0)
    while t < n:
        if s5[t] == 0 or s5[t] == prev:
            prev = s5[t]
            t += 1
            continue
        sd = int(s5[t])
        prev = s5[t]
        s = t
        ent = o[s]
        stop = -np.inf if sd > 0 else np.inf
        hi_r, lo_r = -np.inf, np.inf
        end = min(n, s + cfg.max_hold)
        why = 4
        for u in range(s, end):
            hi_r = max(hi_r, h[u])
            lo_r = min(lo_r, l[u])
            if k.kind == "none":
                cur = -np.inf if sd > 0 else np.inf
            elif k.kind == "atr_trail":
                raw = c[u] - k.mult * atr[u] if sd > 0 else c[u] + k.mult * atr[u]
                if np.isfinite(raw):
                    stop = max(stop, raw) if sd > 0 else min(stop, raw)
                cur = stop
            elif k.kind == "chandelier":
                cur = (hi_r - k.mult * atr[u]) if sd > 0 else (lo_r + k.mult * atr[u])
            elif k.kind == "ma":
                cur = ma[u]
            elif k.kind == "atr":
                cur = ent - k.mult * atr[s] if sd > 0 else ent + k.mult * atr[s]
            elif k.kind == "pct":
                cur = ent * (1 - k.pct) if sd > 0 else ent * (1 + k.pct)
            elif k.kind == "pivot":
                cur = piv_lo[s] if sd > 0 else piv_hi[s]
            else:  # struct
                cur = sl_lo[s] if sd > 0 else sl_hi[s]
            if u == s:
                stop0 = cur
            side5[u] = sd
            stop_px5[u] = cur
            if np.isfinite(cur) and (u > s or cfg.stop_on_entry_bar):
                if (sd > 0 and l[u] <= cur) or (sd < 0 and h[u] >= cur):
                    end = u + 1
                    why = 0
                    break
        starts.append(s); ends.append(end); sides.append(sd)
        entries.append(ent); stop0s.append(stop0); whys.append(why)
        t = end
        prev = s5[end - 1] if end - 1 < n else prev
    z = np.asarray
    return TraderState(z(starts, np.int64), z(ends, np.int64), z(sides, np.int8),
                       z(entries, np.float64), z(stop0s, np.float64), z(whys, np.int64),
                       side5, stop_px5)


# ---------------------------------------------------------------- 집단 상태


@dataclass(slots=True)
class PainResult:
    """봉마다의 집단 상태. 개수가 아니라 **크기**로 센다."""

    n_long: np.ndarray
    n_short: np.ndarray
    #: Σ |미실현손실| — R 단위 (각자의 손절폭으로 정규화). 고통의 총량.
    pain_R_long: np.ndarray
    pain_R_short: np.ndarray
    #: Σ min(|미실현손실|, 3R). 분해 결과 실제로는 0.75R 에서 이미 포화한다.
    pain_R3_long: np.ndarray
    pain_R3_short: np.ndarray
    #: Σ |미실현손실| — ATR 단위 (절대 크기)
    pain_atr_long: np.ndarray
    pain_atr_short: np.ndarray
    #: Σ 미실현이익 (R)
    gain_R_long: np.ndarray
    gain_R_short: np.ndarray
    #: Σ 연속 수중 봉 수 — 시간이 쌓일수록 투항 압력
    uw_long: np.ndarray
    uw_short: np.ndarray
    #: Σ 보유 봉 수
    age_sum: np.ndarray

    @property
    def n_live(self) -> np.ndarray:
        return self.n_long + self.n_short

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return x / np.maximum(self.n_live, 1)

    @property
    def pain_imb_R(self) -> np.ndarray:
        """(숏 고통 − 롱 고통) / 생존자. 양수 = 숏이 더 아프다 = 스퀴즈 기대 방향."""
        return self._norm(self.pain_R_short - self.pain_R_long)

    @property
    def pain_imb_R3(self) -> np.ndarray:
        return self._norm(self.pain_R3_short - self.pain_R3_long)

    @property
    def pain_imb_atr(self) -> np.ndarray:
        return self._norm(self.pain_atr_short - self.pain_atr_long)

    @property
    def uw_imb(self) -> np.ndarray:
        return self._norm(self.uw_short - self.uw_long)


def population_state(ltf: pd.DataFrame, archs: list[Archetype], atr: np.ndarray,
                     cfg: PopConfig = PopConfig()) -> tuple[PainResult, np.ndarray]:
    """집단 상태를 봉 축으로 집계. 두 번째 반환값은 **개수 기반** 불균형(구버전).

    개수는 100명이 0.1% 물린 것과 10명이 5% 물린 것을 구분하지 못한다.
    ``pain_*`` 은 그 차이를 본다.
    """
    n = len(ltf)
    close = ltf["close"].to_numpy(dtype=np.float64)
    a = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    ar = np.arange(n)

    z = lambda: np.zeros(n, dtype=np.float64)
    R = PainResult(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
    cnt_l, cnt_s = z(), z()
    cache: dict = {}

    for arch in archs:
        st = run_trader(ltf, arch, cache, cfg)
        s, e = st.ep_start, st.ep_end
        if len(s) == 0:
            continue
        risk = np.abs(st.ep_entry - st.ep_stop0)
        bad = ~np.isfinite(risk) | (risk <= 0)
        risk = np.where(bad, 2.0 * a[s], risk)
        risk5 = _expand(risk, s, e, n)
        ent5 = _expand(st.ep_entry, s, e, n)   # float32 캐시 대신 원본 정밀도
        live = st.side5 != 0
        sd = st.side5.astype(np.float64)

        pnl = (close - ent5) * sd
        pnl_R = np.nan_to_num(np.where(live, pnl / risk5, 0.0))
        pnl_a = np.nan_to_num(np.where(live, pnl / a, 0.0))

        lg, sh = live & (st.side5 > 0), live & (st.side5 < 0)
        loss = np.maximum(0.0, -pnl_R)
        gain = np.maximum(0.0, pnl_R)
        lossa = np.maximum(0.0, -pnl_a)
        loss3 = np.minimum(loss, PAIN_CAP_R)
        R.pain_R_long += np.where(lg, loss, 0.0)
        R.pain_R_short += np.where(sh, loss, 0.0)
        R.pain_R3_long += np.where(lg, loss3, 0.0)
        R.pain_R3_short += np.where(sh, loss3, 0.0)
        R.gain_R_long += np.where(lg, gain, 0.0)
        R.gain_R_short += np.where(sh, gain, 0.0)
        R.pain_atr_long += np.where(lg, lossa, 0.0)
        R.pain_atr_short += np.where(sh, lossa, 0.0)
        R.n_long += lg
        R.n_short += sh
        cnt_l += lg & (pnl_R < 0)
        cnt_s += sh & (pnl_R < 0)

        # 연속 수중 봉 수: 수중이 아니면 리셋
        uw = live & (pnl_R < 0)
        last_ok = np.maximum.accumulate(np.where(~uw, ar, -1))
        dur = np.where(uw, ar - last_ok, 0).astype(np.float64)
        R.uw_long += np.where(lg, dur, 0.0)
        R.uw_short += np.where(sh, dur, 0.0)

        # 보유 봉 수
        st_mask = np.zeros(n, dtype=bool)
        st_mask[s] = True
        last_st = np.maximum.accumulate(np.where(st_mask, ar, -1))
        age = np.where(live & (last_st >= 0), ar - last_st + 1, 0).astype(np.float64)
        R.age_sum += age

    count_imb = (cnt_s - cnt_l) / np.maximum(R.n_live, 1)
    return R, count_imb


@dataclass(slots=True)
class Position:
    """문서에 적힌 트레이더 객체. 진단·시각화용 — 집계에는 쓰지 않는다."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int
    entry_reason: str
    current_pnl: float
    MAE: float
    MFE: float
    underwater_duration: int
    distance_from_entry: float
    age: int
    current_state: str
    exit_reason: str


def positions_frame(ltf: pd.DataFrame, st: TraderState, arch: Archetype,
                    at: int | None = None) -> pd.DataFrame:
    """특정 시점 ``at`` 에 살아 있던 포지션들의 스냅샷."""
    n = len(ltf)
    at = n - 1 if at is None else at
    c = ltf["close"].to_numpy(np.float64)
    h = ltf["high"].to_numpy(np.float64)
    l = ltf["low"].to_numpy(np.float64)
    rows = []
    live = (st.ep_start <= at) & (st.ep_end > at)
    for k in np.flatnonzero(live):
        s, e, sd = int(st.ep_start[k]), int(st.ep_end[k]), int(st.ep_side[k])
        ent = float(st.ep_entry[k])
        seg = slice(s, at + 1)
        pnl = (c[at] - ent) * sd
        mae = float(np.min((l[seg] - ent) * sd)) if sd > 0 else float(np.min((ent - h[seg]) * 1.0))
        mfe = float(np.max((h[seg] - ent) * sd)) if sd > 0 else float(np.max((ent - l[seg]) * 1.0))
        uw = int(np.sum((c[seg] - ent) * sd < 0))
        rows.append(dict(entry_time=ltf.index[s], entry_price=ent, direction=sd,
                         entry_reason=arch.name, current_pnl=float(pnl),
                         MAE=mae, MFE=mfe, underwater_duration=uw,
                         distance_from_entry=float(c[at] - ent), age=at - s + 1,
                         current_state="수중" if pnl < 0 else "수면위",
                         exit_reason=EXIT_REASONS[int(st.ep_exit[k])] if e <= at else "미청산"))
    return pd.DataFrame(rows)
