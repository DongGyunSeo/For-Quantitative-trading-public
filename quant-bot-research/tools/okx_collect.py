#!/usr/bin/env python3
"""OKX 5분봉 수집기 — 한도를 지키면서 끝까지 간다.

로컬(Windows/macOS/Linux)에서 실행한다. 필요한 건 ``requests`` 하나뿐이다.

    pip install requests
    python okx_collect.py probe                 # 한도·보존 깊이 실측 (몇 분)
    python okx_collect.py universe --top 50     # 심볼 선정 -> universe.json
    python okx_collect.py collect               # 본 수집 (중단·재개 안전)
    python okx_collect.py status                # 진행 현황
    python okx_collect.py verify                # 무결성 + 압축 마무리

확인된 한도 (2026-09 기준, OKX 문서 + ccxt 가중치로 교차 확인)
--------------------------------------------------------------
    GET /api/v5/market/history-candles   20 req / 2s,  limit <= 300   <- 본 수집용
    GET /api/v5/market/candles           40 req / 2s,  limit <= 300   (최근 구간만)

    ※ OKX 문서는 history-candles 의 limit 을 "Maximum is 100" 이라고 적어 뒀는데
      **실측하면 300 까지 준다**. probe 가 이걸 잡았다. 그래서 수집기는 매번
      시작할 때 상한을 직접 재서 쓴다(`discover_limit`). 문서를 믿지 않는다.
    GET /api/v5/public/instruments       20 req / 2s
    GET /api/v5/market/tickers           20 req / 2s

공개 마켓 엔드포인트의 한도는 **IP 에 묶인다.** 심볼을 나눠 병렬로 보내도
심볼마다 예산이 따로 생기지 않는다 — 10 req/s 를 모두가 나눠 쓴다.

직렬로 돌면 왕복 지연(~155ms) 때문에 6.5 req/s 에 머물고, 병렬로 그 틈을
메워 9 req/s 까지 올릴 수는 있다. 하지만 20요청/2초 창 기준으로 여유가
**35%(13/20) → 10%(18/20)** 로 줄어 버스트 한 번이면 넘친다. 이득 1.4배가
위험보다 작으므로 **단일 스레드 + 전역 토큰버킷**으로 간다.

**가장 흔한 실수는 창을 두 개 띄우는 것이다.** `--only` 로 심볼을 나눠 두
프로세스를 돌리면 각자 9 req/s 버킷을 들고 있어 합계 18 req/s — IP 한도를
넘는다. 그래서 `collect` 는 **잠금 파일**로 두 번째 실행을 거부한다.

처리량 산술
-----------
    실측 왕복 지연이 ~155ms 라 초당 6~7요청이 한계다(토큰버킷 9 req/s 에 안 닿는다).
    limit=300 이면  300봉 x 6.5요청/초 ≈ 2,000봉/초.

    5분봉 1년 = 105,120봉 =  ~0.9분
    5.3년     = 552,960봉 =  ~4.6분  (재시도 여유 포함 6분)
    50심볼 x 5.3년 ≈ 3~4시간   (limit=100 이었다면 9~11시간)

중단해도 손해가 없다. 페이지마다 `.part.csv` 에 이어 쓰고, 재개하면 그 지점부터
과거로 계속 내려간다.

페이지네이션 규약 (OKX)
-----------------------
응답은 **최신순**이다. ``after=<ts>`` 는 "그 ts 보다 **이전** 레코드"를 준다.
그래서 과거로 내려가려면 매번 받은 것 중 **가장 오래된 ts** 를 다음 ``after`` 로 쓴다.
``confirm == "1"`` 인 **마감된 봉만** 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:                                     # pragma: no cover
    sys.exit("requests 가 필요하다:  pip install requests")

BASE = "https://www.okx.com"
BAR = "5m"
BAR_MS = 5 * 60 * 1000
COLS = ("time", "open", "high", "low", "close", "volume")

ROOT = Path(os.environ.get("OKX_DATA", "okx_data"))
RAW = ROOT / "raw"
UNIVERSE = ROOT / "universe.json"
MANIFEST = ROOT / "manifest.json"

_STOP = False


def _on_sigint(*_):
    global _STOP
    if _STOP:
        sys.exit("두 번 눌렀다 — 즉시 종료")
    _STOP = True
    print("\n[중단 요청] 현재 페이지까지 저장하고 멈춘다. 다시 실행하면 이어서 간다.",
          flush=True)


signal.signal(signal.SIGINT, _on_sigint)


# ---------------------------------------------------------------- 한도


class Bucket:
    """전역 토큰버킷 + 429 적응 감속.

    OKX 는 초과하면 바로 막지 않고 429 를 준다. 받으면 **속도를 절반으로** 떨어뜨리고
    천천히 회복시킨다 — 고정 sleep 보다 훨씬 안전하다.
    """

    def __init__(self, rps: float, burst: float = 5.0):
        self.base = rps
        self.rps = rps
        self.burst = burst
        self.tokens = burst
        self.t = time.monotonic()
        self.throttled = 0

    def take(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rps)
            self.t = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                # 감속했으면 천천히 회복
                if self.rps < self.base:
                    self.rps = min(self.base, self.rps * 1.02)
                return
            time.sleep((1.0 - self.tokens) / self.rps)

    def slow_down(self) -> None:
        self.throttled += 1
        self.rps = max(self.base * 0.2, self.rps * 0.5)


class Client:
    def __init__(self, rps: float, timeout: float = 20.0, retries: int = 8):
        self.b = Bucket(rps)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "qbot-collector/1.0",
                               "Accept": "application/json"})
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.errors = 0

    def get(self, path: str, **params):
        """성공 시 data 리스트. 복구 불가면 예외."""
        last = None
        for attempt in range(self.retries):
            if _STOP:
                raise KeyboardInterrupt
            self.b.take()
            try:
                r = self.s.get(BASE + path, params=params, timeout=self.timeout)
                self.calls += 1
                if r.status_code == 429:
                    self.b.slow_down()
                    time.sleep(1.0 + attempt + random.random())
                    continue
                if r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    time.sleep(min(60, 2 ** attempt) + random.random())
                    continue
                r.raise_for_status()
                j = r.json()
                code = str(j.get("code", ""))
                if code == "0":
                    return j.get("data", [])
                if code in ("50011", "50013"):          # 한도 / 시스템 바쁨
                    self.b.slow_down()
                    time.sleep(1.0 + attempt)
                    continue
                if code in ("51001", "51000"):          # 없는 종목 / 파라미터 오류
                    raise LookupError(f"{code}: {j.get('msg')}")
                last = f"code {code}: {j.get('msg')}"
                time.sleep(min(30, 2 ** attempt))
            except (requests.Timeout, requests.ConnectionError) as e:
                last = repr(e)
                time.sleep(min(60, 2 ** attempt) + random.random())
            except ValueError as e:                      # JSON 파싱
                last = repr(e)
                time.sleep(min(30, 2 ** attempt))
        self.errors += 1
        raise RuntimeError(f"{path} 실패 ({self.retries}회): {last}")


# ---------------------------------------------------------------- 유틸


def ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def iso(t_ms: int) -> str:
    return datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S+00:00")


def parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def load_json(p: Path, default):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def human(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600}시간 {sec % 3600 // 60}분" if sec >= 3600 else f"{sec // 60}분 {sec % 60}초"


# ---------------------------------------------------------------- 동시 실행 방지


LOCK_STALE_SEC = 300


class RunLock:
    """한 IP 에서 수집기는 하나만. 두 개를 띄우면 한도를 **나눠 쓰는 게 아니라 넘는다.**

    Windows 에서 PID 생존 확인은 표준 라이브러리로 깔끔하지 않으므로 **하트비트**로
    판정한다. 수집 중 페이지를 쓸 때마다 갱신하고, 5분 넘게 갱신이 없으면
    이전 실행이 비정상 종료한 것으로 보고 넘겨받는다.
    """

    def __init__(self, path: Path, force: bool = False, rps: float | None = None):
        self.path = path
        self.force = force
        self.rps = rps
        self.held = False
        self.started = None

    def _read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cur = self._read()
        if cur and not self.force:
            age = time.time() - float(cur.get("beat", 0))
            if age < LOCK_STALE_SEC:
                sys.exit(
                    f"\n다른 수집기가 이미 돌고 있다 (PID {cur.get('pid')}, "
                    f"{int(age)}초 전 갱신, {cur.get('rps')} req/s).\n"
                    f"같은 IP 에서 두 개를 돌리면 한도를 나눠 쓰는 게 아니라 **넘는다**.\n"
                    f"정말 다른 수집기가 없다면 {self.path} 를 지우거나 --force 로 실행하라.")
            print(f"[잠금] 이전 실행이 {int(age)}초간 멈춰 있었다 — 비정상 종료로 보고 넘겨받는다.")
        self.held = True
        self.started = iso(ms(datetime.utcnow()))
        self.beat()

    def beat(self, **extra) -> None:
        if not self.held:
            return
        save_json(self.path, dict(pid=os.getpid(), beat=time.time(),
                                  started=self.started, rps=self.rps, **extra))

    def release(self) -> None:
        if self.held:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.held = False


# ---------------------------------------------------------------- 실측 probe


def cmd_probe(a) -> None:
    """문서를 믿지 말고 직접 재본다 — limit 상한, 보존 깊이, 실제 감속 지점."""
    c = Client(a.rps)
    print("=" * 72)
    print("OKX 실측 probe")
    print("=" * 72)

    inst = c.get("/api/v5/public/instruments", instType="SWAP")
    usdt = [i for i in inst if i.get("settleCcy") == "USDT"
            and i.get("ctType") == "linear" and i.get("state") == "live"]
    print(f"\n[1] 상장 중인 USDT 선형 영구선물: {len(usdt)}개 / 전체 SWAP {len(inst)}개")
    lt = sorted(int(i["listTime"]) for i in usdt if i.get("listTime"))
    for y in (2021, 2022, 2023, 2024):
        cut = ms(datetime(y, 7, 1))
        print(f"    {y}-07-01 이전 상장: {sum(1 for x in lt if x <= cut):>3}개")

    print(f"\n[2] limit 상한 실측  (instId=BTC-USDT-SWAP, bar={BAR})")
    print("    ※ OKX 문서는 history-candles 를 '최대 100' 이라 적어 뒀다. 직접 재본다.")
    for path, want in (("/api/v5/market/history-candles", (100, 300, 500, 1000)),
                       ("/api/v5/market/candles", (300, 301, 1000))):
        got = []
        for L in want:
            try:
                d = c.get(path, instId="BTC-USDT-SWAP", bar=BAR, limit=str(L))
                got.append(f"limit={L} -> {len(d)}개")
            except Exception as e:
                got.append(f"limit={L} -> 거부({type(e).__name__})")
        print(f"    {path:<38} " + " · ".join(got))

    print(f"\n[3] 과거로 얼마나 내려가지는가 — {a.probe_pages}페이지까지만 본다")
    print("    (여기서 '벽' 이라고 나오지 않으면 보존 한계에 닿은 게 아니라 페이지 수를 다 쓴 것이다)")
    for path, lim in (("/api/v5/market/candles", 300),
                      ("/api/v5/market/history-candles", 300)):
        after = None
        oldest = None
        pages = 0
        t0 = time.time()
        while pages < a.probe_pages:
            p = dict(instId="BTC-USDT-SWAP", bar=BAR, limit=str(lim))
            if after:
                p["after"] = str(after)
            d = c.get(path, **p)
            if not d:
                break
            pages += 1
            oldest = int(d[-1][0])
            after = oldest
        span = (time.time() - t0)
        wall = "벽(빈 응답)" if pages < a.probe_pages else "페이지 소진"
        print(f"    {path:<38} {pages:>3}페이지 -> {iso(oldest) if oldest else '—'}"
              f"  ({span:.1f}s, {pages*lim/max(span,1e-9):,.0f}봉/초, {pages/max(span,1e-9):.1f}req/s"
              f", {wall})")

    print(f"\n[4] 한도 반응 — {a.rps:.1f} req/s 로 {c.calls}회 호출, 감속 {c.b.throttled}회")
    print(f"    현재 유효 속도 {c.b.rps:.2f} req/s")
    print(f"\n[5] 권장 설정")
    try:
        lim = discover_limit(c)
        print(f"    요청당 봉 수  {lim}   (문서값 100 이 아니라 실측값을 쓴다)")
    except Exception as e:
        lim = 100
        print(f"    limit 실측 실패 ({e}) -> 100 으로 간다")
    print(f"    --rps 9.0 유지. 왕복 지연 때문에 실제로는 6~7 req/s 로 돌고,")
    print(f"    그래도 {lim}봉/요청이면 초당 {lim*6.5:,.0f}봉이다.")


# ---------------------------------------------------------------- 유니버스


def cmd_universe(a) -> None:
    c = Client(a.rps)
    inst = c.get("/api/v5/public/instruments", instType="SWAP")
    tick = {t["instId"]: t for t in c.get("/api/v5/market/tickers", instType="SWAP")}

    rows = []
    for i in inst:
        if (i.get("settleCcy") != "USDT" or i.get("ctType") != "linear"
                or i.get("state") != "live"):
            continue
        t = tick.get(i["instId"])
        if not t:
            continue
        try:
            last = float(t["last"])
            notional = float(t["vol24h"]) * float(i["ctVal"]) * last
        except (KeyError, ValueError, TypeError):
            continue
        lt = int(i.get("listTime") or 0)
        rows.append(dict(instId=i["instId"], sym=i["instId"].split("-")[0],
                         listTime=lt, list_iso=iso(lt) if lt else "",
                         ctVal=float(i["ctVal"]), last=last,
                         notional24h=notional))
    rows.sort(key=lambda r: -r["notional24h"])

    cut = ms(parse_day(a.min_list_date))
    old = [r for r in rows if r["listTime"] and r["listTime"] <= cut]
    new = [r for r in rows if not (r["listTime"] and r["listTime"] <= cut)]
    pick = old[:a.top]
    short = a.top - len(pick)
    if short > 0:
        # **부족분을 유동성 순으로 채우지 않는다(기본).**
        # OKX 는 토큰화 주식 영구선물(AAPL·NVDA·AMZN·ASML…)을 상장해 뒀고 이것들이
        # 거래량 상위에 들어온다. 전부 2025~2026 상장이라 이력이 몇 달뿐인데
        # 조용히 섞이면 "50심볼 5.3년" 이라는 전제가 깨진다.
        print(f"\n⚠ {a.min_list_date} 이전 상장이 {len(old)}개뿐이다 (요청 {a.top}개).")
        if a.allow_short_history:
            print(f"  --allow-short-history 가 켜져 있어 부족한 {short}개를 "
                  f"유동성 순으로 채운다. 이력이 짧은 종목이 섞인다:")
            for r in new[:short]:
                print(f"    + {r['sym']:<10} 상장 {r['list_iso'][:10]}")
            pick += new[:short]
        else:
            print(f"  {len(pick)}개만 선정한다. 굳이 채우려면 --allow-short-history 를 켜라"
                  f" (이력 짧은 종목이 섞인다) 또는 --min-list-date 를 뒤로 미뤄라.")
            nxt = new[:5]
            if nxt:
                print(f"  참고로 다음 순위는: "
                      + ", ".join(f"{r['sym']}({r['list_iso'][:7]})" for r in nxt))

    for rank, r in enumerate(pick, 1):
        r["rank"] = rank
    save_json(UNIVERSE, dict(created=iso(ms(datetime.utcnow())),
                             bar=BAR, target_start=a.start,
                             min_list_date=a.min_list_date,
                             n_candidates=len(rows), picked=pick))
    print(f"\n후보 {len(rows)}개 중 {len(pick)}개 선정 -> {UNIVERSE}")
    print(f"  {'#':>3} {'심볼':<8}{'instId':<20}{'상장':<12}{'24h 명목($)':>16}{'이력':>8}")
    now = datetime.now(timezone.utc)
    for r in pick:
        yrs = (now - datetime.fromtimestamp(r["listTime"] / 1000, timezone.utc)).days / 365.25
        print(f"  {r['rank']:>3} {r['sym']:<8}{r['instId']:<20}"
              f"{r['list_iso'][:10]:<12}{r['notional24h']:>16,.0f}{yrs:>7.1f}년")
    tot = sum(min((now - datetime.fromtimestamp(r['listTime']/1000, timezone.utc)).days,
                  (now - parse_day(a.start)).days) for r in pick) * 288
    print(f"\n예상 총 봉 수 {tot:,.0f} → 1,000봉/초 기준 약 {human(tot/1000*1.2)}")


# ---------------------------------------------------------------- 본 수집


def discover_limit(c: Client, want: int = 1000) -> int:
    """history-candles 가 실제로 몇 개까지 주는지 **한 번 재서** 쓴다.

    문서는 100 이라고 적혀 있는데 실측은 300 이다. 나중에 또 바뀔 수 있으므로
    상수로 박지 않는다. 한 요청이면 끝난다.
    """
    d = c.get("/api/v5/market/history-candles", instId="BTC-USDT-SWAP",
              bar=BAR, limit=str(want))
    n = len(d)
    return max(100, n)


def part_path(sym: str) -> Path:
    return RAW / f"{sym}_5m_futures.part.csv"


def final_path(sym: str) -> Path:
    return RAW / f"{sym}_5m_futures.csv.gz"


def part_oldest(p: Path) -> tuple[int | None, int]:
    """이어받기용 — 이미 받은 것 중 가장 오래된 ts 와 줄 수."""
    if not p.exists():
        return None, 0
    oldest, n = None, 0
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "ts":
                continue
            n += 1
            t = int(row[0])
            if oldest is None or t < oldest:
                oldest = t
    return oldest, n


def fetch_symbol(c: Client, inst: str, sym: str, start_ms: int,
                 limit: int = 100, flush_every: int = 20, lock: "RunLock | None" = None
                 ) -> dict:
    """과거로 내려가며 `.part.csv` 에 이어 쓴다. 반환값은 이 심볼의 진행 상태."""
    RAW.mkdir(parents=True, exist_ok=True)
    pp = part_path(sym)
    after, have = part_oldest(pp)
    if after is not None and after <= start_ms:
        return dict(done=True, oldest=after, rows=have, reason="이미 목표 도달")

    buf: list[list] = []
    pages = 0
    t0 = time.time()
    reason = "목표 도달"
    fresh = not pp.exists()

    def flush():
        if not buf:
            return
        with pp.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if fresh and f.tell() == 0:
                w.writerow(["ts", "open", "high", "low", "close", "volume"])
            w.writerows(buf)
        buf.clear()

    while True:
        if _STOP:
            reason = "사용자 중단"
            break
        p = dict(instId=inst, bar=BAR, limit=str(limit))
        if after is not None:
            p["after"] = str(after)
        try:
            d = c.get("/api/v5/market/history-candles", **p)
        except LookupError as e:
            reason = f"종목 오류 {e}"
            break
        if not d:
            reason = "보존 한계 도달"
            break
        got = 0
        for row in d:
            # [ts, o, h, l, c, vol(계약), volCcy(기준), volCcyQuote, confirm]
            if len(row) >= 9 and row[8] != "1":
                continue                       # 미마감 봉은 버린다
            t = int(row[0])
            if t < start_ms:
                continue
            buf.append([t, row[1], row[2], row[3], row[4], row[6]])
            got += 1
        oldest_page = int(d[-1][0])
        pages += 1
        have += got
        if oldest_page <= start_ms:
            after = oldest_page
            break
        if after is not None and oldest_page >= after:
            reason = "진전 없음(중복 페이지)"      # 무한루프 방지
            break
        after = oldest_page
        if pages % flush_every == 0:
            flush()
            if lock is not None:
                lock.beat(sym=sym)
            el = time.time() - t0
            print(f"    {sym:<8} {have:>7,}봉  {iso(after)[:16]}  "
                  f"{have/max(el,1e-9):>6,.0f}봉/초", flush=True)
    flush()
    done = reason in ("목표 도달", "보존 한계 도달")
    return dict(done=done, oldest=after, rows=have, reason=reason,
                seconds=round(time.time() - t0, 1))


def cmd_collect(a) -> None:
    uni = load_json(UNIVERSE, None)
    if not uni:
        sys.exit(f"{UNIVERSE} 가 없다. 먼저 `universe` 를 돌려라.")
    lock = RunLock(ROOT / "collect.lock", force=bool(a.force), rps=a.rps)
    lock.acquire()
    try:
        _collect(a, uni, lock)
    finally:
        lock.release()


def _collect(a, uni, lock) -> None:
    man = load_json(MANIFEST, {})
    c = Client(a.rps)
    start_ms = ms(parse_day(a.start or uni["target_start"]))
    picked = uni["picked"]
    if a.only:
        want = {x.strip().upper() for x in a.only.split(",")}
        picked = [r for r in picked if r["sym"] in want]

    todo = [r for r in picked if not man.get(r["sym"], {}).get("done")]
    print(f"수집 대상 {len(todo)} / {len(picked)}심볼 · 목표 시작 {a.start or uni['target_start']}")
    if not todo:
        print("할 일이 없다. `verify` 로 마무리하면 된다.")
        return
    lim = a.limit or discover_limit(c)          # 할 일이 있을 때만 잰다
    print(f"속도 {a.rps} req/s (한도 10 req/s 의 {a.rps/10:.0%}) · "
          f"요청당 {lim}봉 {'(실측)' if not a.limit else '(지정)'}\n")

    t_run = time.time()
    done_bars = 0
    for k, r in enumerate(todo, 1):
        if _STOP:
            break
        if a.max_hours and (time.time() - t_run) / 3600 >= a.max_hours:
            print(f"\n--max-hours {a.max_hours} 도달 — 여기서 멈춘다. 다시 실행하면 이어서 간다.")
            break
        sym, inst = r["sym"], r["instId"]
        print(f"[{k}/{len(todo)}] {sym}  ({inst})", flush=True)
        try:
            st = fetch_symbol(c, inst, sym, start_ms, limit=lim, lock=lock)
        except KeyboardInterrupt:
            break
        except Exception as e:
            st = dict(done=False, rows=0, reason=f"예외 {type(e).__name__}: {e}")
            print(f"    ! {st['reason']}", flush=True)
        man[sym] = {**man.get(sym, {}), **st, "instId": inst,
                    "updated": iso(ms(datetime.utcnow()))}
        save_json(MANIFEST, man)
        done_bars += st.get("rows", 0)
        el = time.time() - t_run
        rate = done_bars / max(el, 1e-9)
        left = len(todo) - k
        print(f"    -> {st.get('rows',0):,}봉 · {st['reason']} · {human(st.get('seconds',0))}"
              f"   [누적 {done_bars:,}봉 {rate:,.0f}봉/초 · 남은 {left}심볼 "
              f"≈ {human(left * st.get('seconds', 600))}]", flush=True)

    print(f"\n호출 {c.calls:,}회 · 감속 {c.b.throttled}회 · 실패 {c.errors}회")
    print(f"완료 {sum(1 for v in man.values() if v.get('done'))} / {len(picked)}심볼")
    print("남은 게 있으면 같은 명령을 다시 실행하면 된다. `verify` 로 마무리한다.")


# ---------------------------------------------------------------- 검수 · 마무리


def finalize(sym: str, keep_part: bool = False) -> dict:
    """part -> 정렬·중복제거·무결성 검사 -> .csv.gz"""
    pp = part_path(sym)
    if not pp.exists():
        return dict(sym=sym, ok=False, reason="part 없음")
    rows: dict[int, tuple] = {}
    bad = 0
    with pp.open("r", encoding="utf-8", newline="") as f:
        for r in csv.reader(f):
            if not r or r[0] == "ts":
                continue
            try:
                t = int(r[0])
                vals = tuple(float(x) for x in r[1:6])
            except ValueError:
                bad += 1
                continue
            if not (vals[1] >= vals[2] and vals[1] >= vals[0] and vals[1] >= vals[3]
                    and vals[2] <= vals[0] and vals[2] <= vals[3]):
                bad += 1                       # high<low 같은 말이 안 되는 봉
            rows[t] = vals
    if not rows:
        return dict(sym=sym, ok=False, reason="행 없음")
    ts = sorted(rows)
    gaps = sum((ts[i + 1] - ts[i]) // BAR_MS - 1 for i in range(len(ts) - 1))
    maxgap = max(((ts[i + 1] - ts[i]) // BAR_MS - 1 for i in range(len(ts) - 1)),
                 default=0)
    out = final_path(sym)
    with gzip.open(out, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for t in ts:
            v = rows[t]
            w.writerow([iso(t), *v])
    if not keep_part:
        pp.unlink()
    return dict(sym=sym, ok=True, rows=len(ts), start=iso(ts[0]), end=iso(ts[-1]),
                gaps=int(gaps), max_gap=int(maxgap), bad_ohlc=bad,
                mb=round(out.stat().st_size / 1e6, 1))


def cmd_verify(a) -> None:
    man = load_json(MANIFEST, {})
    syms = sorted({p.name.split("_")[0] for p in RAW.glob("*_5m_futures.part.csv")})
    if a.only:
        want = {x.strip().upper() for x in a.only.split(",")}
        syms = [s for s in syms if s in want]
    if not syms:
        print("마무리할 part 파일이 없다.")
    print(f"  {'심볼':<8}{'행':>10}{'시작':<12}{'끝':<12}{'공백':>7}{'최장':>6}{'이상봉':>7}{'MB':>6}")
    res = []
    for s in syms:
        r = finalize(s, keep_part=a.keep_part)
        res.append(r)
        if r["ok"]:
            print(f"  {s:<8}{r['rows']:>10,}{r['start'][:10]:<12}{r['end'][:10]:<12}"
                  f"{r['gaps']:>7,}{r['max_gap']:>6}{r['bad_ohlc']:>7}{r['mb']:>6.1f}")
            man[s] = {**man.get(s, {}), "final": r}
        else:
            print(f"  {s:<8}  {r['reason']}")
    save_json(MANIFEST, man)
    tot = sum(r.get("rows", 0) for r in res if r["ok"])
    mb = sum(r.get("mb", 0) for r in res if r["ok"])
    print(f"\n{len([r for r in res if r['ok']])}심볼 · {tot:,}봉 · {mb:.0f}MB (gzip)")
    print(f"파일: {RAW}/*_5m_futures.csv.gz")


def cmd_status(a) -> None:
    uni = load_json(UNIVERSE, None)
    man = load_json(MANIFEST, {})
    if not uni:
        sys.exit(f"{UNIVERSE} 가 없다.")
    picked = uni["picked"]
    done = [r for r in picked if man.get(r["sym"], {}).get("done")]
    part = {p.name.split("_")[0]: p.stat().st_size for p in RAW.glob("*.part.csv")}
    fin = {p.name.split("_")[0]: p.stat().st_size for p in RAW.glob("*.csv.gz")}
    print(f"유니버스 {len(picked)}심볼 · 목표 시작 {uni['target_start']}")
    print(f"수집 완료 {len(done)} · 진행 중 {len(part)} · 압축 완료 {len(fin)}")
    print(f"\n  {'심볼':<8}{'상태':<14}{'행':>10}{'가장 오래된':<20}{'사유':<18}")
    for r in picked:
        m = man.get(r["sym"], {})
        st = "완료" if m.get("done") else ("진행" if r["sym"] in part else "대기")
        if r["sym"] in fin:
            st = "압축완료"
        old = iso(m["oldest"])[:16] if m.get("oldest") else "—"
        print(f"  {r['sym']:<8}{st:<14}{m.get('rows',0):>10,}{old:<20}{m.get('reason','')[:18]:<18}")
    left = [r for r in picked if not man.get(r["sym"], {}).get("done")]
    if left:
        print(f"\n남은 {len(left)}심볼 — `collect` 를 다시 실행하면 이어서 간다.")


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="OKX 5분봉 수집기")
    ap.add_argument("--rps", type=float, default=9.0,
                    help="초당 요청 (history-candles 한도는 10/s, 기본 9.0)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="한도·보존 깊이 실측")
    p.add_argument("--probe-pages", type=int, default=40)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("universe", help="심볼 선정")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--start", default="2021-04-02", help="수집 목표 시작일")
    p.add_argument("--min-list-date", default="2022-07-01",
                   help="이 날짜 이전 상장만 선정 (기본: 4년 이상 이력)")
    p.add_argument("--allow-short-history", action="store_true",
                   help="--top 을 채우려고 이력 짧은 종목까지 끌어온다. "
                        "OKX 의 토큰화 주식 영구선물이 섞이므로 기본은 꺼짐")
    p.set_defaults(fn=cmd_universe)

    p = sub.add_parser("collect", help="본 수집 (중단·재개 안전)")
    p.add_argument("--start", default=None, help="universe.json 의 값을 덮어쓴다")
    p.add_argument("--only", default=None, help="쉼표로 구분한 심볼만")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="요청당 봉 수. 기본은 시작할 때 실측한다(문서값 100 은 틀렸다)")
    p.add_argument("--force", action="store_true",
                   help="잠금 파일을 무시한다. 다른 수집기가 **정말** 없을 때만")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("verify", help="정렬·중복제거·무결성 검사 후 gzip")
    p.add_argument("--only", default=None)
    p.add_argument("--keep-part", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("status", help="진행 현황")
    p.set_defaults(fn=cmd_status)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
