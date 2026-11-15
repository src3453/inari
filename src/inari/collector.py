"""Background data collection for the monitor.

Polls llama-server's HTTP endpoints (/health, /props, /slots, /metrics),
tails its log file and samples host hardware, producing immutable-ish
``Snapshot`` objects for the TUI to render.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from inari.hw import HwSampler, HwStat
from inari.logparse import LogInfo, LogParser, Request

HIST = 120

STATES = ("Loading", "Idle", "Reading", "Generating", "Queued", "Error")


@dataclass
class Snapshot:
    ts: float = 0.0
    url: str = ""
    name: str = ""
    model: str = ""
    state: str = "Loading"
    detail: str = "Starting"
    progress: float = 0.0
    progress_text: str = ""
    n_ctx: int = 0
    ctx_used: int = 0
    speed: float = 0.0
    speed_hist: list[float] = field(default_factory=list)
    pp_speed: float = 0.0
    pp_hist: list[float] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    props: dict = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    info: LogInfo = field(default_factory=LogInfo)
    hw: HwStat = field(default_factory=HwStat)
    cpu_hist: list[float] = field(default_factory=list)
    disk_hist: list[float] = field(default_factory=list)
    gpu_hist: list[float] = field(default_factory=list)
    pid: int | None = None
    alive: bool = True
    exit_code: int | None = None
    started: float | None = None
    cmd: list[str] = field(default_factory=list)
    health: str = ""
    has_logs: bool = False


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].split("{", 1)[0].replace("llamacpp:", "")
        try:
            out[key] = float(parts[1])
        except ValueError:
            pass
    return out


def _next_token(slot: dict) -> dict:
    nt = slot.get("next_token") or {}
    if isinstance(nt, list):
        nt = nt[0] if nt else {}
    return nt


class Collector:
    interval = 0.5

    def __init__(self, url: str, *, log_path: str | None = None, pid: int | None = None,
                 name: str = "", cmd: list[str] | None = None, started: float | None = None,
                 popen: Any = None) -> None:
        self.url = url.rstrip("/")
        self.log_path = Path(log_path) if log_path else None
        self.pid = pid
        self.popen = popen
        self.parser = LogParser()
        self.parser.t0 = started  # maps "M.SS.mmm.uuu" log timestamps to wall clock
        self.hw = HwSampler(pid)
        self.client = httpx.Client(timeout=1.5)
        self.snap = Snapshot(url=self.url, name=name, pid=pid, cmd=cmd or [], started=started,
                             has_logs=self.log_path is not None)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._log_pos = 0
        self._log_buf = b""
        self._prev_decoded: dict[int, tuple[int, int]] = {}
        self._speed_ewma = 0.0
        self._speed_hist: deque[float] = deque(maxlen=HIST)
        self._pp_hist: deque[float] = deque(maxlen=HIST)
        self._cpu_hist: deque[float] = deque(maxlen=HIST)
        self._disk_hist: deque[float] = deque(maxlen=HIST)
        self._gpu_hist: deque[float] = deque(maxlen=HIST)
        self._last_poll = time.monotonic()
        self._props_at = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---------- lifecycle ----------
    def start(self) -> "Collector":
        self._tail(initial=True)
        self.parser.live = True
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def poke(self) -> None:
        self._wake.set()

    def snapshot(self) -> Snapshot:
        with self._lock:
            return copy.copy(self.snap)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # never let the poller die
                self.parser.feed(f"[inari] collector error: {e!r}")
            self._wake.wait(self.interval)
            self._wake.clear()

    # ---------- one poll ----------
    def tick(self) -> None:
        now = time.monotonic()
        dt = max(now - self._last_poll, 1e-3)
        self._last_poll = now
        self._tail()
        s = Snapshot(**{k: getattr(self.snap, k) for k in ("url", "name", "pid", "cmd", "started", "has_logs",
                                                         "props", "model", "n_ctx")})
        s.ts = time.time()
        self._process_state(s)
        self._poll_server(s, dt)
        self._hardware(s)
        s.requests = list(self.parser.requests)
        s.logs = list(self.parser.lines)
        s.info = copy.deepcopy(self.parser.info)
        if not s.model:
            s.model = s.info.model_name
        if not s.n_ctx:
            s.n_ctx = s.info.n_ctx
        s.speed_hist = list(self._speed_hist)
        s.pp_hist = list(self._pp_hist)
        s.cpu_hist = list(self._cpu_hist)
        s.disk_hist = list(self._disk_hist)
        s.gpu_hist = list(self._gpu_hist)
        with self._lock:
            self.snap = s

    def _process_state(self, s: Snapshot) -> None:
        if self.popen is not None:
            code = self.popen.poll()
            if code is not None:
                s.alive, s.exit_code = False, code
        elif self.pid:
            import psutil

            s.alive = psutil.pid_exists(self.pid)

    def _get(self, path: str) -> httpx.Response | None:
        try:
            return self.client.get(self.url + path)
        except httpx.HTTPError:
            return None

    def _poll_server(self, s: Snapshot, dt: float) -> None:
        r = self._get("/health")
        if r is None:
            s.health = "unreachable"
            if not s.alive:
                s.state, s.detail = "Error", f"Server exited (code {s.exit_code})" if s.exit_code is not None else "Server exited"
            elif self.pid and not self.parser.info.listening:
                s.state, s.detail = "Loading", self._loading_detail()
            else:
                s.state, s.detail = "Error", "Unreachable"
            self._push_speed(0.0)
            return
        if r.status_code == 503:
            s.health = "loading"
            s.state, s.detail = "Loading", self._loading_detail()
            self._push_speed(0.0)
            return
        s.health = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"

        if not s.props or time.monotonic() - self._props_at > 10:
            pr = self._get("/props")
            if pr is not None and pr.status_code == 200:
                s.props = pr.json()
                self._props_at = time.monotonic()
        dgs = s.props.get("default_generation_settings", {}) if s.props else {}
        s.n_ctx = int(dgs.get("n_ctx") or s.props.get("n_ctx") or s.n_ctx or 0)
        mp = s.props.get("model_path") or s.props.get("model_alias") or ""
        if mp:
            s.model = Path(mp).stem if not s.model or s.model == self.parser.info.model_name else s.model

        mr = self._get("/metrics")
        if mr is not None and mr.status_code == 200:
            s.metrics = parse_metrics(mr.text)

        sr = self._get("/slots")
        s.slots = sr.json() if sr is not None and sr.status_code == 200 and isinstance(sr.json(), list) else []

        self._derive(s, dt)

    def _loading_detail(self) -> str:
        info = self.parser.info
        if info.buffers:
            return "Loading model into memory"
        if info.devices:
            return "Initialising devices"
        return "Starting llama-server"

    def _push_speed(self, v: float) -> None:
        self._speed_ewma = v if v == 0 else (0.5 * self._speed_ewma + 0.5 * v if self._speed_ewma else v)
        self._speed_hist.append(self._speed_ewma)

    def _derive(self, s: Snapshot, dt: float) -> None:
        decoded_delta = 0
        busy = []
        for slot in s.slots:
            sid = int(slot.get("id", 0))
            nt = _next_token(slot)
            raw_task = slot.get("id_task", slot.get("task_id"))
            task = int(raw_task) if raw_task is not None else -1
            n_dec = int(nt.get("n_decoded", 0) or 0)
            prev = self._prev_decoded.get(sid)
            if slot.get("is_processing"):
                busy.append((slot, nt))
                if prev and prev[0] == task and n_dec >= prev[1]:
                    decoded_delta += n_dec - prev[1]
                else:  # new task on this slot
                    decoded_delta += n_dec
            self._prev_decoded[sid] = (task, n_dec)

        speed = decoded_delta / dt if busy else 0.0
        self._push_speed(speed)
        s.speed = self._speed_ewma

        cur = self.parser.current()
        done = [r for r in self.parser.requests if r.pp_tps > 0]
        s.pp_speed = done[-1].pp_tps if done else 0.0
        self._pp_hist.append(s.pp_speed if cur and cur.status == "Reading" else 0.0)

        deferred = s.metrics.get("requests_deferred", 0)
        # newer servers report prompt progress in /slots. n_prompt_tokens keeps growing
        # with generated tokens, so it is only trusted while the prompt is being read.
        slot_pp: dict[int, float] = {}
        for slot, nt in busy:
            npt = slot.get("n_prompt_tokens")
            if not npt or int(nt.get("n_decoded", 0) or 0) > 0:
                continue
            req = self.parser.request_for_task(slot.get("id_task"))
            processed = slot.get("n_prompt_tokens_processed")
            cached = int(slot.get("n_prompt_tokens_cache") or 0)
            if processed is not None:
                slot_pp[id(slot)] = min(int(processed) / max(int(npt) - cached, 1), 1.0)
            if req is not None:
                req.prompt = max(req.prompt, int(npt))
                req.reused = req.reused or cached

        if busy:
            # show the slot furthest along: generating beats reading
            slot, nt = max(busy, key=lambda b: int(b[1].get("n_decoded", 0) or 0))
            n_dec = int(nt.get("n_decoded", 0) or 0)
            n_remain = int(nt.get("n_remain", -1) or -1)
            cur = self.parser.request_for_task(slot.get("id_task")) or cur
            if n_dec == 0:
                s.state, s.detail = "Reading", "Reading the prompt"
                pp = slot_pp.get(id(slot), cur.pp_progress if cur else 0.0)
                s.progress = pp
                todo = (cur.prompt - cur.reused) if cur else 0
                s.progress_text = f"{int(pp * todo):,} / {todo:,} prompt tokens" if todo > 0 else "prompt processing"
            else:
                s.state, s.detail = "Generating", "Generating a reply"
                if cur is not None:
                    cur.status, cur.output = "Generating", n_dec  # live count until timings arrive
                budget = n_dec + n_remain if n_remain > 0 else max(int(slot.get("n_ctx", s.n_ctx) or 0) - (cur.prompt if cur else 0), n_dec)
                s.progress = n_dec / budget if budget else 0.0
                s.progress_text = f"{n_dec:,} / {budget:,} tokens"
            if len(busy) > 1:
                s.detail += f" ×{len(busy)}"
        elif deferred:
            s.state, s.detail = "Queued", f"{int(deferred)} queued"
        else:
            s.state, s.detail = "Idle", "Ready"
            s.progress, s.progress_text = 0.0, ""

        # context fill: prefer metrics, else last/current request estimate
        kv_tokens = s.metrics.get("kv_cache_tokens")
        if kv_tokens:
            s.ctx_used = int(kv_tokens)
        elif busy:
            used = 0
            for slot, nt in busy:
                req = self.parser.request_for_task(slot.get("id_task"))
                used += (req.prompt if req else 0) + int(nt.get("n_decoded", 0) or 0)
            s.ctx_used = used
        else:
            r = cur or (self.parser.requests[-1] if self.parser.requests else None)
            if r:
                live_dec = int(busy[0][1].get("n_decoded", 0) or 0) if busy else r.output
                s.ctx_used = r.prompt + live_dec
        if s.slots:
            per_slot = int(s.slots[0].get("n_ctx", 0) or 0)
            if per_slot:
                s.n_ctx = s.n_ctx or per_slot

    def _hardware(self, s: Snapshot) -> None:
        hw = self.hw.sample()
        s.hw = hw
        self._cpu_hist.append(hw.cpu)
        self._disk_hist.append(hw.disk_read)
        self._gpu_hist.append(hw.gpus[0].util or 0.0 if hw.gpus else 0.0)

    # ---------- log tailing ----------
    def _tail(self, initial: bool = False) -> None:
        if self.log_path is None or not self.log_path.exists():
            return
        try:
            with open(self.log_path, "rb") as f:
                size = f.seek(0, 2)
                if size < self._log_pos:  # truncated / rotated
                    self._log_pos, self._log_buf = 0, b""
                if initial and size > 4_000_000:
                    self._log_pos = size - 4_000_000
                f.seek(self._log_pos)
                data = f.read()
                self._log_pos = f.tell()
        except OSError:
            return
        if not data:
            return
        data = self._log_buf + data
        *lines, self._log_buf = data.split(b"\n")
        for raw in lines:
            self.parser.feed(raw.decode("utf-8", errors="replace"))
