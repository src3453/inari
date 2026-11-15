"""Incremental parser for llama-server log output.

Extracts device / VRAM buffer info, model metadata and a per-request history
(prompt, reused cache tokens, output, speed, duration) from the text log.
Patterns are best-effort and tolerant of version differences.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

RE_DEVICE = re.compile(r"using device (\S+) \((.+?)\).*?-\s*(\d+) MiB free")
RE_BUFFER = re.compile(r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
RE_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
RE_PRINT_INFO = re.compile(r"print_info:\s+([a-zA-Z_][\w ]*?)\s*=\s*(.+)$")
RE_GENERAL_NAME = re.compile(r"general\.name\s+str\s+=\s+(.+)$")
RE_NCTX = re.compile(r"llama_context:\s+n_ctx\s+=\s+(\d+)")
RE_LISTEN = re.compile(r"server is listening on (\S+)")
RE_TASK = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(-?\d+)")
RE_NEW_PROMPT = re.compile(r"new prompt.*?(?:n_prompt_tokens|task\.n_tokens)\s*=\s*(\d+)")
RE_CACHE_RM = re.compile(r"(?:kv cache rm|memory_seq_rm)\s*\[(\d+),\s*end\)")
RE_PP_PROGRESS = re.compile(r"prompt processing progress.*?progress\s*=\s*([\d.]+)")
RE_PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_EVAL = re.compile(r"(?<!prompt )\beval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+) ms")
RE_ERROR = re.compile(r"\b(error|failed|fatal|exception|abort)\b", re.IGNORECASE)
RE_WARN = re.compile(r"\b(warn|warning)\b", re.IGNORECASE)
# newer builds prefix lines with "M.SS.mmm.uuu L " (time since start + level I/W/E/D)
RE_PREFIX = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ([IWED]) ")


def line_level(line: str) -> str:
    """'E', 'W', 'I' or '' for a raw log line."""
    m = RE_PREFIX.match(line)
    if m:
        return m.group(5)
    if RE_ERROR.search(line) and "error_" not in line:
        return "E"
    if RE_WARN.search(line):
        return "W"
    return ""


@dataclass
class Request:
    task: int
    slot: int
    started: float | None
    prompt: int = 0
    reused: int = 0
    output: int = 0
    pp_tps: float = 0.0
    tps: float = 0.0
    duration: float = 0.0
    status: str = "Reading"
    pp_progress: float = 0.0
    _reuse_seen: bool = field(default=False, repr=False)


@dataclass
class LogInfo:
    devices: dict[str, tuple[str, int]] = field(default_factory=dict)  # id -> (desc, free MiB)
    buffers: dict[str, dict[str, float]] = field(default_factory=dict)  # device -> kind -> MiB
    offload: tuple[int, int] | None = None
    model_info: dict[str, str] = field(default_factory=dict)
    model_name: str = ""
    n_ctx: int = 0
    listening: str = ""
    errors: int = 0
    warnings: int = 0

    def gpu_buffers(self) -> dict[str, dict[str, float]]:
        return {d: b for d, b in self.buffers.items() if not _is_host(d)}

    def vram_used_mib(self) -> float:
        return sum(sum(b.values()) for b in self.gpu_buffers().values())

    def host_buffer_mib(self) -> float:
        return sum(sum(b.values()) for d, b in self.buffers.items() if _is_host(d))


def _is_host(dev: str) -> bool:
    d = dev.upper()
    return d.startswith("CPU") or "HOST" in d


class LogParser:
    def __init__(self, max_lines: int = 3000, max_requests: int = 500):
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.requests: deque[Request] = deque(maxlen=max_requests)
        self.info = LogInfo()
        self._by_task: dict[int, Request] = {}
        self._timing: Request | None = None
        self.live = False  # False while replaying backlog -> no wall-clock timestamps
        self.t0: float | None = None  # wall-clock start of the server, maps relative log timestamps
        self._now: float | None = None

    def _stamp(self) -> float | None:
        if self._now is not None:
            return self._now
        return time.time() if self.live else None

    def feed(self, line: str) -> None:
        line = line.rstrip("\r\n")
        if not line.strip():
            return
        self.lines.append(line)
        info = self.info

        self._now = None
        m = RE_PREFIX.match(line)
        if m and self.t0 is not None:
            mins, sec, ms, us = (int(x) for x in m.group(1, 2, 3, 4))
            self._now = self.t0 + mins * 60 + sec + ms / 1e3 + us / 1e6
        lvl = line_level(line)
        if lvl == "E":
            info.errors += 1
        elif lvl == "W":
            info.warnings += 1

        if m := RE_DEVICE.search(line):
            info.devices[m.group(1)] = (m.group(2), int(m.group(3)))
            return
        if m := RE_BUFFER.search(line):
            kind = {"KV": "kv", "RS": "kv"}.get(m.group(2), m.group(2))
            dev = info.buffers.setdefault(m.group(1), {})
            dev[kind] = dev.get(kind, 0.0) + float(m.group(3))
            return
        if m := RE_OFFLOAD.search(line):
            info.offload = (int(m.group(1)), int(m.group(2)))
            return
        if m := RE_GENERAL_NAME.search(line):
            info.model_name = m.group(1).strip()
            return
        if m := RE_PRINT_INFO.search(line):
            info.model_info.setdefault(m.group(1).strip(), m.group(2).strip())
            return
        if m := RE_NCTX.search(line):
            info.n_ctx = int(m.group(1))
            return
        if m := RE_LISTEN.search(line):
            info.listening = m.group(1)
            return

        self._requests(line)

    def _requests(self, line: str) -> None:
        tm = RE_TASK.search(line)
        if tm and int(tm.group(2)) >= 0:
            slot, task = int(tm.group(1)), int(tm.group(2))
            req = self._by_task.get(task)
            new_prompt = RE_NEW_PROMPT.search(line)
            if req is None and (new_prompt or "processing task" in line):
                req = Request(task=task, slot=slot, started=self._stamp())
                self._by_task[task] = req
                self.requests.append(req)
                if len(self._by_task) > 200:
                    for k in list(self._by_task)[:100]:
                        self._by_task.pop(k, None)
            if req is None:
                return
            if new_prompt:
                req.prompt = int(new_prompt.group(1))
                return
            if self._timing_line(req, line):
                return
            if m := RE_CACHE_RM.search(line):
                # first one = cache reuse; later ones (one per prompt chunk) = progress
                n = int(m.group(1))
                if not req._reuse_seen:
                    req.reused, req._reuse_seen = n, True
                elif req.prompt > req.reused:
                    req.pp_progress = (n - req.reused) / (req.prompt - req.reused)
            elif m := RE_PP_PROGRESS.search(line):
                req.pp_progress = float(m.group(1))
            elif "prompt done" in line:
                req.pp_progress = 1.0
                req.status = "Generating"
            elif "stop processing" in line or "release" in line:
                if req.status not in ("Done", "Cancelled"):
                    req.status = "Cancelled" if "cancel" in line.lower() else "Done"
            elif "print_timing" in line:
                self._timing = req
            return

        # older builds print the timing block on untagged lines after "print_timing"
        if self._timing is not None:
            self._timing_line(self._timing, line)

    def _timing_line(self, req: Request, line: str) -> bool:
        if m := RE_PROMPT_EVAL.search(line):
            req.pp_tps = float(m.group(3))
            if not req.prompt:
                req.prompt = req.reused + int(m.group(2))
        elif m := RE_EVAL.search(line):
            req.output = int(m.group(2))
            req.tps = float(m.group(3))
        elif m := RE_TOTAL.search(line):
            req.duration = float(m.group(1)) / 1000.0
            if req.status != "Cancelled":
                req.status = "Done"
            self._timing = None
        else:
            return False
        return True

    def request_for_task(self, task: object) -> Request | None:
        try:
            return self._by_task.get(int(task))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def current(self) -> Request | None:
        for r in reversed(self.requests):
            if r.status in ("Reading", "Generating"):
                return r
        return None
