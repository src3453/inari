"""Benchmark-driven auto-tuning for a profile.

The heuristic in :mod:`inari.tuning` guesses from VRAM and GGUF metadata; this module *measures*.
Like the community tuners for llama.cpp (llama-optimus, llama-dyno, …) it tries candidate settings,
times a fixed workload and keeps the fastest — but instead of ``llama-bench`` it starts the real
``llama-server`` with the profile's own command line (same context size, KV cache type, parallel
slots, mmproj …). So a candidate that would run out of memory at your context length fails here
too, and the numbers are the ones you will get when serving.

Search (coordinate descent, cheapest first):

1. baseline — the profile as it is now, plus llama.cpp's own ``--fit on`` for comparison
2. placement — GPU layers (dense) or MoE expert layers on the CPU (``--n-cpu-moe``), found with a
   step-halving hill climb that backs off towards the CPU when a candidate fails to load
3. ``--ubatch-size`` / ``--batch-size`` — bigger micro-batches speed up prompt processing
4. ``--threads`` — only when part of the model runs on the CPU
5. (thorough) placement again, ±1 around the winner, since a bigger micro-batch costs VRAM
"""

from __future__ import annotations

import math
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import psutil

from inari import argspec, hfcache
from inari.argspec import Catalog
from inari.config import Config, logs_dir
from inari.gguf import ModelInfo
from inari.profiles import Profile
from inari.tuning import Hardware

GOALS = {
    "balanced": "a typical chat request (long prompt + reply)",
    "gen": "generation speed (tokens/s while replying)",
    "prompt": "prompt processing speed (tokens/s while reading)",
}
DEPTHS = {"quick": 10, "thorough": 24}  # max server starts
KNOBS = ("fit", "n-gpu-layers", "n-cpu-moe", "cpu-moe", "ubatch-size", "batch-size", "threads")
SHORT = {"fit": "fit", "n-gpu-layers": "ngl", "n-cpu-moe": "ncmoe", "cpu-moe": "cmoe", "ubatch-size": "ub",
         "batch-size": "b", "threads": "t"}
ALL = 999  # n-gpu-layers: everything
_OOM_RE = re.compile(r"out of memory|failed to allocate|unable to allocate|cudaMalloc failed|hipMalloc|"
                     r"ErrorOutOfDeviceMemory|not enough memory|failed to load model", re.I)
_FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
           "report on renewable energy, distributed systems and the history of mathematics. ")


class Cancelled(Exception):
    pass


@dataclass
class Trial:
    knobs: dict[str, Any]
    label: str
    status: str = "pending"  # running | ok | fail
    pp: float = 0.0  # prompt tokens/s
    tg: float = 0.0  # generated tokens/s
    load_s: float = 0.0
    error: str = ""
    score: float = 0.0
    note: str = ""  # stage, e.g. "placement"


def describe(knobs: dict[str, Any]) -> str:
    parts = []
    for k in KNOBS:
        v = knobs.get(k)
        if v is None or v is False:
            continue
        if k == "n-gpu-layers" and v == ALL:
            v = "all"
        parts.append(SHORT[k] if v is True else f"{SHORT[k]} {v}")
    return " · ".join(parts) or "llama.cpp defaults"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill(proc: subprocess.Popen) -> None:
    try:
        p = psutil.Process(proc.pid)
        procs = p.children(recursive=True) + [p]
    except psutil.NoSuchProcess:
        return
    for x in procs:
        try:
            x.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=8)
    for x in alive:
        try:
            x.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


class Tuner:
    """Runs in a background thread; the UI reads ``trials`` / ``status`` / ``best`` while it works."""

    def __init__(self, cfg: Config, cat: Catalog, hw: Hardware, info: ModelInfo | None, size: int,
                 profile: Profile, *, binary: str, goal: str = "balanced", depth: str = "quick") -> None:
        self.cfg, self.cat, self.hw, self.info, self.size = cfg, cat, hw, info, size
        self.profile, self.binary = profile, binary
        self.goal, self.depth = goal, depth
        self.max_trials = DEPTHS.get(depth, 10)
        self.base = argspec.from_args(profile.args, cat)  # canonical keys
        self.trials: list[Trial] = []
        self.status = "starting…"
        self.done = False
        self.error = ""
        self.started = 0.0
        self.finished = 0.0
        self.log_path: Path = logs_dir() / f"autotune-{profile.name or 'new'}.log"
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._tokens: list[int] | None = None
        self._thread: threading.Thread | None = None

        gpu = bool(hw.devices)
        self.n_gen = 256 if depth == "thorough" else 128
        n_prompt = 2048 if gpu else 512
        ctx = self.base.get("ctx-size") or (info.ctx_train if info else 0) or 4096
        try:
            ctx = int(ctx)
        except (TypeError, ValueError):
            ctx = 4096
        self.n_prompt = max(64, min(n_prompt, ctx - self.n_gen - 16))
        self.repeats = 2 if depth == "thorough" else 1

    # ───────── public ─────────
    def start(self) -> None:
        self.started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None:
            threading.Thread(target=_kill, args=(proc,), daemon=True).start()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def runs(self) -> int:
        return sum(1 for t in self.trials if t.status in ("ok", "fail"))

    @property
    def baseline(self) -> Trial | None:
        return self.trials[0] if self.trials and self.trials[0].status == "ok" else None

    @property
    def best(self) -> Trial | None:
        ok = [t for t in self.trials if t.status == "ok"]
        return max(ok, key=lambda t: t.score) if ok else None

    def changes(self) -> dict[str, Any]:
        """Knob values of the winner that differ from the profile (None = drop the setting)."""
        b = self.best
        if b is None:
            return {}
        return {k: b.knobs.get(k) for k in KNOBS if b.knobs.get(k) != self.base.get(k)}

    def gain(self) -> float | None:
        b, base = self.best, self.baseline
        if not b or not base or not base.score:
            return None
        return b.score / base.score - 1

    def unit(self) -> str:
        return {"balanced": "req/min", "gen": "tok/s", "prompt": "tok/s"}[self.goal]

    def elapsed(self) -> float:
        end = self.finished or time.monotonic()
        return end - self.started if self.started else 0.0

    # ───────── scoring ─────────
    def _score(self, pp: float, tg: float) -> float:
        if self.goal == "gen":
            return tg
        if self.goal == "prompt":
            return pp
        if not pp or not tg:
            return 0.0
        # requests per minute for a "typical" chat turn: a long prompt and a medium reply
        secs = 2048 / pp + 256 / tg
        return 60 / secs

    # ───────── trial execution ─────────
    def _check(self) -> None:
        if self.cancelled:
            raise Cancelled

    def _left(self) -> bool:
        return self.runs < self.max_trials

    def _key(self, knobs: dict[str, Any]) -> tuple:
        return tuple((k, knobs.get(k)) for k in KNOBS)

    def _find(self, knobs: dict[str, Any]) -> Trial | None:
        key = self._key(knobs)
        return next((t for t in self.trials if self._key(t.knobs) == key), None)

    def _try(self, knobs: dict[str, Any], note: str) -> Trial | None:
        """Measure ``knobs`` (cached). None when the trial budget is used up."""
        self._check()
        knobs = {k: knobs.get(k) for k in KNOBS}
        old = self._find(knobs)
        if old is not None:
            return old
        if not self._left():
            return None
        t = Trial(knobs, describe(knobs), note=note)
        self.trials.append(t)
        self._measure(t)
        self._check()
        return t

    def _command(self, knobs: dict[str, Any], port: int) -> list[str]:
        vals = dict(self.base)
        for k, v in knobs.items():
            if v is None:
                vals.pop(k, None)
            else:
                vals[k] = v
        p = self.profile
        prof = Profile(name="autotune", model=p.model, host="127.0.0.1", port=port,
                       args=argspec.to_args(vals, self.cat), extra=list(p.extra), env=dict(p.env))
        return prof.build_command(self.cfg, binary=self.binary)

    def _measure(self, t: Trial) -> None:
        t.status = "running"
        port = _free_port()
        cmd = self._command(t.knobs, port)
        env = dict(os.environ)
        if self.cfg.hf_cache and "LLAMA_CACHE" not in env:
            env["LLAMA_CACHE"] = str(hfcache.cache_dir(self.cfg))
        env.update(self.profile.env)
        kw: dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        with open(self.log_path, "ab") as log:
            log.write(f"\n===== trial {len(self.trials)}: {t.label}\n$ {subprocess.list2cmdline(cmd)}\n".encode())
            log.flush()
            offset = log.tell()
            self.status = f"loading model — {t.label}"
            t0 = time.monotonic()
            try:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, **kw)
            except OSError as e:
                t.status, t.error = "fail", str(e)
                return
        self._proc = proc
        url = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(timeout=httpx.Timeout(900, connect=2)) as c:
                self._wait_ready(c, url, proc, t, t0)
                t.load_s = time.monotonic() - t0
                self._bench(c, url, t)
            t.score = self._score(t.pp, t.tg)
            t.status = "ok"
        except Cancelled:
            t.status, t.error = "fail", "cancelled"
        except _Failed as e:
            t.status, t.error = "fail", str(e) or self._reason(offset)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            t.status = "fail"
            t.error = self._reason(offset) if proc.poll() is not None else f"request failed: {e}"
        finally:
            _kill(proc)
            self._proc = None

    def _wait_ready(self, c: httpx.Client, url: str, proc: subprocess.Popen, t: Trial, t0: float) -> None:
        gib = self.size / 1024 ** 3 if self.size else 8
        limit = max(180.0, gib * 30)
        while True:
            self._check()
            if proc.poll() is not None:
                raise _Failed("")
            try:
                if c.get(url + "/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() - t0 > limit:
                raise _Failed(f"not ready after {limit:.0f}s")
            self.status = f"loading model ({time.monotonic() - t0:.0f}s) — {t.label}"
            time.sleep(0.4)

    def _prompt(self, c: httpx.Client, url: str) -> list[int]:
        if self._tokens is None:
            r = c.post(url + "/tokenize", json={"content": _FILLER})
            r.raise_for_status()
            toks = [x if isinstance(x, int) else x.get("id") for x in r.json().get("tokens", [])]
            toks = [x for x in toks if isinstance(x, int)] or [1]
            self._tokens = toks
        reps = math.ceil(self.n_prompt / len(self._tokens))
        return (self._tokens * reps)[: self.n_prompt]

    def _bench(self, c: httpx.Client, url: str, t: Trial) -> None:
        prompt = self._prompt(c, url)
        self.status = f"warming up — {t.label}"
        c.post(url + "/completion", json={"prompt": prompt[:32], "n_predict": 8, "cache_prompt": False})
        for i in range(self.repeats):
            self._check()
            self.status = (f"measuring {self.n_prompt} prompt + {self.n_gen} generated tokens"
                           + (f" ({i + 1}/{self.repeats})" if self.repeats > 1 else "") + f" — {t.label}")
            r = c.post(url + "/completion", json={
                "prompt": prompt, "n_predict": self.n_gen, "cache_prompt": False, "ignore_eos": True,
                "temperature": 0.0, "stream": False,
            })
            if r.status_code != 200:
                raise _Failed(f"HTTP {r.status_code}: {r.text[:120]}")
            tm = r.json()["timings"]
            t.pp = max(t.pp, float(tm.get("prompt_per_second") or 0))
            t.tg = max(t.tg, float(tm.get("predicted_per_second") or 0))

    def _reason(self, offset: int) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return "server exited"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if _OOM_RE.search(ln):
                return "out of memory" if re.search(r"memory|alloc", ln, re.I) else ln[-120:]
        for ln in reversed(lines):
            if "error" in ln.lower() or "failed" in ln.lower():
                return ln[-120:]
        return "server exited" + (f": {lines[-1][-100:]}" if lines else "")

    # ───────── the search ─────────
    def _run(self) -> None:
        try:
            self._plan()
            self.status = "done"
        except Cancelled:
            self.status = "cancelled"
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
        finally:
            self.finished = time.monotonic()
            self.done = True

    def _start_knobs(self) -> dict[str, Any]:
        return {k: self.base.get(k) for k in KNOBS}

    def _need(self) -> int:
        """Bytes of VRAM the whole model wants at the profile's context (weights + KV, rough)."""
        kv = 0
        if self.info:
            ctx = self.base.get("ctx-size") or self.info.ctx_train
            try:
                kv = self.info.kv_bytes(int(ctx), str(self.base.get("cache-type-k") or "f16"),
                                        str(self.base.get("cache-type-v") or "f16"))
            except (TypeError, ValueError):
                kv = 0
        return int(self.size + kv)

    def _plan(self) -> None:
        cur = self._start_knobs()
        self._try(cur, "baseline")
        gpu = bool(self.hw.devices)
        info = self.info
        fit_off: dict[str, Any] = {"fit": "off"} if "fit" in self.cat else {}

        # llama.cpp's own fitting as a second reference point
        manual = cur.get("fit") == "off" or any(cur.get(k) is not None for k in ("n-gpu-layers", "n-cpu-moe",
                                                                                 "cpu-moe"))
        if gpu and "fit" in self.cat and manual:
            self._try({**cur, "fit": "on", "n-gpu-layers": None, "n-cpu-moe": None, "cpu-moe": None}, "baseline")

        # ── placement ──
        place: Callable[[int], dict[str, Any]] | None = None
        lo = hi = start = 0
        toward_cpu = 1
        if gpu and info and info.layers and self.size:
            vram = self.hw.vram
            need = self._need()
            usable = vram * 0.85
            layers = info.layers
            keep = {k: cur.get(k) for k in ("ubatch-size", "batch-size", "threads")}
            if info.moe and "n-cpu-moe" in self.cat:
                def place(x: int) -> dict[str, Any]:
                    return {**keep, **fit_off, "n-gpu-layers": ALL, "n-cpu-moe": x or None, "cpu-moe": None}
                lo, hi, toward_cpu = 0, layers, 1
                start = 0 if need <= usable else min(layers, math.ceil(layers * (need - usable) / max(need, 1)) + 1)
                if isinstance(cur.get("n-cpu-moe"), int):
                    start = cur["n-cpu-moe"]
            elif "n-gpu-layers" in self.cat:
                def place(x: int) -> dict[str, Any]:
                    return {**keep, **fit_off, "n-gpu-layers": ALL if x >= layers + 1 else x, "n-cpu-moe": None,
                            "cpu-moe": None}
                lo, hi, toward_cpu = 0, layers + 1, -1
                start = hi if need <= usable else max(0, int((layers + 1) * usable / max(need, 1)))
                ngl = cur.get("n-gpu-layers")
                if isinstance(ngl, int) and ngl < hi:
                    start = ngl
        if place is not None:
            fits = self._need() <= self.hw.vram * 0.85
            self.status = "searching layer placement"
            x = self._climb(place, lo, hi, start, toward_cpu, local=fits and self.depth == "quick")
            cur = dict(self.best.knobs) if self.best else place(x)

        # ── micro-batch ──
        best = self.best
        if best is not None:
            cur = dict(best.knobs)
        sizes = [s for s in (1024, 2048, 4096 if self.depth == "thorough" else 0) if s and s <= self.n_prompt]
        if not gpu:
            sizes = [s for s in (256, 1024) if s <= self.n_prompt]
        base_b = int(cur.get("batch-size") or 2048)
        for ub in sizes:
            if not self._left():
                break
            self.status = f"trying micro-batch {ub}"
            self._try({**cur, "ubatch-size": ub, "batch-size": max(ub, base_b)}, "batch")
        best = self.best
        if best is not None:
            cur = dict(best.knobs)

        # ── threads (only matters when the CPU does part of the work) ──
        cores = self.hw.cores or (psutil.cpu_count(logical=False) or 0)
        logical = psutil.cpu_count() or cores
        cpu_work = (not gpu or bool(cur.get("n-cpu-moe")) or bool(cur.get("cpu-moe"))
                    or (isinstance(cur.get("n-gpu-layers"), int) and cur["n-gpu-layers"] < ALL)
                    or (cur.get("fit") != "off" and self._need() > self.hw.vram * 0.85))
        if cpu_work and cores > 2:
            cands = [cores // 2]
            if self.depth == "thorough":
                cands += [max(1, cores - 2), logical]
            for n in dict.fromkeys(cands):
                if not self._left():
                    break
                if n == cur.get("threads"):
                    continue
                self.status = f"trying {n} CPU threads"
                self._try({**cur, "threads": n}, "threads")

        # ── thorough: placement once more around the winner (a bigger ubatch costs VRAM) ──
        best = self.best
        if self.depth == "thorough" and place is not None and best is not None and best.knobs.get("fit") != "on":
            key = "n-cpu-moe" if toward_cpu == 1 else "n-gpu-layers"
            v = best.knobs.get(key)
            x = (v or 0) if toward_cpu == 1 else (hi if v == ALL else int(v or 0))
            extra = {k: best.knobs.get(k) for k in ("ubatch-size", "batch-size", "threads")}
            for c in (x - 1, x + 1):
                if lo <= c <= hi and self._left():
                    self._try({**place(c), **extra}, "placement")

    def _climb(self, place: Callable[[int], dict[str, Any]], lo: int, hi: int, start: int, toward_cpu: int,
               *, local: bool) -> int:
        """Step-halving hill climb over one integer. Failed loads score 0 and push towards the CPU."""
        x = max(lo, min(hi, start))
        t = self._try(place(x), "placement")
        if t is None:
            return x
        # move towards the CPU until something loads
        step = max(1, (hi - lo) // 8)
        while t.status != "ok":
            nx = max(lo, min(hi, x + toward_cpu * step))
            if nx == x:
                return x
            x = nx
            t = self._try(place(x), "placement")
            if t is None:
                return x
            step *= 2
        best_x, best_s = x, t.score
        if local:  # it fits on paper and loaded fine — one neighbour is enough to confirm
            step = 1
        else:
            step = max(1, (hi - lo) // 8)
        while step >= 1:
            moved = False
            for c in (best_x - step, best_x + step):
                if not lo <= c <= hi:
                    continue
                r = self._try(place(c), "placement")
                if r is None:
                    return best_x
                if r.status == "ok" and r.score > best_s * 1.01:
                    best_x, best_s, moved = c, r.score, True
                    break
            if not moved:
                if local:
                    break
                step //= 2
        return best_x


class _Failed(Exception):
    pass


# ───────── shared rendering (wizard + `inari tune`) ─────────
def render(tuner: Tuner, p: Any, rows: int = 12):
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    head = Text()
    head.append("Auto-tune", style=f"bold {p.accent}")
    head.append(f"   goal: {tuner.goal} — {GOALS[tuner.goal]}   ·   {tuner.depth} "
                f"({tuner.runs}/{tuner.max_trials} runs · {tuner.elapsed():.0f}s)", style=p.dim)
    t = Table(box=None, expand=True, padding=(0, 1), show_edge=False, header_style=f"bold {p.text}")
    t.add_column("#", justify="right", width=3, no_wrap=True)
    t.add_column("settings", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("prompt t/s", justify="right", no_wrap=True)
    t.add_column("gen t/s", justify="right", no_wrap=True)
    t.add_column(tuner.unit(), justify="right", no_wrap=True)
    t.add_column("load", justify="right", no_wrap=True)
    t.add_column("", no_wrap=True, max_width=36, overflow="ellipsis")
    best = tuner.best
    trials = tuner.trials[-rows:]
    first = len(tuner.trials) - len(trials)
    for i, tr in enumerate(trials, start=first + 1):
        win = tr is best
        st = f"bold {p.good}" if win else (p.text if tr.status == "ok" else p.dim)
        if tr.status == "ok":
            status = Text("★ best" if win else ("baseline" if i == 1 else ""), style=p.good if win else p.dim)
            nums = (f"{tr.pp:,.0f}", f"{tr.tg:,.1f}", f"{tr.score:,.1f}", f"{tr.load_s:.0f}s")
        elif tr.status == "running":
            status = Text("running…", style=p.accent2)
            nums = ("", "", "", "")
        else:
            status = Text(tr.error or "failed", style=p.bad)
            nums = ("—", "—", "—", "")
        t.add_row(str(i), Text(tr.label, style=st), *[Text(n, style=st) for n in nums], status)
    foot = Text()
    if not tuner.done:
        foot.append("▸ " + tuner.status, style=p.accent2)
    elif tuner.error:
        foot.append(f"✖ {tuner.error}", style=p.bad)
    elif best is None:
        foot.append("✖ no candidate ran — see " + str(tuner.log_path), style=p.bad)
    else:
        g = tuner.gain()
        foot.append("✔ best: ", style=f"bold {p.good}")
        foot.append(best.label, style=f"bold {p.text}")
        foot.append(f"   prompt {best.pp:,.0f} · gen {best.tg:,.1f} tok/s", style=p.text)
        if g is not None:
            foot.append(f"   {g:+.0%} vs current", style=p.good if g > 0.005 else p.dim)
        if tuner.cancelled:
            foot.append("   (cancelled early)", style=p.warn)
    return Group(head, Text(""), t, Text(""), foot)
