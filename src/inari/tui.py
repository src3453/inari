"""The Inari monitor: a full-screen, Strata-style dashboard rendered with rich."""

from __future__ import annotations

import time
import webbrowser
from datetime import datetime
from typing import Iterable

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inari import theme as th
from inari.collector import STATES, Collector, Snapshot
from inari.keys import KeyReader
from inari.logparse import Request, line_level

VIEWS = ["Overview", "Serving", "Requests", "Engine", "Hardware", "Logs", "Connect"]
PARTIAL = " ▏▎▍▌▋▊▉"
SPARK = " ▁▂▃▄▅▆▇█"
GB = 1024 ** 3
MB = 1024 ** 2


# ───────────────────────── formatting ─────────────────────────
def k(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}K" if n >= 10_000 else f"{n / 1000:.1f}K"
    return f"{int(n)}"


def pct(a: float, b: float) -> float:
    return max(0.0, min(1.0, a / b)) if b else 0.0


def dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


# ───────────────────────── widgets ─────────────────────────
class Bar:
    """Horizontal bar that fills whatever width it is given."""

    def __init__(self, frac: float, color: str, track: str) -> None:
        self.frac, self.color, self.track = max(0.0, min(1.0, frac)), color, track

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        w = max(options.max_width, 1)
        full = self.frac * w
        n = int(full)
        part = int((full - n) * 8)
        t = Text("█" * n, style=self.color)
        if n < w:
            if part:
                t.append(PARTIAL[part], style=f"{self.color} on {self.track}")
                n += 1
            t.append(" " * (w - n), style=f"on {self.track}")
        yield t


class Spark:
    """Multi-row sparkline, right-aligned to the newest sample."""

    def __init__(self, values: list[float], color: str, rows: int = 2, floor: float = 0.0) -> None:
        self.values, self.color, self.rows, self.floor = values, color, rows, floor

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        w = max(options.max_width, 1)
        vals = self.values[-w:]
        vals = [0.0] * (w - len(vals)) + list(vals)
        top = max(max(vals, default=0.0), self.floor) or 1.0
        levels = [round(v / top * self.rows * 8) if v > 0 else 0 for v in vals]
        for r in range(self.rows - 1, -1, -1):
            line = "".join(SPARK[max(0, min(8, lv - r * 8))] for lv in levels)
            yield Text(line, style=self.color)


# ───────────────────────── app ─────────────────────────
class App:
    def __init__(self, collector: Collector, *, theme_name: str = "inari", managed: bool = False,
                 console: Console | None = None) -> None:
        self.c = collector
        self.console = console or Console(highlight=False)
        self.theme_name = theme_name
        self.p = th.get(theme_name)
        self.managed = managed
        self.view = 0
        self.frozen = False
        self.filter = ""
        self.filter_mode = False
        self.confirm_quit = False
        self.help = False
        self.scroll = 0
        self.snap = collector.snapshot()
        self.result: str = "quit"
        self.flash: tuple[str, float] | None = None

    # ── loop ──
    def run(self) -> str:
        with KeyReader() as keys, Live(console=self.console, screen=True, auto_refresh=False,
                                       redirect_stdout=False, redirect_stderr=False) as live:
            last = 0.0
            while True:
                pressed = keys.drain()
                for key in pressed:
                    if self.handle(key):
                        return self.result
                now = time.monotonic()
                if pressed or now - last >= 0.25:
                    if not self.frozen:
                        self.snap = self.c.snapshot()
                    live.update(self.render(), refresh=True)
                    last = now
                time.sleep(0.04)

    def say(self, msg: str) -> None:
        self.flash = (msg, time.monotonic() + 2.5)

    def handle(self, key: str) -> bool:
        if self.filter_mode:
            if key in ("enter", "esc"):
                if key == "esc":
                    self.filter = ""
                self.filter_mode = False
            elif key == "backspace":
                self.filter = self.filter[:-1]
            elif len(key) == 1 and key.isprintable():
                self.filter += key
            return False
        if self.confirm_quit:
            if key in ("y", "s"):
                self.result = "stop"
                return True
            if key in ("n", "k"):
                self.result = "keep"
                return True
            self.confirm_quit = False
            return False
        if key in ("q", "ctrl+c"):
            if self.managed and self.snap.alive:
                self.confirm_quit = True
                return False
            self.result = "quit"
            return True
        if key in "1234567" and len(key) == 1:
            self.view, self.scroll, self.help = int(key) - 1, 0, False
        elif key in ("right", "tab"):
            self.view, self.scroll = (self.view + 1) % len(VIEWS), 0
        elif key == "left":
            self.view, self.scroll = (self.view - 1) % len(VIEWS), 0
        elif key == "/":
            self.filter_mode = True
        elif key == "esc":
            self.filter, self.help = "", False
        elif key == " ":
            self.frozen = not self.frozen
        elif key == "r":
            self.c.poke()
            self.say("refreshed")
        elif key == "t":
            self.theme_name = th.next_name(self.theme_name)
            self.p = th.get(self.theme_name)
            self.say(f"theme: {self.theme_name}")
        elif key == "b":
            webbrowser.open(self.snap.url)
            self.say(f"opened {self.snap.url}")
        elif key == "?":
            self.help = not self.help
        elif key in ("up", "k"):
            self.scroll += 1
        elif key in ("down", "j"):
            self.scroll = max(0, self.scroll - 1)
        elif key == "pgup":
            self.scroll += 15
        elif key == "pgdn":
            self.scroll = max(0, self.scroll - 15)
        elif key == "end":
            self.scroll = 0
        return False

    # ── frame ──
    def render(self) -> RenderableType:
        w, h = self.console.size
        body_h = max(h - 6, 5)
        root = Layout()
        root.split_column(
            Layout(self.header(), size=2),
            Layout(Text("─" * w, style=self.p.dim), size=1),
            Layout(self.body(body_h), size=body_h),
            Layout(self.hints(), size=1),
            Layout(Text(" "), size=1),  # note: an empty Text is falsy -> Layout placeholder
            Layout(self.statusbar(), size=1),
        )
        return root

    def body(self, height: int) -> RenderableType:
        if self.help:
            return self.view_help()
        return [self.view_overview, self.view_serving, self.view_requests, self.view_engine,
                self.view_hardware, self.view_logs, self.view_connect][self.view](height)

    # ── chrome ──
    def state_color(self, state: str) -> str:
        p = self.p
        return {"Generating": p.accent, "Reading": p.info, "Idle": p.good, "Queued": p.warn,
                "Loading": p.accent2, "Error": p.bad}.get(state, p.text)

    def header(self) -> RenderableType:
        p, s = self.p, self.snap
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True, overflow="ellipsis")
        g.add_column(justify="right", no_wrap=True)
        left = Text()
        left.append("INARI", style=f"bold {p.accent}")
        left.append("  " + VIEWS[self.view], style=f"bold {p.accent2}")
        left.append("    " + s.url, style=f"bold {p.text}")
        if s.model:
            left.append("   " + s.model, style=f"bold {p.text}")
        if s.n_ctx:
            left.append(f" {k(s.n_ctx)}", style=p.accent2)
        right = Text()
        sc = self.state_color(s.state)
        right.append("● ", style=sc)
        right.append(s.detail, style=sc)
        if s.state in ("Generating", "Reading"):
            right.append(f" · {s.progress * 100:.0f}%", style=sc)
        right.append("   ")
        right.append("[FROZEN]" if self.frozen else "[LIVE]", style=f"bold {p.warn if self.frozen else p.accent}")
        g.add_row(left, right)

        tabs = Text()
        for i, v in enumerate(VIEWS):
            if i == self.view:
                tabs.append(f"[{i + 1} {v}]", style=f"bold {p.accent}")
            else:
                tabs.append(f" {i + 1} {v} ", style=p.text)
            tabs.append(" ")
        g.add_row(tabs, Text(""))
        return g

    def hints(self) -> RenderableType:
        p = self.p
        if self.filter_mode:
            return Text.assemble(("/ ", f"bold {p.accent}"), ("filter: ", p.dim), (self.filter, p.text),
                                 ("▌", p.accent), ("   Enter apply · Esc clear", p.dim))
        if self.confirm_quit:
            return Text.assemble(("Stop llama-server?  ", f"bold {p.warn}"), ("y", f"bold {p.accent}"),
                                 (" stop   ", p.text), ("n", f"bold {p.accent}"), (" keep serving in background   ", p.text),
                                 ("any other key", f"bold {p.accent}"), (" cancel", p.text))
        t = Text()
        for key, label in (("1-7", "views"), ("/", "filter"), ("Space", "freeze"), ("r", "refresh"),
                           ("t", "theme"), ("b", "open in browser"), ("?", "help"), ("q", "quit")):
            t.append(key, style=f"bold {p.accent}")
            t.append(f" {label}    ", style=p.dim)
        if self.filter:
            t.append(f"filter: {self.filter}", style=p.accent2)
        return t

    def statusbar(self) -> RenderableType:
        p, s = self.p, self.snap
        t = Text()
        if self.flash and time.monotonic() < self.flash[1]:
            t.append(self.flash[0], style=f"bold {p.accent2}")
            return t
        if self.managed:
            t.append("Managed", style=f"bold {p.accent}")
            t.append(f" · pid {s.pid}" if s.pid else "", style=p.text)
            if s.started:
                t.append(f" · up {dur(time.time() - s.started)}", style=p.text)
            t.append("   |   q asks: keep serving or stop", style=p.text)
        else:
            t.append("Read only", style=f"bold {p.accent}")
            t.append("   |   q quit", style=p.text)
        t.append("   |   7 connect an app", style=p.text)
        if s.info.errors:
            t.append(f"   |   {s.info.errors} errors in log", style=p.bad)
        return t

    # ── shared bits ──
    def card(self, title: str, value: str, sub: str, spark: list[float] | None = None,
             spark_color: str | None = None, floor: float = 0.0) -> Panel:
        p = self.p
        parts: list[RenderableType] = [Text(value, style=f"bold {p.text}", no_wrap=True, overflow="ellipsis"),
                                       Text(sub, style=p.dim, no_wrap=True, overflow="ellipsis")]
        parts.append(Spark(spark, spark_color or p.accent, rows=2, floor=floor) if spark is not None else Text("\n"))
        return Panel(Group(*parts), title=Text(f" {title} ", style=p.text), title_align="left",
                     border_style=p.dim, box=box.ROUNDED, padding=(0, 1), height=6)

    def labeled_bar(self, label: str, value: str, frac: float, color: str | None = None) -> Group:
        p = self.p
        head = Table.grid(expand=True)
        head.add_column(no_wrap=True)
        head.add_column(justify="right", no_wrap=True)
        head.add_row(Text(label, style=f"bold {p.accent}"), Text(value, style=p.text))
        return Group(head, Bar(frac, color or p.accent, p.track))

    def grid(self, items: list[RenderableType], cols: int) -> Table:
        g = Table.grid(expand=True, padding=(0, 1))
        for _ in range(cols):
            g.add_column(ratio=1)
        for i in range(0, len(items), cols):
            row = items[i:i + cols]
            g.add_row(*(row + [Text("")] * (cols - len(row))))
        return g

    def requests_table(self, reqs: Iterable[Request], limit: int, wide: bool = False) -> Table:
        p = self.p
        t = Table(box=None, show_edge=False, pad_edge=False, header_style=f"{p.text}", expand=True, padding=(0, 1))
        cols = [("TIME", "left"), ("STATUS", "left"), ("PROMPT", "right"), ("REUSED", "right")]
        if wide:
            cols = [("TASK", "right")] + cols + [("PP TOK/S", "right")]
        cols += [("OUTPUT", "right"), ("TOK/S", "right"), ("DURATION", "right")]
        for name, j in cols:
            t.add_column(name, justify=j, no_wrap=True)
        rows = list(reqs)
        if self.filter:
            f = self.filter.lower()
            rows = [r for r in rows if f in r.status.lower() or f in str(r.task)]
        rows = rows[::-1]
        if wide:
            rows = rows[min(self.scroll, max(len(rows) - 1, 0)):]
        status_col = {"Done": p.good, "Generating": p.accent, "Reading": p.info, "Cancelled": p.warn}
        for r in rows[:limit]:
            tm = datetime.fromtimestamp(r.started).strftime("%H:%M:%S") if r.started else "  —"
            cells = [Text(tm, style=p.text), Text(r.status, style=status_col.get(r.status, p.text)),
                     Text(f"{r.prompt:,}", style=p.text), Text(f"{r.reused:,}", style=p.text)]
            if wide:
                cells = [Text(str(r.task), style=p.dim)] + cells + [
                    Text(f"{r.pp_tps:,.1f}" if r.pp_tps else "--", style=p.text)]
            cells += [Text(f"{r.output:,}", style=p.text),
                      Text(f"{r.tps:.1f}" if r.tps else "--", style=p.text),
                      Text(f"{r.duration:.1f} s" if r.duration else ("…" if r.status != "Done" else "--"), style=p.text)]
            t.add_row(*cells)
        if not rows:
            t.add_row(Text("no requests yet", style=p.dim), *[Text("")] * (len(cols) - 1))
        return t

    # ── views ──
    def state_strip(self) -> Group:
        p, s = self.p, self.snap
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True)
        g.add_column(justify="right", no_wrap=True)
        left = Text("Model state  ", style=f"bold {p.accent}")
        for st in STATES:
            if st == s.state:
                left.append(f"[{st}]", style=f"bold {self.state_color(st)}")
            else:
                left.append(st, style=p.dim)
            left.append("   ")
        right = Text(f"{s.progress_text} · {s.progress * 100:.0f}%" if s.progress_text else "", style=p.text)
        g.add_row(left, right)
        return Group(g, Bar(s.progress, self.state_color(s.state), p.track))

    def hw_cards(self) -> list[Panel]:
        p, s = self.p, self.snap
        hw, info = s.hw, s.info
        gpu = hw.gpus[0] if hw.gpus else None
        dev_desc = next(iter(info.devices.values()), None)
        vram_est = info.vram_used_mib() * MB
        off = f"{info.offload[0]}/{info.offload[1]} layers" if info.offload else ""
        avg = s.metrics.get("predicted_tokens_seconds")

        cards = [
            self.card("Speed", f"{s.speed:.1f} tok/s",
                      "now" + (f" · avg {avg:.1f}" if avg else ""), s.speed_hist, floor=1.0),
            self.card("Prompt", f"{s.pp_speed:,.0f} tok/s", "prompt eval · last request", s.pp_hist, p.info, floor=1.0),
        ]
        if gpu:
            cards.append(self.card("GPU load", f"{gpu.util:.0f} %" if gpu.util is not None else "-- %", gpu.name,
                                   s.gpu_hist, floor=100.0))
            used = f"{gpu.mem_used / GB:.1f}" if gpu.mem_used is not None else "--"
            tot = f"{gpu.mem_total / GB:.0f}" if gpu.mem_total else "--"
            cards.append(self.card("VRAM", f"{used} / {tot} GB",
                                   f"llama {vram_est / GB:.1f} GB" + (f" · {off}" if off else "")))
            temp = f"{gpu.temp:.0f} °C" if gpu.temp is not None else "-- °C"
            pw = f"{gpu.power:.0f} W" if gpu.power is not None else "-- W"
            if gpu.power_limit:
                pw += f" / {gpu.power_limit:.0f} W"
            cards.append(self.card("GPU temp / power", temp, pw))
        else:
            cards.append(self.card("GPU", dev_desc[0] if dev_desc else "-- %",
                                   (f"{next(iter(info.devices))}" if dev_desc else "no GPU device in log")
                                   + ("  · load n/a" if dev_desc else "")))
            free = f" of {dev_desc[1] / 1024:.1f} GB free" if dev_desc else ""
            cards.append(self.card("VRAM", f"{vram_est / GB:.2f} GB" if vram_est else "-- GB",
                                   (f"llama buffers{free}") + (f" · {off}" if off else "")))
            cards.append(self.card("GPU temp", "-- °C", "pip install inari[nvidia] for NVML"))
        cards += [
            self.card("CPU", f"{hw.cpu:.0f} %", f"{hw.cores} cores · {hw.threads} threads", s.cpu_hist, floor=100.0),
            self.card("Disk read", f"{hw.disk_read / MB:.1f} MB/s", f"write {hw.disk_write / MB:.1f} MB/s",
                      s.disk_hist, floor=5 * MB),
        ]
        if hw.proc_rss is not None:
            cards.append(self.card("llama-server", f"{hw.proc_rss / GB:.2f} GB RSS",
                                   f"cpu {hw.proc_cpu or 0:.0f}% · {hw.proc_threads} threads"
                                   + (f" · up {dur(time.time() - s.started)}" if s.started else "")))
        else:
            cards.append(self.card("llama-server", "remote" if not s.pid else ("exited" if not s.alive else "--"),
                                   f"health: {s.health or '?'}"))
        return cards

    def view_overview(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        w = self.console.size.width
        cols = 4 if w >= 110 else 2
        cards = self.hw_cards()
        if cols == 2:
            cards = cards[:4] if height < 30 else cards
        info = s.info
        vram_est = info.vram_used_mib()
        dev_free = sum(d[1] for d in info.devices.values())
        gpu = s.hw.gpus[0] if s.hw.gpus else None
        if gpu and gpu.mem_total:
            vram_frac, vram_txt = pct(gpu.mem_used or 0, gpu.mem_total), f"{(gpu.mem_used or 0) / GB:.1f} / {gpu.mem_total / GB:.0f} GB"
        else:
            vram_frac = pct(vram_est, dev_free) if dev_free else 0.0
            vram_txt = f"{vram_est / 1024:.2f} GB" + (f" / {dev_free / 1024:.1f} GB" if dev_free else "")
        ram_frac = pct(s.hw.ram_used, s.hw.ram_total)
        left = Group(
            self.labeled_bar("Context fill", f"{k(s.ctx_used)} / {k(s.n_ctx)}  {pct(s.ctx_used, s.n_ctx) * 100:.0f}%",
                             pct(s.ctx_used, s.n_ctx)),
            self.labeled_bar("VRAM", vram_txt, vram_frac, p.accent2),
            self.labeled_bar("System RAM", f"{s.hw.ram_used / GB:.1f} / {s.hw.ram_total / GB:.0f} GB", ram_frac,
                             p.bad if ram_frac > 0.9 else (p.warn if ram_frac > 0.75 else p.accent)),
        )
        rows_left = max(height - 3 - 13 - 2, 3)
        right = Group(Text("Recent requests", style=f"bold {p.accent}"),
                      self.requests_table(s.requests, rows_left - 1))
        bottom = Table.grid(expand=True, padding=(0, 3))
        bottom.add_column(ratio=2)
        bottom.add_column(ratio=3)
        bottom.add_row(left, right)
        return Group(self.state_strip(), Text(""), self.grid(cards, cols), Text(""), bottom)

    def view_serving(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        t = Table(box=box.SIMPLE_HEAD, border_style=p.dim, header_style=f"bold {p.accent}", expand=True)
        for c, j in (("SLOT", "right"), ("STATE", "left"), ("TASK", "right"), ("CTX", "right"),
                     ("DECODED", "right"), ("REMAIN", "right"), ("TEMP", "right"), ("TOP-P", "right"),
                     ("TOP-K", "right"), ("MIN-P", "right")):
            t.add_column(c, justify=j, style=p.text)
        for slot in s.slots:
            nt = slot.get("next_token") or {}
            nt = nt[0] if isinstance(nt, list) and nt else (nt if isinstance(nt, dict) else {})
            prm = slot.get("params", {}) or {}
            busy = slot.get("is_processing")
            st = ("Generating" if nt.get("n_decoded") else "Reading") if busy else "Idle"
            t.add_row(str(slot.get("id")), Text(st, style=self.state_color(st)),
                      str(slot.get("id_task", "")), f"{slot.get('n_ctx', 0):,}",
                      f"{nt.get('n_decoded', 0):,}", str(nt.get("n_remain", "")),
                      str(prm.get("temperature", "")), str(prm.get("top_p", "")),
                      str(prm.get("top_k", "")), str(prm.get("min_p", "")))
        if not s.slots:
            t.add_row("", Text("no slot data (server loading, or started with --no-slots)", style=p.dim))

        m = s.metrics
        mt = Table.grid(padding=(0, 3))
        mt.add_column(style=p.dim)
        mt.add_column(style=f"bold {p.text}", justify="right")
        rows = [
            ("health", s.health or "?"),
            ("slots", str(s.props.get("total_slots", len(s.slots)))),
            ("processing / deferred", f"{m.get('requests_processing', 0):.0f} / {m.get('requests_deferred', 0):.0f}"),
            ("prompt tokens total", f"{m.get('prompt_tokens_total', 0):,.0f}"),
            ("generated tokens total", f"{m.get('tokens_predicted_total', 0):,.0f}"),
            ("avg prompt tok/s", f"{m.get('prompt_tokens_seconds', 0):,.1f}"),
            ("avg generation tok/s", f"{m.get('predicted_tokens_seconds', 0):,.1f}"),
            ("decode calls", f"{m.get('n_decode_total', 0):,.0f}"),
            ("busy slots per decode", f"{m.get('n_busy_slots_per_decode', 0):.2f}"),
        ]
        for a, b in rows:
            mt.add_row(a, b)
        if not m:
            mt.add_row("metrics", "unavailable (needs --metrics)")
        return Group(self.state_strip(), Text(""),
                     Panel(t, title=Text(" Slots ", style=p.text), title_align="left", border_style=p.dim, box=box.ROUNDED),
                     Panel(mt, title=Text(" Server metrics ", style=p.text), title_align="left", border_style=p.dim,
                           box=box.ROUNDED))

    def view_requests(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        reqs = s.requests
        done = [r for r in reqs if r.status == "Done"]
        summary = Text()
        summary.append("Requests  ", style=f"bold {p.accent}")
        summary.append(f"{len(reqs)} seen · {len(done)} done", style=p.text)
        if done:
            tps = [r.tps for r in done if r.tps]
            reuse = sum(r.reused for r in done) / max(sum(r.prompt for r in done), 1)
            summary.append(f" · avg {sum(tps) / max(len(tps), 1):.1f} tok/s · cache reuse {reuse * 100:.0f}%", style=p.text)
        summary.append("   ↑/↓ scroll · / filter by status or task", style=p.dim)
        return Group(summary, Text(""), self.requests_table(reqs, height - 4, wide=True))

    def view_engine(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        kv = Table.grid(padding=(0, 2))
        kv.add_column(style=p.dim, no_wrap=True)
        kv.add_column(style=p.text)
        props = s.props
        dgs = props.get("default_generation_settings", {}) if props else {}
        add = lambda a, b: kv.add_row(a, str(b)) if b not in (None, "") else None  # noqa: E731
        add("profile", s.name)
        add("model", props.get("model_path") or s.model)
        add("build", props.get("build_info"))
        add("n_ctx", f"{s.n_ctx:,}" if s.n_ctx else "")
        add("slots", props.get("total_slots"))
        mods = props.get("modalities")
        if isinstance(mods, dict):
            add("modalities", ", ".join(k_ for k_, v in mods.items() if v) or "text")
        params = dgs.get("params", {}) if isinstance(dgs.get("params"), dict) else {}
        for key_ in ("temperature", "top_k", "top_p", "min_p", "n_predict"):
            add(f"default {key_}", params.get(key_))
        mi = s.info.model_info
        for key_ in ("arch", "model type", "model params", "file type", "file size", "n_ctx_train", "n_layer",
                     "n_head", "n_head_kv", "n_embd", "n_expert", "n_expert_used", "vocab type", "n_vocab"):
            add(key_, mi.get(key_))
        if s.info.offload:
            add("offloaded", f"{s.info.offload[0]}/{s.info.offload[1]} layers")
        cmd = Text(" ".join(s.cmd) if s.cmd else "(not launched by inari)", style=p.accent2)
        tmpl = (props.get("chat_template") or "").strip().splitlines()
        tmpl_t = Text("\n".join(tmpl[:8]) + ("\n…" if len(tmpl) > 8 else ""), style=p.dim) if tmpl else Text("—", style=p.dim)
        engine = Panel(kv, title=Text(" Engine ", style=p.text), title_align="left", border_style=p.dim,
                       box=box.ROUNDED)
        side = Group(
            Panel(cmd, title=Text(" Command ", style=p.text), title_align="left", border_style=p.dim, box=box.ROUNDED),
            Panel(tmpl_t, title=Text(" Chat template ", style=p.text), title_align="left", border_style=p.dim,
                  box=box.ROUNDED),
        )
        if self.console.size.width >= 120:
            g = Table.grid(expand=True, padding=(0, 1))
            g.add_column(ratio=2)
            g.add_column(ratio=3)
            g.add_row(engine, side)
            return g
        return Group(engine, side)

    def view_hardware(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        hw, info = s.hw, s.info
        cores = []
        for i, v in enumerate(hw.per_cpu):
            row = Table.grid(expand=True)
            row.add_column(width=4, no_wrap=True)
            row.add_column(ratio=1)
            row.add_column(width=5, justify="right", no_wrap=True)
            row.add_row(Text(f"{i:>3}", style=p.dim), Bar(v / 100, p.accent, p.track), Text(f"{v:3.0f}%", style=p.text))
            cores.append(row)
        ncols = 4 if len(cores) > 16 else 2
        cpu_panel = Panel(self.grid(cores, ncols), title=Text(f" CPU {hw.cpu:.0f}% ", style=p.text), title_align="left",
                          border_style=p.dim, box=box.ROUNDED)

        bt = Table(box=box.SIMPLE_HEAD, border_style=p.dim, header_style=f"bold {p.accent}", expand=True)
        for c, j in (("DEVICE", "left"), ("NAME", "left"), ("MODEL", "right"), ("KV", "right"),
                     ("COMPUTE", "right"), ("TOTAL", "right"), ("FREE AT LOAD", "right")):
            bt.add_column(c, justify=j, style=p.text, no_wrap=True)
        devs = set(info.buffers) | set(info.devices)
        for d in sorted(devs):
            b = info.buffers.get(d, {})
            desc, free = info.devices.get(d, ("", 0))
            f = lambda x: f"{x / 1024:.2f} GB" if x else "--"  # noqa: E731
            bt.add_row(d, desc, f(b.get("model", 0)), f(b.get("kv", 0)), f(b.get("compute", 0)),
                       f(sum(b.values())), f(free))
        if not devs:
            bt.add_row(Text("no device info yet (needs a log from an inari-launched server)", style=p.dim))
        for g in hw.gpus:
            bt.add_row(Text("NVML", style=p.accent2), g.name,
                       "", "", "", f"{(g.mem_used or 0) / GB:.2f} GB",
                       f"pcie rx {(g.pcie_rx or 0) / MB:.0f} MB/s")
        mem = Group(
            self.labeled_bar("System RAM", f"{hw.ram_used / GB:.1f} / {hw.ram_total / GB:.1f} GB", pct(hw.ram_used, hw.ram_total)),
            self.labeled_bar("Swap", f"{hw.swap_used / GB:.1f} / {hw.swap_total / GB:.1f} GB", pct(hw.swap_used, hw.swap_total),
                             p.warn),
            self.labeled_bar("llama-server RSS", f"{(hw.proc_rss or 0) / GB:.2f} GB", pct(hw.proc_rss or 0, hw.ram_total),
                             p.accent2),
        )
        return Group(cpu_panel,
                     Panel(bt, title=Text(" Devices & llama.cpp buffers ", style=p.text), title_align="left",
                           border_style=p.dim, box=box.ROUNDED),
                     mem)

    def view_logs(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        if not s.has_logs:
            return Text("Logs are only available for servers started by inari (attach with `inari monitor <profile>`).",
                        style=p.dim)
        lines = s.logs
        if self.filter:
            f = self.filter.lower()
            lines = [ln for ln in lines if f in ln.lower()]
        n = max(height - 1, 1)
        end = max(len(lines) - self.scroll, 0)
        view = lines[max(end - n, 0):end]
        out = Text()
        for ln in view:
            lvl = line_level(ln)
            if lvl == "E":
                st = p.bad
            elif lvl == "W":
                st = p.warn
            elif "slot" in ln or "srv" in ln:
                st = p.accent2
            elif ln.startswith("[inari]") or ln.startswith("$ "):
                st = f"bold {p.accent}"
            else:
                st = p.text
            out.append(ln.expandtabs(4)[: self.console.size.width - 1] + "\n", style=st)
        title = Text(f"Logs  {len(lines):,} lines", style=f"bold {p.accent}")
        if self.scroll:
            title.append(f"  (scrolled {self.scroll} · End to follow)", style=p.warn)
        return Group(title, out)

    def view_connect(self, height: int) -> RenderableType:
        p, s = self.p, self.snap
        base = s.url
        model = s.model or "local"
        blocks = [
            ("OpenAI-compatible endpoint", f"{base}/v1\napi key: anything (e.g. sk-no-key)  ·  model: {model}"),
            ("Web UI", f"{base}/   (press b to open)"),
            ("Environment", f'OPENAI_BASE_URL={base}/v1\nOPENAI_API_KEY=sk-no-key'),
            ("curl", f'curl {base}/v1/chat/completions -H "Content-Type: application/json" \\\n'
                     f'  -d \'{{"messages":[{{"role":"user","content":"hello"}}]}}\''),
            ("Python (openai)", f'from openai import OpenAI\nclient = OpenAI(base_url="{base}/v1", api_key="sk-no-key")\n'
                                f'r = client.chat.completions.create(model="{model}", messages=[{{"role": "user", "content": "hello"}}])\n'
                                'print(r.choices[0].message.content)'),
        ]
        panels = [Panel(Text(body, style=p.accent2), title=Text(f" {title} ", style=p.text), title_align="left",
                        border_style=p.dim, box=box.ROUNDED) for title, body in blocks]
        return Group(*panels)

    def view_help(self) -> RenderableType:
        p = self.p
        t = Table.grid(padding=(0, 3))
        t.add_column(style=f"bold {p.accent}", no_wrap=True)
        t.add_column(style=p.text)
        for a, b in (("1 … 7  ← →  Tab", "switch view"), ("/", "filter (logs: text, requests: status/task id)"),
                     ("Esc", "clear filter / close help"), ("Space", "freeze / unfreeze the display"),
                     ("r", "refresh now"), ("t", "cycle theme (inari · amber · lemon · kitsune)"),
                     ("b", "open the llama.cpp web UI"), ("↑ ↓ PgUp PgDn End", "scroll logs / requests"),
                     ("q", "quit (managed servers: keep serving or stop)"), ("?", "toggle this help")):
            t.add_row(a, b)
        return Panel(t, title=Text(" Help ", style=f"bold {p.accent}"), title_align="left", border_style=p.accent,
                     box=box.ROUNDED, padding=(1, 2))
