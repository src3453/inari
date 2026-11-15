"""Bare ``inari``: a full-screen main menu — pick a profile to serve, or start the setup wizard."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

from inari import server
from inari import theme as th
from inari.config import Config
from inari.hubtui import window
from inari.inari_logo import banner
from inari.keys import KeyReader
from inari.modal import ConfirmModal
from inari.profiles import Profile, list_profiles
from inari.themepick import ThemePicker


@dataclass
class Row:
    kind: str  # "profile" | "router" | "new" | "hub" | "demo" | "quit"
    name: str = ""
    model: str = ""
    listen: str = ""
    running: bool = False
    missing: bool = False
    error: str = ""


ACTIONS = [("router", "⇄ Router", "serve every profile from one port, loaded on demand"),
           ("new", "+ New profile", "guided setup wizard"),
           ("hub", "↓ Hub", "search Hugging Face and download GGUFs"),
           ("demo", "▷ Demo", "try the monitor with a simulated server"),
           ("quit", "× Quit", "")]


class Menu:
    """``run()`` returns ``(action, profile_name)``:
    serve · edit · stop · new · hub · demo · quit. Deleting a profile happens in place."""

    def __init__(self, cfg: Config, console: Console, *, select: str = "", flash: str = "") -> None:
        self.cfg = cfg
        self.console = console
        self.p = th.get(cfg.theme)
        self.picker = ThemePicker()
        self.modal = ConfirmModal()
        self.rows: list[Row] = []
        self.sel = 0
        self.confirm: tuple[str, str, str] | None = None  # (question, action, name)
        self.flash = (flash, time.monotonic() + 4) if flash else ("", 0.0)
        self.refresh()
        if select:
            self.sel = next((i for i, r in enumerate(self.rows) if r.name == select), self.sel)
        elif not self.profiles():
            self.sel = next(i for i, r in enumerate(self.rows) if r.kind == "new")

    def profiles(self) -> list[Row]:
        return [r for r in self.rows if r.kind == "profile"]

    def refresh(self) -> None:
        active = server.active_runs()
        runs = {r.name for r in active}
        self.router_url = next((r.url for r in active if r.name == "@router"), "")
        rows: list[Row] = []
        for n in list_profiles():
            try:
                p = Profile.load(n)
            except Exception as e:
                rows.append(Row("profile", n, error=str(e)))
                continue
            try:
                mp = p.model_path(self.cfg)
                missing = bool(mp) and not mp.exists()
            except Exception:
                missing = True
            rows.append(Row("profile", n, model=p.model.replace("\\", "/").rsplit("/", 1)[-1], listen=f"{p.host}:{p.port}", running=n in runs,
                            missing=missing))
        rows += [Row(k, running=k == "router" and bool(self.router_url)) for k, _, _ in ACTIONS]
        self.rows = rows
        self.has_bin = bool(self.cfg.server_binary())
        self.sel = min(self.sel, len(rows) - 1)

    # ───────── loop ─────────
    def run(self) -> tuple[str, str]:
        with KeyReader() as keys, Live(console=self.console, screen=True, auto_refresh=False,
                                       redirect_stdout=False, redirect_stderr=False) as live:
            last = poll = 0.0
            while True:
                pressed = keys.drain()
                for key in pressed:
                    res = self.handle(key)
                    if res:
                        return res
                now = time.monotonic()
                if now - poll >= 2.0:  # pick up servers started / stopped elsewhere
                    self.refresh()
                    poll = now
                if pressed or now - last >= 0.5:
                    live.update(self.render(), refresh=True)
                    last = now
                time.sleep(0.03)

    def say(self, msg: str) -> None:
        self.flash = (msg, time.monotonic() + 3.5)

    def handle(self, key: str) -> tuple[str, str] | None:
        if self.picker.active:
            if self.picker.handle(key) == "pick" and self.picker.name != self.cfg.theme:
                self.cfg.theme = self.picker.name
                self.cfg.save()
                self.console.push_theme(th.rich_theme(th.get(self.cfg.theme)))  # [accent] etc. in later prints
                self.say(f"theme: {self.cfg.theme}")
            self.p = th.get(self.picker.name)
            return None
        if self.modal.active:
            self.modal.handle(key)
            return None
        if key in ("ctrl+c", "q", "Q"):
            return ("quit", "")
        if self.confirm:
            _, act, name = self.confirm
            self.confirm = None
            return (act, name) if key in ("y", "Y", "enter") else None
        n = len(self.rows)
        step = {"up": -1, "down": 1, "k": -1, "j": 1, "pgup": -10, "pgdn": 10}.get(key)
        if step is not None:
            self.sel = max(0, min(n - 1, self.sel + step))
            return None
        if key == "home":
            self.sel = 0
            return None
        if key == "end":
            self.sel = n - 1
            return None
        if key in ("n", "s"):
            return ("new", "")
        if key == "h":
            return ("hub", "")
        if key == "r":
            return ("router", "")
        if key == "t":
            self.picker.open(self.cfg.theme)
            return None
        if key.isdigit() and key != "0":
            i = int(key) - 1
            if i < len(self.profiles()):
                self.sel = i
                return self.enter()
            return None
        row = self.rows[self.sel]
        if key == "enter":
            return self.enter()
        if row.kind != "profile":
            return None
        if key == "e":
            return ("edit", row.name)
        if key in ("d", "delete"):
            self.ask_delete(row)
            return None
        if key == "x":
            if not row.running:
                self.say(f"{row.name} is not running")
                return None
            self.confirm = (f"Stop {row.name}?", "stop", row.name)
        return None

    def ask_delete(self, row: Row) -> None:
        if row.running:
            self.say(f"{row.name} is serving — stop it first (x)")
            return
        path = Profile.path_for(row.name)
        p = self.p
        lines = [Text.assemble(("Delete profile ", p.text), (row.name, f"bold {p.accent}"), ("?", p.text)),
                 Text("")]
        if row.model:
            lines.append(Text.assemble(("model    ", p.dim), (row.model, p.text)))
        lines.append(Text.assemble(("file     ", p.dim), (str(path), p.text)))
        lines += [Text(""), Text("The model file itself is kept. This can't be undone.", style=p.warn)]

        def do() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                self.say(f"couldn't delete {row.name}: {e}")
                return
            self.refresh()
            self.say(f"✔ deleted profile {row.name}")

        self.modal.open("Delete profile", lines, do)

    def enter(self) -> tuple[str, str] | None:
        row = self.rows[self.sel]
        if row.kind != "profile":
            return (row.kind, "")
        if row.error:
            self.say(f"can't read {row.name}: {row.error}")
            return None
        if row.missing and not row.running:
            self.say(f"model file for {row.name} is missing — e to pick another, h to download")
            return None
        return ("serve", row.name)

    # ───────── render ─────────
    def render(self) -> RenderableType:
        p = self.p
        w, h = self.console.size
        body_h = max(h - 4 - 1 - 1 - 2, 5)
        root = Layout()
        root.split_column(
            Layout(self.header(), size=4),
            Layout(Text("─" * w, style=p.dim), size=1),
            Layout(self.picker.render(p) if self.picker.active else
                   self.modal.render(p) if self.modal.active else self.body(body_h), size=body_h),
            Layout(self.status(), size=1),
            Layout(self.hints(), size=2),
        )
        return root

    def header(self) -> RenderableType:
        return banner(self.p, self.console)

    def body(self, h: int) -> RenderableType:
        p = self.p
        profs = [(i, r) for i, r in enumerate(self.rows) if r.kind == "profile"]
        acts = [(i, r) for i, r in enumerate(self.rows) if r.kind != "profile"]
        name_w = min(max([len(r.name) for _, r in profs] + [13]), 30)

        pt = Table.grid(padding=(0, 2), expand=True)
        pt.add_column(width=2, no_wrap=True)
        pt.add_column(width=name_w, no_wrap=True, overflow="ellipsis")
        pt.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
        pt.add_column(no_wrap=True)
        pt.add_column(no_wrap=True)
        rows = max(h - 2 - len(acts) - 1, 1)  # heading, blank, actions
        cur_p = next((j for j, (i, _) in enumerate(profs) if i == self.sel), 0)
        a, b = window(cur_p, len(profs), rows)
        for i, r in profs[a:b]:
            on = i == self.sel
            name = Text(r.name, style=f"bold {p.accent}" if on else p.text)
            if r.error:
                model, listen, st = Text(r.error, style=p.bad), Text(""), Text("")
            else:
                model = Text(r.model or "—", style=p.bad if r.missing else (p.text if on else p.dim))
                listen = Text(r.listen, style=p.dim)
                st = (Text("● serving", style=p.good) if r.running else
                      Text("model missing", style=p.bad) if r.missing else Text("stopped", style=p.dim))
            cur = Text("▶" if on else (str(i + 1) if i < 9 else ""), style=f"bold {p.accent}" if on else p.dim)
            pt.add_row(cur, name, model, listen, st)

        head = Text("  Profiles", style=f"bold {p.accent}")
        if profs:
            head.append(f"   {len(profs)}" + (f"  ·  showing {a + 1}–{b}" if b - a < len(profs) else ""), style=p.dim)
        else:
            head.append("   none yet — choose + New profile to run setup", style=p.accent2)

        at = Table.grid(padding=(0, 2))
        at.add_column(width=2, no_wrap=True)
        at.add_column(width=name_w, no_wrap=True)
        at.add_column(no_wrap=True, overflow="ellipsis")
        for i, r in acts:
            on = i == self.sel
            label, desc = next((lb, d) for k, lb, d in ACTIONS if k == r.kind)
            dt = Text(desc, style=p.dim)
            if r.running:
                dt = Text("● serving", style=p.good) + Text(f" at {self.router_url} — Enter opens the monitor",
                                                            style=p.dim)
            at.add_row(Text("▶" if on else "", style=f"bold {p.accent}"),
                       Text(label, style=f"bold {p.accent}" if on else p.accent2), dt)
        return Group(head, pt, Text(""), at) if profs else Group(head, Text(""), at)

    def status(self) -> RenderableType:
        p = self.p
        if self.confirm:
            return Text(f" {self.confirm[0]}  y / n", style=f"bold {p.warn}")
        msg, until = self.flash
        if msg and time.monotonic() < until:
            return Text(" " + msg, style=p.accent2)
        if not self.has_bin:
            return Text(" llama-server not found — + New profile walks you through setting it up", style=p.warn)
        return Text(" inari --help lists every command", style=p.dim)

    def hints(self) -> RenderableType:
        p = self.p
        row = self.rows[self.sel]
        items = [("↑↓", "choose"), ("Enter", "serve" if row.kind == "profile" and not row.running
                                     else ("monitor" if row.kind == "profile" else "open"))]
        if row.kind == "profile":
            items.append(("e", "edit (setup)"))
            if row.running:
                items.append(("x", "stop"))
            else:
                items.append(("d", "delete"))
        items += [("r", "router"), ("n", "new profile"), ("h", "hub"), ("t", "theme"), ("q", "quit")]
        t = Text()
        for key, label in items:
            t.append(key, style=f"bold {p.accent}")
            t.append(f" {label}   ", style=p.dim)
        return t
