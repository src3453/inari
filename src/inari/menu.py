"""Bare ``inari``: a full-screen main menu — pick a profile to serve, or start the setup wizard."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

from rich import box
from rich.align import Align
from rich.panel import Panel

from inari import server
from inari import theme as th
from inari.config import Config, config_path
from inari.hubtui import window
from inari.inari_logo import banner
from inari.keys import KeyReader
from inari.modal import ChoiceModal, ConfirmModal
from inari.profiles import Profile, list_profiles
from inari.themepick import ThemePicker


@dataclass
class Row:
    kind: str  # "profile" | "router" | "new" | "import" | "hub" | "demo" | "quit"
    name: str = ""
    model: str = ""
    listen: str = ""
    running: bool = False
    missing: bool = False
    error: str = ""


ACTIONS = [("router", "⇄ Router", "serve every profile from one port, loaded on demand"),
           ("new", "+ Setup & models", "new profile wizard · pick, download or delete local models"),
           ("import", "⇲ Import command", "paste a llama-server command line → profile"),
           ("hub", "↓ Hub", "search Hugging Face and download GGUFs"),
           ("demo", "▷ Demo", "try the monitor with a simulated server"),
           ("settings", "⚙ Settings", "theme, download backend, Hugging Face token, llama-server path"),
           ("quit", "× Quit", "")]
TUNE_OPTIONS = [("balanced", "b", "Balanced", "quick · a typical chat request"),
                ("gen", "g", "Generation", "quick · tokens/s while replying"),
                ("prompt", "p", "Prompt", "quick · tokens/s reading the prompt"),
                ("thorough", "t", "Thorough", "balanced · up to 24 runs, ×2 each")]
TIPS = ["inari --help lists every command",
        "press 1–9 to jump straight to a profile",
        "t on a profile auto-tunes GPU layers, batch and threads for speed",
        "c on a profile opens a terminal chat — type / for its commands",
        "in chat, /think shows or hides the model's reasoning",
        "in chat, /regen answers your last message again",
        "in chat, /system sets a system prompt for a fresh conversation",
        "⇄ Router serves every profile from one port and loads models on demand",
        "⇲ Import turns an existing llama-server command line into a profile",
        "inari pull repo:QUANT downloads a GGUF without opening the hub",
        "inari profile set NAME ctx-size=32768 changes a profile from the shell",
        "inari ps lists running servers; inari stop NAME stops one",
        "x stops a running profile — the monitor isn't needed to keep it serving",
        "the http download backend resumes partial downloads (o → Settings)",
        "o → Settings → Theme previews every theme live",
        "o → Settings → Hugging Face token unlocks gated models in the hub",
        "inari demo shows the monitor with a simulated server",
        "learn about helping foxes in UK! read more: https://foxproject.org.uk/ (personal recommendation, not affiliated with inari)"]
TIP_SECONDS = 20.0  # how long one tip stays before the next one


class Settings:
    """The Settings panel of the main menu.
    Rows: theme (opens the picker) · download backend · Hugging Face token (inline edit) · info."""

    ROWS = ("theme", "backend", "token", "server", "config")
    EDITABLE = 3  # the first N rows can be changed here

    def __init__(self) -> None:
        self.active = False
        self.sel = 0
        self.editing = False  # typing a new Hugging Face token
        self.buf = ""

    @staticmethod
    def mask(token: str) -> str:
        return f"{token[:3]}{'•' * 8}{token[-4:]}" if len(token) > 10 else "•" * len(token)

    def token_value(self, cfg: Config) -> tuple[str, str]:
        if self.editing:
            return (self.mask(self.buf) if self.buf else "") + "▏", "Enter: save (empty clears)  ·  Esc: cancel"
        if cfg.hf_token:
            return self.mask(cfg.hf_token), "Enter: replace  ·  Del: clear  ·  for gated / private repos"
        if os.environ.get("HF_TOKEN"):
            return "not set (using HF_TOKEN env)", "Enter: paste a token  ·  huggingface.co/settings/tokens"
        return "not set", "Enter: paste a token  ·  huggingface.co/settings/tokens"

    def render(self, p: th.Palette, cfg: Config) -> RenderableType:
        from inari.hub import BACKENDS

        t = Table.grid(padding=(0, 2))
        t.add_column(width=2, no_wrap=True)
        t.add_column(no_wrap=True, min_width=18)
        t.add_column(no_wrap=True, overflow="ellipsis")
        backend = cfg.download_backend if cfg.download_backend in BACKENDS else "auto"
        token, token_note = self.token_value(cfg)
        rows = [("Theme", cfg.theme, "Enter: choose with a live preview"),
                ("Download backend", backend, "←→: auto · xet · http (http resumes partials)"),
                ("Hugging Face token", token, token_note),
                ("llama-server", cfg.server_binary() or "not found", "change it in + Setup & models (step 1)"),
                ("Config file", str(config_path()), "all settings: inari config show / set")]
        for i, (label, val, note) in enumerate(rows):
            on = i == self.sel
            editable = i < self.EDITABLE
            v = Text()
            v.append("‹ " if on and i == 1 else "  " if i == 1 else "", style=p.accent)
            v.append(val, style=f"bold {p.accent2}" if on and editable else (p.text if editable else p.dim))
            v.append(" ›" if on and i == 1 else "", style=p.accent)
            t.add_row(Text("▶" if on else "", style=f"bold {p.accent}"),
                      Text(label, style=f"bold {p.accent}" if on else p.text), v)
            t.add_row("", "", Text(note, style=p.dim))
        if self.editing:
            hint = Text.assemble(("type / paste", f"bold {p.accent}"), (" the token (hf_…)   ", p.dim),
                                 ("Enter", f"bold {p.accent}"), (" save   ", p.dim),
                                 ("Esc", f"bold {p.accent}"), (" cancel", p.dim))
        else:
            hint = Text.assemble(("↑↓", f"bold {p.accent}"), (" choose   ", p.dim), ("Enter", f"bold {p.accent}"),
                                 (" change   ", p.dim), ("Esc", f"bold {p.accent}"), (" back", p.dim))
        body = Table.grid()
        body.add_row(t)
        body.add_row(Text(""))
        body.add_row(hint)
        return Align.center(Panel(body, title=Text(" Settings ", style=f"bold {p.accent}"), title_align="left",
                                  border_style=p.accent, box=box.ROUNDED, padding=(1, 3), width=88),
                            vertical="middle")


class Menu:
    """``run()`` returns ``(action, profile_name)``:
    serve · chat · tune:<goal|thorough> · edit · stop · new · hub · demo · quit.
    Deleting a profile and the Settings panel (theme, …) are handled in place."""

    last_launch = "serve"  # what Enter on a profile chose last time — focused first in the chooser

    def __init__(self, cfg: Config, console: Console, *, select: str = "", flash: str = "") -> None:
        self.cfg = cfg
        self.console = console
        self.p = th.get(cfg.theme)
        self.picker = ThemePicker()
        self.modal = ConfirmModal()
        self.chooser = ChoiceModal()
        self.tune_chooser = ChoiceModal()
        self.settings = Settings()
        self.rows: list[Row] = []
        self.sel = 0
        self.confirm: tuple[str, str, str] | None = None  # (question, action, name)
        self.flash = (flash, time.monotonic() + 4) if flash else ("", 0.0)
        self.tips = random.sample(TIPS, len(TIPS))  # shuffled once; rotates every TIP_SECONDS
        self.tips_t0 = time.monotonic()
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
        if self.settings.active:
            return self.key_settings(key)
        if self.tune_chooser.active:
            got = self.tune_chooser.handle(key)
            if got and got != "no":
                return ("tune:" + got, self.rows[self.sel].name)
            return None
        if self.modal.active:
            self.modal.handle(key)
            return None
        if self.chooser.active:
            got = self.chooser.handle(key)
            if got in ("serve", "chat"):
                Menu.last_launch = got
                return (got, self.rows[self.sel].name)
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
        if key == "i":
            return ("import", "")
        if key == "r":
            return ("router", "")
        if key == "o":
            self.open_settings()
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
        if key == "c":
            return self.launch("chat")
        if key == "t":
            self.ask_tune(row)
            return None
        if key in ("d", "delete"):
            self.ask_delete(row)
            return None
        if key == "x":
            if not row.running:
                self.say(f"{row.name} is not running")
                return None
            self.confirm = (f"Stop {row.name}?", "stop", row.name)
        return None

    def open_settings(self) -> None:
        self.settings.active = True
        self.settings.sel = 0
        self.settings.editing = False

    def key_token(self, key: str) -> None:
        """Typing a Hugging Face token in Settings — every key goes to the field."""
        s = self.settings
        if key == "esc":
            s.editing = False
        elif key == "enter":
            s.editing = False
            tok = s.buf.strip()
            if tok == self.cfg.hf_token:
                return
            self.cfg.hf_token = tok
            self.cfg.save()
            self.say("✔ Hugging Face token saved" if tok else "Hugging Face token cleared")
        elif key == "backspace":
            s.buf = s.buf[:-1]
        elif key == "ctrl+c":
            s.buf = ""
        elif len(key) == 1 and key.isprintable() and not key.isspace():
            s.buf += key

    def key_settings(self, key: str) -> tuple[str, str] | None:
        from inari.hub import BACKENDS

        s = self.settings
        n = len(Settings.ROWS)
        if s.editing:
            self.key_token(key)
            return None
        if key in ("esc", "q", "Q", "o", "b"):
            s.active = False
        elif key == "ctrl+c":
            return ("quit", "")
        elif key in ("up", "k"):
            s.sel = (s.sel - 1) % n
        elif key in ("down", "j", "tab"):
            s.sel = (s.sel + 1) % n
        elif Settings.ROWS[s.sel] == "theme" and key in ("enter", " ", "right", "l"):
            self.picker.open(self.cfg.theme)
        elif Settings.ROWS[s.sel] == "backend" and key in ("enter", " ", "right", "left", "l", "h"):
            cur = self.cfg.download_backend if self.cfg.download_backend in BACKENDS else "auto"
            d = -1 if key in ("left", "h") else 1
            self.cfg.download_backend = BACKENDS[(BACKENDS.index(cur) + d) % len(BACKENDS)]
            self.cfg.save()
            self.say(f"download backend: {self.cfg.download_backend}")
        elif Settings.ROWS[s.sel] == "token" and key in ("enter", " ", "right", "l"):
            s.editing = True
            s.buf = ""
        elif Settings.ROWS[s.sel] == "token" and key in ("delete", "backspace") and self.cfg.hf_token:
            self.cfg.hf_token = ""
            self.cfg.save()
            self.say("Hugging Face token cleared")
        return None

    def ask_tune(self, row: Row) -> None:
        if row.error:
            self.say(f"can't read {row.name}: {row.error}")
        elif row.running:
            self.say(f"{row.name} is serving — stop it first (x) so the GPU memory is free")
        elif row.missing:
            self.say(f"model file for {row.name} is missing — e to pick another, h to download")
        elif not self.has_bin:
            self.say("llama-server not found — + Setup & models walks you through setting it up")
        else:
            p = self.p
            lines = [Text.assemble(("Auto-tune ", p.text), (row.name, f"bold {p.accent}")),
                     Text("Starts llama-server several times with candidate GPU-layer / MoE / batch / thread "
                          "settings and keeps the fastest. Asks before saving.", style=p.dim)]
            self.tune_chooser.open("Tune", lines, TUNE_OPTIONS)

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
        if row.kind == "settings":
            self.open_settings()
            return None
        if row.kind != "profile":
            return (row.kind, "")
        if row.error:
            self.say(f"can't read {row.name}: {row.error}")
            return None
        if row.missing and not row.running:
            self.say(f"model file for {row.name} is missing — e to pick another, h to download")
            return None
        p = self.p
        lines = [Text.assemble(("Start ", p.text), (row.name, f"bold {p.accent}"),
                               ("  ● serving" if row.running else "", p.good)),
                 Text.assemble(("model  ", p.dim), (row.model or "—", p.text))]
        self.chooser.open("Start", lines, [
            ("serve", "s", "Monitor" if row.running else "Serve", "re-attach the monitor" if row.running
             else "start llama-server + monitor (keeps serving)"),
            ("chat", "c", "Chat (cli)", "chat in the terminal, like llama-cli"
             + ("" if row.running else " — stops it when you leave"))], default=Menu.last_launch)
        return None

    def launch(self, what: str) -> tuple[str, str] | None:
        row = self.rows[self.sel]
        if row.error:
            self.say(f"can't read {row.name}: {row.error}")
            return None
        if row.missing and not row.running:
            self.say(f"model file for {row.name} is missing — e to pick another, h to download")
            return None
        return (what, row.name)

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
                   self.settings.render(p, self.cfg) if self.settings.active else
                   self.tune_chooser.render(p) if self.tune_chooser.active else
                   self.modal.render(p) if self.modal.active else
                   self.chooser.render(p) if self.chooser.active else self.body(body_h), size=body_h),
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
            head.append("   none yet — choose + Setup & models to make one", style=p.accent2)

        at = Table.grid(padding=(0, 2))
        at.add_column(width=2, no_wrap=True)
        at.add_column(width=max(name_w, max(len(lb) for _, lb, _ in ACTIONS)), no_wrap=True)
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
            return Text(" llama-server not found — + Setup & models walks you through setting it up", style=p.warn)
        tip = self.tips[int((time.monotonic() - self.tips_t0) // TIP_SECONDS) % len(self.tips)]
        return Text.assemble((" Tip ", f"bold {p.accent}"), (tip, p.dim))

    def hints(self) -> RenderableType:
        p = self.p
        row = self.rows[self.sel]
        items = [("↑↓", "choose"), ("Enter", "serve / chat" if row.kind == "profile" and not row.running
                                     else ("monitor / chat" if row.kind == "profile" else "open"))]
        if row.kind == "profile":
            items += [("c", "chat"), ("t", "tune"), ("e", "edit (setup)")]
            if row.running:
                items.append(("x", "stop"))
            else:
                items.append(("d", "delete"))
        items += [("r", "router"), ("n", "setup & models"), ("i", "import"), ("h", "hub"), ("o", "settings"), ("q", "quit")]
        t = Text()
        for key, label in items:
            t.append(key, style=f"bold {p.accent}")
            t.append(f" {label}   ", style=p.dim)
        return t
