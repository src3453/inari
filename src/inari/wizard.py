"""``inari setup`` ("Setup & models"): a full-screen wizard that builds a launch profile step by step.
Its model step doubles as the local model manager (pick, download from the Hub, delete from disk).

Steps: llama-server → model → name → context & memory → GPU → chat → network →
advanced (every llama-server flag, with Default / Tuned / Yours columns) → review.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inari import argspec, hub, tuning
from inari import theme as th
from inari.argspec import ArgSpec, Catalog
from inari.config import Config
from inari.gguf import ModelInfo, read_info
from inari.hubtui import window
from inari.keys import KeyReader
from inari.modal import ConfirmModal
from inari.profiles import NAME_RE, Profile, list_profiles, parse_value
from inari.tui import Bar

STEPS = [("server", "llama-server"), ("model", "Model"), ("name", "Name"), ("context", "Context & memory"),
         ("gpu", "GPU offload"), ("chat", "Chat"), ("network", "Network"), ("advanced", "Advanced"),
         ("review", "Review")]
DEFAULT = object()  # option meaning "leave it to llama.cpp"
CMD_ROWS = 4


@dataclass
class Opt:
    value: Any
    label: str = ""


@dataclass
class Field:
    key: str  # canonical llama-server key, or @name / @description / @host / @port
    label: str
    options: list[Opt] = field(default_factory=list)
    custom: bool = False  # Enter types a value
    note: str = ""


@dataclass
class Input:
    label: str
    buf: str
    submit: Callable[[str], str | None]  # returns an error message to keep the prompt open
    live: Callable[[str], None] | None = None  # called on every keystroke (filter)


@dataclass
class Result:
    name: str
    path: Path
    serve: bool = False


def show(v: Any) -> str:
    if v is True:
        return "on"
    if v is False:
        return "off"
    if v is None:
        return "default"
    if isinstance(v, list):
        return ", ".join(show(x) for x in v)
    return str(v)


def copy_to_clipboard(text: str) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd, input=text.encode(), check=True)
                return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def free_port(skip: str = "") -> int:
    used = set()
    for n in list_profiles():
        if n == skip:
            continue
        try:
            used.add(Profile.load(n).port)
        except Exception:
            pass
    port = 8080
    while port in used:
        port += 1
    return port


class Wizard:
    def __init__(self, cfg: Config, console: Console, *, name: str = "", theme_name: str = "kitsune",
                 start: str = "", seed: Profile | None = None, origin: str = "") -> None:
        self.cfg = cfg
        self.console = console
        self.theme_name = theme_name
        self.p = th.get(theme_name)

        self.binary = cfg.server_binary()
        self.cat: Catalog = argspec.load(cfg, self.binary)
        self.hw = tuning.detect(self.binary)
        self.models = self._scan_models()

        # the profile being built
        self.editing = ""  # name of the profile we loaded, if any
        self.name, self.description = "", ""
        self.name_touched = False
        self.model, self.model_path, self.model_size = "", None, 0
        self.info: ModelInfo | None = None
        self.host, self.port = "127.0.0.1", free_port()
        self.server_bin, self.extra, self.env = "", [], {}
        self.mine: dict[str, Any] = {}  # your values (None = force llama.cpp's default)
        self.tuned: dict[str, Any] = {}
        self.why: dict[str, str] = {}
        self.apply_tuned = True

        self.origin = origin  # e.g. "imported from a command" — shown in the header

        if seed is not None:  # a profile built elsewhere (inari import), not saved yet
            self._apply(seed)
            self.editing = ""
            self.dirty = True
        elif name and Profile.path_for(name).exists():
            self._load(name)
        elif name:
            self.name, self.name_touched = name, True

        self.step = next((i for i, (k, _) in enumerate(STEPS) if k == start), 0)
        if not start:
            self.step = 0 if not self.binary else (1 if not self.model else 2)
        self.sel = 0  # row in the current step
        self.msel = next((i for i, m in enumerate(self.models) if m.ref == self.model), 0)
        self.asel = 0
        self.filter = ""
        self.exp_only = False
        self.cmd_scroll = 0
        self.input: Input | None = None
        self.confirm: tuple[str, Callable[[], bool]] | None = None
        self.modal = ConfirmModal()
        self.flash: tuple[str, float] | None = None
        self.result: Result | None = None
        self.exit_reason = ""
        self.dirty = seed is not None

    # ───────── model / profile state ─────────
    def _scan_models(self) -> list[hub.LocalModel]:
        return [m for m in hub.local_models(self.cfg) if "mmproj" not in m.path.name.lower()]

    def _load(self, name: str) -> None:
        self._apply(Profile.load(name))
        self.editing = name

    def _apply(self, p: Profile) -> None:
        self.name = p.name
        self.name_touched = True
        self.description = p.description
        self.host, self.port = p.host, p.port
        self.server_bin, self.extra, self.env = p.server_bin, list(p.extra), dict(p.env)
        self.mine = argspec.from_args(p.args, self.cat)
        self.apply_tuned = False  # an existing profile already says what it wants
        if p.model:
            path = p.model_path(self.cfg)
            size = next((m.size for m in self.models if m.ref == p.model), 0)
            if not size and path and path.exists():
                size = path.stat().st_size
            self.set_model(p.model, path, size, keep_mmproj=True)

    def set_model(self, ref: str, path: Path | None, size: int, *, keep_mmproj: bool = False) -> None:
        self.model, self.model_path, self.model_size = ref, path, size
        self.info = read_info(path) if path and path.exists() else None
        self.tuned, self.why = tuning.tune(self.cat, self.hw, self.info, size)
        if not keep_mmproj:
            proj = self.mmproj_for(path)
            if proj and "mmproj" in self.cat:
                self.mine["mmproj"] = hub.model_ref(self.cfg, proj)
            else:
                self.mine.pop("mmproj", None)
        if not self.name_touched and path:
            self.name = path.stem.split("-0000")[0].lower().replace(" ", "-")

    @staticmethod
    def mmproj_for(path: Path | None) -> Path | None:
        if not path or not path.parent.exists():
            return None
        cands = sorted(path.parent.glob("*mmproj*.gguf"))
        return next((c for c in cands if "f16" in c.name.lower()), cands[0] if cands else None)

    def value(self, key: str) -> Any:
        """Effective value, or None when llama.cpp's default applies."""
        if key in self.mine:
            return self.mine[key]
        if self.apply_tuned and key in self.tuned:
            return self.tuned[key]
        return None

    def source(self, key: str) -> str:
        if key in self.mine:
            return "yours"
        if self.apply_tuned and key in self.tuned:
            return "tuned"
        return "default"

    def effective(self) -> dict[str, Any]:
        out = dict(self.tuned) if self.apply_tuned else {}
        out.update(self.mine)
        return {k: v for k, v in out.items() if v is not None}

    def set_value(self, key: str, v: Any) -> None:
        """Set a setting; picking the tuned value or the default drops the override."""
        self.dirty = True
        if v is DEFAULT:
            if self.apply_tuned and key in self.tuned:
                self.mine[key] = None
            else:
                self.mine.pop(key, None)
        elif self.apply_tuned and key in self.tuned and self.tuned[key] == v:
            self.mine.pop(key, None)
        else:
            self.mine[key] = v

    def reset(self, key: str) -> None:
        self.dirty = True
        self.mine.pop(key, None)

    def profile(self) -> Profile:
        return Profile(name=self.name or "new", model=self.model, description=self.description, host=self.host,
                       port=self.port, server_bin=self.server_bin, args=argspec.to_args(self.effective(), self.cat),
                       extra=list(self.extra), env=dict(self.env))

    def command(self) -> str:
        try:
            cmd = self.profile().build_command(self.cfg, binary=self.cfg.server_binary(self.server_bin)
                                               or "llama-server")
        except Exception as e:  # noqa: BLE001
            return f"(cannot build the command: {e})"
        return subprocess.list2cmdline(cmd)

    # ───────── guided steps ─────────
    def _opts(self, key: str, values: list[Any], fmt: Callable[[Any], str] = show) -> list[Opt]:
        spec = self.cat.get(key)
        d = f"default ({spec.default})" if spec and spec.default else "default"
        out = [Opt(DEFAULT, d)]
        seen = set()
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None):
            values = sorted(v for v in values if v is not None)
        for v in values:
            if v in seen or v is None:
                continue
            seen.add(v)
            out.append(Opt(v, fmt(v)))
        return out

    def fields(self) -> list[Field]:
        key = STEPS[self.step][0]
        c, info = self.cat, self.info
        f: list[Field] = []

        def add(k: str, label: str, values: list[Any] | None = None, *, custom: bool = False,
                fmt: Callable[[Any], str] = show, note: str = "") -> None:
            spec = c.get(k)
            if spec is None:
                return
            if values is None:
                values = [True, False] if spec.kind == "toggle" else [True] if spec.kind == "flag" else spec.choices
            f.append(Field(k, label, self._opts(k, values, fmt), custom, note or spec.help))

        if key == "name":
            f.append(Field("@name", "Profile name", custom=True,
                           note="letters, digits, . _ - · saved to ~/.inari/profiles/<name>.toml"))
            f.append(Field("@description", "Description", custom=True, note="shown in inari profile ls"))
        elif key == "context":
            steps = [s for s in tuning.CTX_STEPS if not info or not info.ctx_train or s <= info.ctx_train]
            if info and info.ctx_train:
                steps.append(info.ctx_train)
            add("ctx-size", "Context length", [self.tuned.get("ctx-size")] + steps, custom=True,
                fmt=lambda v: f"{v:,}" if isinstance(v, int) else show(v),
                note="how many tokens the model can see at once — more costs KV-cache memory")
            kv = ["f16", "q8_0", "q4_0"]
            add("cache-type-k", "KV cache type (K)", kv, note="q8_0 halves KV memory with little quality loss")
            add("cache-type-v", "KV cache type (V)", kv, note="quantized V needs flash attention")
            add("flash-attn", "Flash attention")
            add("parallel", "Parallel slots", [1, 2, 4, 8], custom=True,
                note="requests served at once — the context is shared between slots")
        elif key == "gpu":
            add("fit", "Auto-fit to VRAM")
            ngl = c.get("n-gpu-layers")
            all_v = "all" if ngl and ngl.default == "auto" else 999
            add("n-gpu-layers", "GPU layers", [self.tuned.get("n-gpu-layers"), all_v, 0], custom=True,
                note="layers kept in VRAM: all = full offload, 0 = CPU only")
            if info is None or info.moe:
                layers = info.layers if info else 0
                add("n-cpu-moe", "MoE layers on CPU",
                    [self.tuned.get("n-cpu-moe")] + ([layers // 4, layers // 2, layers] if layers else []),
                    custom=True, note="keep the experts of the first N layers in RAM (frees VRAM for big MoE)")
                add("cpu-moe", "All MoE experts on CPU")
            add("mmap", "Memory-map the model")
            add("mlock", "Lock model in RAM")
            add("threads", "CPU threads", [self.hw.cores, max(self.hw.cores // 2, 1)] if self.hw.cores else [],
                custom=True)
        elif key == "chat":
            add("jinja", "Jinja chat template")
            add("reasoning", "Reasoning / thinking")
            add("reasoning-budget", "Thinking budget", [-1, 0, 1024, 4096, 16384], custom=True)
            add("temperature", "Temperature", [0.2, 0.6, 0.7, 0.8, 1.0], custom=True)
            add("top-p", "Top-p", [0.8, 0.9, 0.95, 1.0], custom=True)
            add("top-k", "Top-k", [20, 40, 64, 0], custom=True)
            add("min-p", "Min-p", [0.0, 0.01, 0.05], custom=True)
            add("repeat-penalty", "Repeat penalty", [1.0, 1.05, 1.1], custom=True)
        elif key == "network":
            f.append(Field("@host", "Listen on", [Opt("127.0.0.1", "127.0.0.1 — this PC only"),
                                                  Opt("0.0.0.0", "0.0.0.0 — whole network")], custom=True,
                           note="0.0.0.0 lets other machines connect — set an API key too"))
            f.append(Field("@port", "Port", [Opt(p, str(p)) for p in sorted({self.port, free_port(self.editing),
                                                                             8080, 8081, 8000})], custom=True))
            add("api-key", "API key", [], custom=True, note="clients must send it as a Bearer token")
            add("alias", "Model name (API)", [], custom=True, note="the model id clients see in /v1/models")
            add("webui", "Web UI")
        return f

    def field_value(self, fd: Field) -> tuple[Any, str]:
        """(value to display, source)."""
        if fd.key == "@name":
            return self.name, "yours"
        if fd.key == "@description":
            return self.description, "yours"
        if fd.key == "@host":
            return self.host, "yours"
        if fd.key == "@port":
            return self.port, "yours"
        return self.value(fd.key), self.source(fd.key)

    def set_field(self, fd: Field, v: Any) -> None:
        self.dirty = True
        if fd.key == "@name":
            self.name, self.name_touched = str(v), True
        elif fd.key == "@description":
            self.description = str(v)
        elif fd.key == "@host":
            self.host = str(v)
        elif fd.key == "@port":
            self.port = int(v)
        else:
            self.set_value(fd.key, v)

    def cycle(self, fd: Field, d: int) -> None:
        if not fd.options:
            return
        cur, src = self.field_value(fd)
        if src == "default" or cur is None:
            i = 0
        else:
            i = next((n for n, o in enumerate(fd.options) if o.value == cur), -1)
        self.set_field(fd, fd.options[(i + d) % len(fd.options)].value)

    # ───────── advanced table ─────────
    def rows(self) -> list[tuple[str, ArgSpec | None]]:
        known = self.cat.visible
        unknown = [k for k in self.mine if k not in self.cat and not k.startswith("@")]
        pinned = [s for s in known if s.key in self.mine or (self.apply_tuned and s.key in self.tuned)]
        rest = [s for s in known if s not in pinned]
        out: list[tuple[str, ArgSpec | None]] = [(k, None) for k in unknown]
        out += [(s.key, s) for s in pinned + rest]
        q = self.filter.lower().strip()
        if self.exp_only:
            out = [(k, s) for k, s in out if s and s.experimental]
        if q:
            out = [(k, s) for k, s in out if q in k.lower() or (s and (
                q in s.help.lower() or any(q in n.lower() for n in s.names + s.neg_names)))]
        return out

    def edit_row(self, key: str, spec: ArgSpec | None) -> None:
        kind = spec.kind if spec else "value"
        if kind == "flag":
            self.set_value(key, DEFAULT if self.value(key) else True)
        elif kind in ("toggle", "choice"):
            self.cycle_row(key, spec, 1)
        else:
            self.ask_value(key, spec)

    def cycle_row(self, key: str, spec: ArgSpec | None, d: int) -> None:
        if spec is None:
            return
        vals: list[Any] = {"toggle": [True, False], "flag": [True], "choice": spec.choices}.get(spec.kind, [])
        if not vals:
            return
        opts = [DEFAULT] + vals
        cur = self.value(key)
        i = 0 if cur is None else next((n for n, v in enumerate(opts) if v is not DEFAULT and v == cur), 0)
        self.set_value(key, opts[(i + d) % len(opts)])

    def ask_value(self, key: str, spec: ArgSpec | None) -> None:
        cur = self.value(key)
        label = f"--{key}" + (f" {spec.metavar}" if spec and spec.metavar else "")

        def submit(raw: str) -> str | None:
            if not raw.strip():
                self.reset(key)
            else:
                self.set_value(key, parse_value(raw.strip()))
            return None

        self.input = Input(label, "" if cur is None else show(cur), submit)

    def add_by_name(self) -> None:
        def submit(raw: str) -> str | None:
            raw = raw.strip()
            if not raw:
                return None
            k, eq, v = raw.partition("=")
            if not eq and " " in k:
                k, _, v = k.partition(" ")
                eq = "="
            spec, negated = self.cat.lookup(k)
            if spec is None:
                key = k.strip().lstrip("-")
                self.set_value(key, parse_value(v) if eq else True)
                self.say(f"--{key} isn't in this llama-server's --help — added as-is")
            else:
                key = spec.key
                val: Any = (False if negated else True) if not eq else parse_value(v)
                if spec.kind == "value" and not eq:
                    self.filter, self.asel = "", 0
                    self.ask_value(key, spec)
                    self._jump(key)
                    return None
                self.set_value(key, val)
            self.filter = ""
            self._jump(key)
            return None

        self.input = Input("add setting (name or name=value)", "", submit)

    def _jump(self, key: str) -> None:
        rows = self.rows()
        self.asel = next((i for i, (k, _) in enumerate(rows) if k == key), self.asel)

    # ───────── loop ─────────
    def run(self) -> str:
        """Returns "done" or "hub" (open the Hub, then call run() again)."""
        self.exit_reason = ""
        with KeyReader() as keys, Live(console=self.console, screen=True, auto_refresh=False,
                                       redirect_stdout=False, redirect_stderr=False) as live:
            last = 0.0
            while True:
                pressed = keys.drain()
                for key in pressed:
                    if self.handle(key):
                        return self.exit_reason or "done"
                now = time.monotonic()
                if pressed or now - last >= 0.5:
                    live.update(self.render(), refresh=True)
                    last = now
                time.sleep(0.03)

    def say(self, msg: str) -> None:
        self.flash = (msg, time.monotonic() + 3.5)

    def go(self, d: int) -> None:
        new = max(0, min(len(STEPS) - 1, self.step + d))
        if d > 0:
            err = self.check(STEPS[self.step][0])
            if err:
                self.say(err)
                return
        self.step, self.sel = new, 0

    def check(self, step: str) -> str:
        if step == "model" and not self.model:
            return "pick a model first (Enter), download one (h) or type a path (p)"
        if step == "name" and not NAME_RE.match(self.name or ""):
            return "set a profile name — letters, digits, . _ -"
        return ""

    def quit(self) -> bool:
        if not self.dirty:
            return True
        self.confirm = ("Quit without saving?", lambda: True)
        return False

    # ───────── keys ─────────
    def handle(self, key: str) -> bool:
        if key == "ctrl+c":
            return True
        if self.modal.active:
            self.modal.handle(key)
            return False
        if self.confirm:
            _, act = self.confirm
            self.confirm = None
            return act() if key in ("y", "Y", "enter") else False
        if self.input:
            return self.key_input(key)
        step = STEPS[self.step][0]
        if key == "tab":
            self.cmd_scroll += 1
            return False
        if key == "c" and step != "name":
            self.say("command copied to the clipboard" if copy_to_clipboard(self.command())
                     else "no clipboard tool found")
            return False
        if key == "u" and step in ("advanced", "review"):
            self.apply_tuned = not self.apply_tuned
            self.dirty = True
            self.say("tuned values " + ("applied" if self.apply_tuned else "ignored — only your values are used"))
            return False
        if key in ("q", "Q"):
            return self.quit()
        handler = {"server": self.key_server, "model": self.key_model, "advanced": self.key_advanced,
                   "review": self.key_review}.get(step, self.key_form)
        return handler(key)

    def key_input(self, key: str) -> bool:
        inp = self.input
        assert inp is not None
        if key == "enter":
            err = inp.submit(inp.buf)
            if err:
                self.say(err)
            elif self.input is inp:
                self.input = None
        elif key == "esc":
            self.input = None
        elif key == "backspace":
            inp.buf = inp.buf[:-1]
        elif key == "\x15":
            inp.buf = ""
        elif len(key) == 1 and key.isprintable():
            inp.buf += key
        if inp.live and self.input is inp:
            inp.live(inp.buf)
        return False

    def nav(self, key: str, cur: int, n: int) -> int:
        step = {"up": -1, "down": 1, "k": -1, "j": 1, "pgup": -10, "pgdn": 10}.get(key)
        if step is not None:
            return max(0, min(n - 1, cur + step)) if n else 0
        if key == "home":
            return 0
        if key == "end":
            return max(n - 1, 0)
        return cur

    def common(self, key: str) -> bool:
        """n / Esc between steps. True if handled."""
        if key == "n":
            self.go(1)
            return True
        if key in ("esc", "b"):
            self.go(-1)
            return True
        return False

    def key_server(self, key: str) -> bool:
        if key == "enter" or key == "e":
            def submit(raw: str) -> str | None:
                raw = raw.strip().strip('"')
                if not raw:
                    return None
                p = Path(raw).expanduser()
                if p.is_dir():
                    p = next((p / n for n in ("llama-server.exe", "llama-server") if (p / n).exists()), p)
                if not p.exists() and not shutil.which(raw):
                    return f"not found: {raw}"
                self.cfg.llama_server = str(p)
                self.cfg.save()
                self.binary = self.cfg.server_binary()
                self.say("reading llama-server --help …")
                self.cat = argspec.load(self.cfg, self.binary)
                self.hw = tuning.detect(self.binary)
                self.set_model(self.model, self.model_path, self.model_size, keep_mmproj=True)
                self.say(f"✔ {len(self.cat.visible)} settings read from {Path(self.binary or raw).name}")
                return None

            self.input = Input("path to llama-server (file or folder)", self.cfg.llama_server, submit)
            return False
        if key in ("esc", "b"):
            return self.quit()
        self.common(key)
        return False

    def key_model(self, key: str) -> bool:
        n = len(self.models)
        if key == "enter" and n:
            m = self.models[self.msel]
            self.set_model(m.ref, m.path, m.size)
            self.dirty = True
            self.go(1)
        elif key == "h":
            self.exit_reason = "hub"
            return True
        elif key == "p":
            def submit(raw: str) -> str | None:
                p = Path(raw.strip().strip('"')).expanduser()
                if not p.is_file():
                    return f"not a file: {p}"
                self.set_model(hub.model_ref(self.cfg, p), p, p.stat().st_size)
                self.dirty = True
                self.go(1)
                return None

            self.input = Input("model path (.gguf)", "", submit)
        elif key == "m":
            proj = self.mmproj_for(self.model_path)
            if not proj:
                self.say("no mmproj file next to this model")
            elif self.mine.get("mmproj"):
                self.mine.pop("mmproj")
                self.say("vision projector off")
            else:
                self.mine["mmproj"] = hub.model_ref(self.cfg, proj)
                self.say(f"vision projector: {proj.name}")
        elif key == "r":
            self.models = self._scan_models()
        elif key in ("d", "delete") and n:
            self.ask_delete_model(self.models[self.msel])
        elif not self.common(key):
            self.msel = self.nav(key, self.msel, n)
        return False

    def ask_delete_model(self, m: hub.LocalModel) -> None:
        p = self.p
        files = hub.model_files(m)
        users = []
        for name in list_profiles():
            try:
                if Profile.load(name).model == m.ref:
                    users.append(name)
            except Exception:
                continue
        lines = [Text.assemble(("Delete this model from disk?", p.text)), Text(""),
                 Text.assemble(("model    ", p.dim), (m.ref, f"bold {p.accent}")),
                 Text.assemble(("size     ", p.dim), (hub.human(m.size), p.text),
                               (f"  ·  {len(files)} files" if len(files) > 1 else "", p.dim)),
                 Text.assemble(("where    ", p.dim), ("Hugging Face cache (shared with llama.cpp -hf)"
                                                       if m.source == "hf" else str(m.path.parent), p.text))]
        if users:
            lines.append(Text.assemble(("used by  ", p.dim), (", ".join(users), f"bold {p.warn}")))
        if m.ref == self.model:
            lines.append(Text("This is the model chosen for this profile.", style=p.warn))
        lines += [Text(""), Text("The file is removed permanently — this can't be undone.", style=p.warn)]

        def do() -> None:
            try:
                freed = hub.delete_model(self.cfg, m)
            except PermissionError:
                self.say(f"{m.path.name} is in use — stop the server using it first")
                return
            except OSError as e:
                self.say(f"couldn't delete {m.path.name}: {e}")
                return
            if m.ref == self.model:
                self.model, self.model_path, self.model_size, self.info = "", None, 0, None
                self.mine.pop("mmproj", None)
                self.dirty = True
            self.models = self._scan_models()
            self.msel = min(self.msel, max(len(self.models) - 1, 0))
            self.say(f"✔ deleted {m.ref} — {hub.human(freed)} freed"
                     + (f" · profiles now missing it: {', '.join(users)}" if users else ""))

        self.modal.open("Delete model", lines, do)

    def key_form(self, key: str) -> bool:
        fs = self.fields()
        self.sel = min(self.sel, max(len(fs) - 1, 0))
        fd = fs[self.sel] if fs else None
        if fd and key in ("left", "h"):
            self.cycle(fd, -1)
        elif fd and key in ("right", "l", " "):
            self.cycle(fd, 1)
        elif fd and key in ("enter", "e"):
            if fd.custom:
                cur, src = self.field_value(fd)

                def submit(raw: str, fd: Field = fd) -> str | None:
                    raw = raw.strip()
                    if fd.key == "@name":
                        if not NAME_RE.match(raw):
                            return "letters, digits, . _ - (must start with a letter or digit)"
                        self.set_field(fd, raw)
                        return None
                    if fd.key == "@port":
                        if not raw.isdigit() or not 0 < int(raw) < 65536:
                            return "port must be 1-65535"
                        self.set_field(fd, int(raw))
                        return None
                    if fd.key.startswith("@"):
                        self.set_field(fd, raw)
                    elif not raw:
                        self.reset(fd.key)
                    else:
                        self.set_field(fd, parse_value(raw))
                    return None

                self.input = Input(fd.label, "" if cur is None or (src == "default" and not fd.key.startswith("@"))
                                   else show(cur), submit)
            else:
                self.cycle(fd, 1)
        elif fd and key in ("delete", "backspace") and not fd.key.startswith("@"):
            self.reset(fd.key)
        elif not self.common(key):
            self.sel = self.nav(key, self.sel, len(fs))
        return False

    def key_advanced(self, key: str) -> bool:
        rows = self.rows()
        self.asel = min(self.asel, max(len(rows) - 1, 0))
        cur = rows[self.asel] if rows else None
        if key == "/":
            def live(buf: str) -> None:
                self.filter, self.asel = buf, 0

            self.input = Input("filter", self.filter, lambda _: None, live)
        elif key == "x":
            self.exp_only = not self.exp_only
            self.asel = 0
        elif key == "a":
            self.add_by_name()
        elif cur and key in ("enter", "e"):
            self.edit_row(*cur) if key == "enter" else self.ask_value(*cur)
        elif cur and key in ("left", "right"):
            self.cycle_row(cur[0], cur[1], -1 if key == "left" else 1)
        elif cur and key in ("delete", "backspace"):
            self.reset(cur[0])
        elif key == "esc" and self.filter:
            self.filter = ""
        elif not self.common(key):
            self.asel = self.nav(key, self.asel, len(rows))
        return False

    def key_review(self, key: str) -> bool:
        if key in ("enter", "s"):
            return self.save(serve=key == "s")
        self.common(key)
        return False

    def save(self, *, serve: bool) -> bool:
        for s in ("model", "name"):
            err = self.check(s)
            if err:
                self.say(err)
                return False
        path = Profile.path_for(self.name)

        def do() -> bool:
            prof = self.profile()
            prof.name = self.name
            self.result = Result(self.name, prof.save(), serve)
            return True

        if path.exists() and self.name != self.editing:
            self.confirm = (f"Profile '{self.name}' exists — overwrite it?", do)
            return False
        return do()

    # ───────── render ─────────
    def render(self) -> RenderableType:
        p = self.p
        w, h = self.console.size
        cmd_lines = self.cmd_wrapped(w - 2)
        rows = min(CMD_ROWS, max(len(cmd_lines), 1))
        body_h = max(h - 2 - 1 - 1 - 1 - rows - 1 - 3, 5)
        root = Layout()
        root.split_column(
            Layout(self.header(), size=2),
            Layout(Text("─" * w, style=p.dim), size=1),
            Layout(self.body(body_h), size=body_h),
            Layout(Text("─" * w, style=p.dim), size=1),
            Layout(self.cmd_view(cmd_lines, rows), size=rows + 1),
            Layout(Text("─" * w, style=p.dim), size=1),
            Layout(self.describe(), size=1),
            Layout(self.hints(), size=2),
        )
        return root

    def header(self) -> RenderableType:
        p = self.p
        t = Text()
        t.append("INARI", style=f"bold {p.accent}")
        t.append("  Set up", style=f"bold {p.accent2}")
        t.append(f"    Step {self.step + 1} of {len(STEPS)}: {STEPS[self.step][1]}", style=f"bold {p.text}")
        if self.editing:
            t.append(f"    editing {self.editing}", style=p.dim)
        elif self.origin:
            t.append(f"    {self.origin}", style=p.dim)
        dots = Text()
        for i, (_, label) in enumerate(STEPS):
            on = i == self.step
            dots.append(label if on else "●", style=f"bold {p.accent}" if on else (p.accent2 if i < self.step
                                                                                    else p.dim))
            dots.append("  ")
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True, overflow="ellipsis")
        g.add_row(t)
        g.add_row(dots)
        return g

    def body(self, h: int) -> RenderableType:
        step = STEPS[self.step][0]
        if step == "server":
            return self.view_server()
        if step == "model":
            return self.modal.render(self.p) if self.modal.active else self.view_model(h)
        if step == "advanced":
            return self.view_advanced(h)
        if step == "review":
            return self.view_review(h)
        return self.view_form(h)

    def title(self, title: str, sub: str = "") -> Text:
        t = Text()
        t.append(title, style=f"bold {self.p.accent}")
        if sub:
            t.append(f"   {sub}", style=self.p.dim)
        return t

    def kv(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(style=self.p.dim, no_wrap=True)
        t.add_column(style=self.p.text, overflow="fold")
        return t

    def view_server(self) -> RenderableType:
        p = self.p
        t = self.kv()
        if self.binary:
            t.add_row("binary", Text(self.binary, style=f"bold {p.text}"))
            t.add_row("build", self.cat.version or "—")
            t.add_row("settings", f"{len(self.cat.visible)} flags read from --help")
        else:
            t.add_row("binary", Text("not found", style=f"bold {p.bad}"))
            t.add_row("", Text("Point inari at llama-server(.exe) or the folder that contains it (Enter).\n"
                               "Builds: https://github.com/ggml-org/llama.cpp/releases", style=p.text))
            t.add_row("settings", f"{len(self.cat.visible)} common flags (built-in list until a binary is set)")
        t.add_row("devices", self.hw.describe())
        for d in self.hw.devices:
            t.add_row("", Text(f"{d.id}: {d.name}  {tuning.gb(d.total)} total · {tuning.gb(d.free)} free",
                               style=p.accent2))
        t.add_row("RAM", f"{tuning.gb(self.hw.ram)} · {self.hw.cores} cores")
        return Group(self.title("Which llama-server?", "the build decides which settings exist"), Text(""), t)

    def view_model(self, h: int) -> RenderableType:
        p = self.p
        rows = max(h - 8, 3)
        t = Table(box=box.SIMPLE_HEAD, border_style=p.dim, header_style=f"bold {p.accent}", expand=True,
                  padding=(0, 1), show_edge=False)
        t.add_column("", width=1, no_wrap=True)
        t.add_column("MODEL", ratio=1, no_wrap=True, overflow="ellipsis")
        t.add_column("SIZE", justify="right", no_wrap=True)
        t.add_column("IN", no_wrap=True)
        if not self.models:
            t.add_row("", Text("No local models yet — press h to search Hugging Face, or p to type a path.",
                               style=p.dim), "", "")
        a, b = window(self.msel, len(self.models), rows)
        for i in range(a, b):
            m = self.models[i]
            on = i == self.msel
            mark = "●" if m.ref == self.model else ("▸" if on else " ")
            t.add_row(Text(mark, style=p.accent), Text(m.ref, style=f"bold {p.accent2}" if on else p.text),
                      hub.human(m.size), Text(m.source, style=p.dim), style=f"on {p.track}" if on else "")
        info = self.kv()
        if self.model:
            mi = self.info
            info.add_row("chosen", Text(self.model, style=f"bold {p.text}"))
            if mi:
                desc = f"{mi.arch}  {mi.size_label}".strip()
                if mi.moe:
                    desc += f"  · MoE {mi.experts_used}/{mi.experts} experts"
                info.add_row("model", f"{desc}  · {mi.layers} layers · trained ctx {mi.ctx_train:,}")
            proj = self.mmproj_for(self.model_path)
            if proj:
                on = bool(self.mine.get("mmproj"))
                info.add_row("vision", Text(f"[{'x' if on else ' '}] {proj.name}  (m toggles)",
                                            style=p.accent2 if on else p.dim))
        return Group(self.title("Models", f"{len(self.models)} on disk · Enter uses one for this profile · d deletes it from disk"), t, Text(""), info)

    def view_form(self, h: int) -> RenderableType:
        p = self.p
        step = STEPS[self.step][0]
        fs = self.fields()
        self.sel = min(self.sel, max(len(fs) - 1, 0))
        t = Table.grid(padding=(0, 2), expand=True)
        t.add_column(width=1, no_wrap=True)
        t.add_column(no_wrap=True, min_width=22)
        t.add_column(no_wrap=True, ratio=1, overflow="ellipsis")
        t.add_column(no_wrap=True, justify="right")
        for i, fd in enumerate(fs):
            on = i == self.sel
            v, src = self.field_value(fd)
            if (src == "default" or v is None) and not fd.key.startswith("@"):
                spec = self.cat.get(fd.key)
                shown = Text(f"default ({spec.default})" if spec and spec.default else "default", style=p.dim)
            else:
                label = next((o.label for o in fd.options if o.value == v), show(v))
                if fd.key == "ctx-size" and isinstance(v, int):
                    label = f"{v:,}"
                shown = Text(label or "—", style=f"bold {p.accent2}" if on else p.text)
            val = Text()
            val.append("‹ " if on and fd.options else "  ", style=p.accent)
            val.append_text(shown)
            val.append(" ›" if on and fd.options else "", style=p.accent)
            tag = {"yours": ("yours", p.accent), "tuned": ("tuned", p.good), "default": ("", p.dim)}[src]
            t.add_row(Text("▸" if on else " ", style=p.accent),
                      Text(fd.label, style=f"bold {p.text}" if on else p.text), val,
                      Text(tag[0] if not fd.key.startswith("@") else "", style=tag[1]),
                      style=f"on {p.track}" if on else "")
        titles = {"name": ("Name the profile", "how you'll start it: inari serve <name>"),
                  "context": ("Context & memory", "how much the model remembers, and what that costs"),
                  "gpu": ("GPU offload", "where the weights live"),
                  "chat": ("Chat", "template, thinking and sampling defaults"),
                  "network": ("Network", "who can reach the server")}
        parts: list[RenderableType] = [self.title(*titles.get(step, (step, ""))), Text("")]
        parts.append(t)
        extra = self.step_info(step)
        if extra is not None:
            parts += [Text(""), extra]
        return Group(*parts)

    def step_info(self, step: str) -> RenderableType | None:
        p, mi = self.p, self.info
        if step == "context" and mi:
            ctx = self.value("ctx-size") or mi.ctx_train
            kt, vt = self.value("cache-type-k") or "f16", self.value("cache-type-v") or "f16"
            kv = mi.kv_bytes(int(ctx) if str(ctx).isdigit() else 0, kt, vt)
            t = self.kv()
            t.add_row("model", f"trained on {mi.ctx_train:,} tokens")
            if kv:
                t.add_row("KV cache", Text(f"≈ {tuning.gb(kv)} for {int(ctx):,} tokens ({kt}/{vt})",
                                           style=p.accent2))
            return t
        if step == "gpu":
            t = self.kv()
            t.add_row("devices", self.hw.describe())
            vram = self.hw.vram
            if vram and self.model_size:
                ctx = self.value("ctx-size")
                kv = mi.kv_bytes(int(ctx), self.value("cache-type-k") or "f16",
                                 self.value("cache-type-v") or "f16") if mi and isinstance(ctx, int) else 0
                need = self.model_size + kv
                frac = need / vram
                color = p.good if frac < 0.9 else (p.warn if frac < 1.2 else p.bad)
                t.add_row("needs", Text(f"≈ {tuning.gb(need)} (weights {tuning.gb(self.model_size)}"
                                        f" + KV {tuning.gb(kv)}) of {tuning.gb(vram)} free VRAM", style=color))
                bar = Table.grid()
                bar.add_column(width=48)
                bar.add_row(Bar(min(frac, 1.0), color, p.track))
                t.add_row("", bar)
            return t
        return None

    def view_advanced(self, h: int) -> RenderableType:
        p = self.p
        rows = self.rows()
        self.asel = min(self.asel, max(len(rows) - 1, 0))
        n_rows = max(h - 3, 3)
        t = Table(box=None, expand=True, padding=(0, 1), show_edge=False, header_style=f"bold {p.text}")
        t.add_column("", width=1, no_wrap=True)
        t.add_column("Setting", no_wrap=True, width=28, overflow="ellipsis")
        t.add_column("Default", no_wrap=True, width=12, overflow="ellipsis")
        t.add_column("Tuned", no_wrap=True, width=12, overflow="ellipsis")
        t.add_column("Yours", no_wrap=True, width=12, overflow="ellipsis")
        t.add_column("What it does", no_wrap=True, ratio=1, overflow="ellipsis")
        a, b = window(self.asel, len(rows), n_rows)
        for i in range(a, b):
            key, spec = rows[i]
            on = i == self.asel
            st = f"bold {p.accent}" if on else p.text
            tuned = show(self.tuned[key]) if key in self.tuned else "-"
            yours = show(self.mine[key]) if key in self.mine else "-"
            what = Text()
            if spec is None:
                what.append("not in this llama-server's --help (passed as-is)", style=p.warn)
            else:
                if spec.experimental:
                    what.append("[experimental] ", style=p.warn)
                what.append(spec.help, style=st if on else p.text)
            t.add_row(Text("›" if on else " ", style=p.accent), Text(f"--{key}", style=st),
                      Text(spec.default if spec and spec.default else "-", style=st if on else p.dim),
                      Text(tuned, style=st if on else (p.good if tuned != "-" else p.dim)),
                      Text(yours, style=f"bold {p.accent2}" if yours != "-" else (st if on else p.dim)),
                      what, style=f"on {p.track}" if on else "")
        head = Text()
        head.append("All settings", style=f"bold {p.accent}")
        head.append("   (optional · for tinkerers)", style=p.dim)
        if self.filter or (self.input and self.input.live):
            head.append(f"   filter: {self.filter}", style=f"bold {p.accent2}")
        else:
            head.append("   / filter", style=p.dim)
        head.append("   x " + ("showing experimental only" if self.exp_only else "show experimental only"),
                    style=p.accent2 if self.exp_only else p.dim)
        if not self.apply_tuned:
            head.append("   tuned values ignored (u)", style=p.warn)
        foot = Text(f"{a + 1 if rows else 0}-{b} of {len(rows)}", style=p.dim)
        return Group(head, t, foot)

    def view_review(self, h: int) -> RenderableType:
        p = self.p
        t = self.kv()
        path = Profile.path_for(self.name) if self.name else None
        status = ""
        if path and path.exists():
            status = "  (updates it)" if self.name == self.editing else "  (exists — will be overwritten)"
        t.add_row("profile", Text.assemble((self.name or "—", f"bold {p.accent}"), (status, p.warn)))
        if path:
            t.add_row("file", str(path))
        t.add_row("model", self.model or Text("— none —", style=p.bad))
        t.add_row("listen", f"http://{self.host}:{self.port}")
        if self.description:
            t.add_row("description", self.description)
        eff = self.effective()
        lines = Table.grid(padding=(0, 2), expand=True)
        lines.add_column(no_wrap=True, style=p.accent2, min_width=22)
        lines.add_column(no_wrap=True, max_width=48, overflow="ellipsis")
        lines.add_column(style=p.dim, overflow="ellipsis", no_wrap=True, ratio=1)
        shown = 0
        for k, v in eff.items():
            if shown >= max(h - 9, 1):
                lines.add_row("…", f"{len(eff) - shown} more", "")
                break
            src = self.source(k)
            lines.add_row(f"--{k}", Text(show(v), style=f"bold {p.text}"),
                          Text(src + (f" · {self.why[k]}" if src == "tuned" and k in self.why else ""),
                               style=p.good if src == "tuned" else p.dim))
            shown += 1
        if not eff:
            lines.add_row("", Text("everything at llama.cpp defaults", style=p.dim), "")
        return Group(self.title("Review", "Enter saves the profile · s saves and starts it"), Text(""), t,
                     Text(""), self.title("Settings"), lines)

    def cmd_wrapped(self, width: int) -> list[str]:
        text = Text(self.command())
        return [ln.plain for ln in text.wrap(self.console, max(width, 20))]

    def cmd_view(self, lines: list[str], rows: int) -> RenderableType:
        p = self.p
        max_scroll = max(len(lines) - rows, 0)
        if self.cmd_scroll > max_scroll:
            self.cmd_scroll = 0
        out = Text()
        out.append("Command that will run", style=f"bold {p.accent}")
        out.append(" (your choices above are in it)", style=p.dim)
        if max_scroll:
            out.append(f"   Tab scroll {self.cmd_scroll + 1}/{max_scroll + 1}", style=p.dim)
        for ln in lines[self.cmd_scroll:self.cmd_scroll + rows]:
            out.append("\n" + ln, style=p.text)
        return out

    def describe(self) -> RenderableType:
        p = self.p
        if self.confirm:
            return Text.assemble((self.confirm[0] + "  ", f"bold {p.warn}"), ("y", f"bold {p.accent}"),
                                 (" yes   ", p.text), ("any other key", f"bold {p.accent}"), (" no", p.text))
        if self.input:
            return Text.assemble((self.input.label + ": ", f"bold {p.accent}"), (self.input.buf, p.text),
                                 ("▌", p.accent))
        if self.flash and time.monotonic() < self.flash[1]:
            return Text(self.flash[0], style=f"bold {p.accent2}", no_wrap=True, overflow="ellipsis")
        step = STEPS[self.step][0]
        key, label, help_ = "", "", ""
        if step == "advanced":
            rows = self.rows()
            if rows:
                key, spec = rows[min(self.asel, len(rows) - 1)]
                help_ = spec.help if spec else "unknown to this llama-server build"
                label = f"--{key}"
        elif step not in ("server", "model", "review"):
            fs = self.fields()
            if fs:
                fd = fs[min(self.sel, len(fs) - 1)]
                key, label, help_ = fd.key, fd.label, fd.note
        if not label:
            return Text(" ")
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(f"{label}: ", style=p.dim)
        t.append(help_, style=p.dim)
        if key in self.why and self.apply_tuned:
            t.append(f"  ·  tuned: {self.why[key]}", style=p.good)
        return t

    def hints(self) -> RenderableType:
        p = self.p
        step = STEPS[self.step][0]
        if self.input:
            items = [("Enter", "ok"), ("Esc", "cancel"), ("Ctrl+U", "clear")]
            if step == "advanced":
                items.append(("empty", "= reset"))
        elif step == "server":
            items = [("Enter", "set path"), ("n", "next"), ("c", "copy command"), ("q", "quit")]
        elif step == "model":
            items = [("↑↓", "choose"), ("Enter", "use"), ("h", "download from Hub"), ("p", "path"),
                     ("m", "mmproj"), ("d", "delete from disk"), ("n", "next"), ("Esc", "back"), ("q", "quit")]
        elif step == "advanced":
            items = [("n", "next"), ("↑↓", "choose"), ("Enter", "edit"), ("←→", "cycle"), ("Esc", "back"),
                     ("Del", "reset"), ("x", "experimental"), ("/", "filter"), ("a", "add by name"),
                     ("u", "tuned on/off"), ("c", "copy command"), ("Tab", "scroll command")]
        elif step == "review":
            items = [("Enter", "save"), ("s", "save & serve"), ("u", "tuned on/off"), ("c", "copy command"),
                     ("Esc", "back"), ("q", "quit")]
        else:
            items = [("n", "next"), ("↑↓", "choose"), ("←→", "change"), ("Enter", "type a value"),
                     ("Del", "reset"), ("Esc", "back"), ("c", "copy command"), ("q", "quit")]
        t = Text()
        for key, label in items:
            t.append(key, style=f"bold {p.accent}")
            t.append(f" {label}   ", style=p.dim)
        return t


def run_wizard(cfg: Config, console: Console, *, name: str = "", start: str = "", seed: Profile | None = None,
               origin: str = "") -> Result | None:
    with console.status("[accent]Reading llama-server settings and devices…[/]"):
        wiz = Wizard(cfg, console, name=name, theme_name=cfg.theme, start=start, seed=seed, origin=origin)
    while True:
        try:
            reason = wiz.run()
        except KeyboardInterrupt:
            return None
        if reason != "hub":
            return wiz.result
        from inari.hubtui import HubApp

        app = HubApp(cfg, console, theme_name=wiz.theme_name)
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        wiz.models = wiz._scan_models()
        done = [j for j in app.dl.jobs if j.status == "done" and j.path]
        if done:
            j = done[-1]
            wiz.msel = next((i for i, m in enumerate(wiz.models) if m.path == j.path), wiz.msel)
            m = wiz.models[wiz.msel] if wiz.models else None
            if m:
                wiz.set_model(m.ref, m.path, m.size)
                if j.mmproj_path:
                    wiz.mine["mmproj"] = hub.model_ref(cfg, j.mmproj_path)
                wiz.dirty = True
                wiz.say(f"✔ downloaded {m.path.name} — selected")


if __name__ == "__main__":  # pragma: no cover
    c = Config.load()
    print(run_wizard(c, th.make_console(c.theme)), file=sys.stderr)
