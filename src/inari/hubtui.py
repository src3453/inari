"""Inari Hub: a full-screen Hugging Face GGUF browser (search → pick quant → download → profile)."""

from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inari import hub
from inari import theme as th
from inari.config import Config
from inari.keys import KeyReader
from inari.profiles import NAME_RE, Profile, new_profile
from inari.tui import Bar, dur, k


def params(n: int) -> str:
    if not n:
        return "—"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def ago(ts: float | None) -> str:
    if not ts:
        return "—"
    d = max(0.0, time.time() - ts) / 86400
    if d < 1:
        return "today"
    if d < 30:
        return f"{int(d)}d"
    if d < 365:
        return f"{int(d / 30)}mo"
    return f"{d / 365:.1f}y"


def window(sel: int, n: int, rows: int) -> tuple[int, int]:
    """Visible [start, end) slice that keeps ``sel`` in view, roughly centred."""
    rows = max(rows, 1)
    if n <= rows:
        return 0, n
    start = max(0, min(sel - rows // 2, n - rows))
    return start, start + rows


class HubApp:
    def __init__(self, cfg: Config, console: Console, *, query: str = "", theme_name: str = "inari") -> None:
        self.cfg = cfg
        self.console = console
        self.theme_name = theme_name
        self.p = th.get(theme_name)
        self.dl = hub.Downloader(cfg)

        self.screen = "search"  # search | files
        self.focus = "main"  # main | downloads
        self.query = query
        self.editing = not query
        self.sort = 0
        self.hits: list[hub.RepoHit] = []
        self.sel = 0
        self.searching = False
        self.searched = ""
        self.error = ""

        self.repo: hub.RepoHit | None = None
        self.groups: list[hub.GgufGroup] = []
        self.projs: list[hub.GgufGroup] = []
        self.fsel = 0
        self.loading_files = False
        self.with_mmproj = False

        self.jsel = 0
        self.naming: hub.Job | None = None  # profile-name prompt for this job
        self.name_buf = ""
        self.confirm_quit = False
        self.flash: tuple[str, float] | None = None
        self._gen = 0  # discards stale background results
        if query:
            self.search()

    # ───────── background work ─────────
    def _bg(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def search(self) -> None:
        self._gen += 1
        gen, q, sort = self._gen, self.query, hub.SORTS[self.sort][0]
        self.searching, self.error = True, ""

        def work() -> None:
            try:
                hits = hub.search_repos(q, self.cfg, sort=sort)
                err = ""
            except Exception as e:
                hits, err = [], f"search failed: {e}"
            if gen == self._gen:
                self.hits, self.sel, self.error, self.searched, self.searching = hits, 0, err, q, False

        self._bg(work)

    def open_repo(self, hit: hub.RepoHit) -> None:
        self._gen += 1
        gen = self._gen
        self.repo, self.groups, self.projs, self.fsel = hit, [], [], 0
        self.screen, self.loading_files, self.error = "files", True, ""

        def work() -> None:
            try:
                groups = hub.list_groups(hit.id, self.cfg)
                err = ""
            except Exception as e:
                groups, err = [], f"could not list files: {e}"
            if gen != self._gen:
                return
            self.groups = [g for g in groups if not g.is_mmproj]
            self.projs = [g for g in groups if g.is_mmproj]
            self.with_mmproj = bool(self.projs) and hit.pipeline == "image-text-to-text"
            self.fsel = self._default_pick()
            self.error, self.loading_files = err, False

        self._bg(work)

    def _default_pick(self) -> int:
        for pref in ("Q4_K_M", "Q4_K_XL", "Q4_0", "Q5_K_M"):
            for i, g in enumerate(self.groups):
                if g.quant.endswith(pref):
                    return i
        return 0

    def projector(self) -> hub.GgufGroup | None:
        if not self.projs:
            return None
        return next((p for p in self.projs if "f16" in p.name.lower()), self.projs[0])

    # ───────── loop ─────────
    def run(self) -> None:
        with KeyReader() as keys, Live(console=self.console, screen=True, auto_refresh=False,
                                       redirect_stdout=False, redirect_stderr=False) as live:
            last = 0.0
            while True:
                pressed = keys.drain()
                for key in pressed:
                    if self.handle(key):
                        return
                now = time.monotonic()
                if pressed or now - last >= 0.25:
                    live.update(self.render(), refresh=True)
                    last = now
                time.sleep(0.03)

    def say(self, msg: str) -> None:
        self.flash = (msg, time.monotonic() + 3.0)

    # ───────── keys ─────────
    def handle(self, key: str) -> bool:
        if key == "ctrl+c":
            for j in self.dl.jobs:
                j.cancel()
            return True
        if self.confirm_quit:
            self.confirm_quit = False
            if key in ("y", "Y"):
                for j in self.dl.jobs:
                    j.cancel()
                return True
            return False
        if self.naming is not None:
            return self.key_naming(key)
        if self.editing:
            return self.key_edit(key)

        if key in ("q", "Q"):
            if self.dl.busy():
                self.confirm_quit = True
                return False
            return True
        if key == "t":
            self.theme_name = th.next_name(self.theme_name)
            self.p = th.get(self.theme_name)
            return False
        if key == "tab":
            if self.dl.jobs:
                self.focus = "downloads" if self.focus == "main" else "main"
                self.jsel = min(self.jsel, len(self.dl.jobs) - 1)
            return False
        if self.focus == "downloads":
            return self.key_downloads(key)
        if self.screen == "search":
            return self.key_search(key)
        return self.key_files(key)

    def key_edit(self, key: str) -> bool:
        if key == "enter":
            self.editing = False
            self.search()
        elif key == "esc":
            self.editing = False
        elif key == "backspace":
            self.query = self.query[:-1]
        elif key == "\x15":  # ctrl+u
            self.query = ""
        elif len(key) == 1 and key.isprintable():
            self.query += key
        return False

    def key_naming(self, key: str) -> bool:
        job = self.naming
        assert job is not None
        if key == "enter":
            name = self.name_buf.strip()
            if not NAME_RE.match(name):
                self.say("profile names: letters, digits, . _ -")
                return False
            if Profile.path_for(name).exists():
                self.say(f"profile '{name}' already exists — pick another name")
                return False
            self.make_profile(job, name)
            self.naming = None
        elif key == "esc":
            self.naming = None
        elif key == "backspace":
            self.name_buf = self.name_buf[:-1]
        elif len(key) == 1 and key.isprintable():
            self.name_buf += key
        return False

    def move(self, cur: int, n: int, key: str, page: int = 10) -> int:
        step = {"up": -1, "down": 1, "k": -1, "j": 1, "pgup": -page, "pgdn": page}.get(key)
        if step is not None:
            return max(0, min(n - 1, cur + step)) if n else 0
        if key == "home":
            return 0
        if key == "end":
            return max(n - 1, 0)
        return cur

    def key_search(self, key: str) -> bool:
        if key in ("/", "s", "i"):
            self.editing = True
        elif key == "o":
            self.sort = (self.sort + 1) % len(hub.SORTS)
            self.search()
        elif key == "r":
            self.search()
        elif key in ("enter", "right", "l") and self.hits:
            self.open_repo(self.hits[self.sel])
        elif key == "b" and self.hits:
            webbrowser.open(f"https://huggingface.co/{self.hits[self.sel].id}")
        elif key == "esc":
            self.editing = True
        else:
            self.sel = self.move(self.sel, len(self.hits), key)
        return False

    def key_files(self, key: str) -> bool:
        if key in ("esc", "left", "h", "backspace"):
            self.screen = "search"
            self._gen += 1
            self.loading_files = False
        elif key in ("enter", "d") and self.groups and self.repo:
            g = self.groups[self.fsel]
            proj = self.projector() if self.with_mmproj else None
            if hub.local_state(self.repo.id, g, self.cfg) == "done" and (
                    proj is None or hub.local_state(self.repo.id, proj, self.cfg) == "done"):
                self.say(f"{g.name} is already downloaded")
                return False
            if any(j.active and j.repo == self.repo.id and j.group.name == g.name for j in self.dl.jobs):
                self.say(f"{g.name} is already queued")
                return False
            self.dl.jobs = [j for j in self.dl.jobs if not (
                j.status in ("cancelled", "failed") and j.repo == self.repo.id and j.group.name == g.name)]
            self.dl.add(self.repo.id, g, proj)
            self.say(f"queued {g.name} ({hub.human(g.size + (proj.size if proj else 0))})")
        elif key == "m":
            if self.projs:
                self.with_mmproj = not self.with_mmproj
            else:
                self.say("this repo has no vision projector (mmproj)")
        elif key == "b" and self.repo:
            webbrowser.open(f"https://huggingface.co/{self.repo.id}")
        elif key == "/":
            self.screen, self.editing = "search", True
        else:
            self.fsel = self.move(self.fsel, len(self.groups), key)
        return False

    def key_downloads(self, key: str) -> bool:
        jobs = self.dl.jobs
        if not jobs:
            self.focus = "main"
            return False
        self.jsel = min(self.jsel, len(jobs) - 1)
        job = jobs[self.jsel]
        if key in ("x", "delete"):
            if job.active:
                job.cancel()
                self.say(f"cancelling {job.group.name} (partial file kept for resume)")
        elif key in ("p", "enter"):
            if job.status != "done":
                self.say("the download hasn't finished yet")
            elif job.profile:
                self.say(f"profile '{job.profile}' already made — inari serve {job.profile}")
            else:
                self.naming = job
                self.name_buf = Path(job.group.name).stem.split("-0000")[0].lower()
        elif key == "esc":
            self.focus = "main"
        else:
            self.jsel = self.move(self.jsel, len(jobs), key)
        return False

    def make_profile(self, job: hub.Job, name: str) -> None:
        assert job.path is not None
        try:
            model = str(job.path.resolve().relative_to(self.cfg.models_path().resolve())).replace("\\", "/")
        except ValueError:
            model = str(job.path)
        prof = new_profile(name, model=model, description=f"from {job.repo}")
        if job.mmproj_path:
            prof.args["mmproj"] = str(job.mmproj_path)
        prof.save()
        job.profile = name
        self.say(f"✔ profile '{name}' saved — run it with: inari serve {name}")

    # ───────── render ─────────
    def render(self) -> RenderableType:
        w, h = self.console.size
        jobs_h = min(len(self.dl.jobs), 5) + 2 if self.dl.jobs else 0
        body_h = max(h - 6 - jobs_h, 5)
        parts = [
            Layout(self.header(), size=2),
            Layout(Text("─" * w, style=self.p.dim), size=1),
            Layout(self.view_search(body_h) if self.screen == "search" else self.view_files(body_h), size=body_h),
        ]
        if jobs_h:
            parts.append(Layout(self.view_downloads(jobs_h), size=jobs_h))
        parts += [Layout(self.hints(), size=1), Layout(Text(" "), size=1), Layout(self.statusbar(), size=1)]
        root = Layout()
        root.split_column(*parts)
        return root

    def header(self) -> RenderableType:
        p = self.p
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True, overflow="ellipsis")
        g.add_column(justify="right", no_wrap=True)
        left = Text()
        left.append("INARI", style=f"bold {p.accent}")
        left.append("  Hub", style=f"bold {p.accent2}")
        left.append("    huggingface.co", style=f"bold {p.text}")
        if self.screen == "files" and self.repo:
            left.append(f"/{self.repo.id}", style=f"bold {p.text}")
        right = Text()
        active = [j for j in self.dl.jobs if j.status == "downloading"]
        queued = sum(1 for j in self.dl.jobs if j.status == "queued")
        if active:
            j = active[0]
            right.append("● ", style=p.accent)
            speed = f"{hub.human(j.speed)}/s" if j.speed else "starting…"
            right.append(f"downloading {j.frac * 100:.0f}% · {speed}", style=p.accent)
            if queued:
                right.append(f" · +{queued} queued", style=p.accent2)
        elif self.searching or self.loading_files:
            right.append("● ", style=p.info)
            right.append("searching…" if self.searching else "listing files…", style=p.info)
        else:
            right.append("● ", style=p.good)
            right.append("ready", style=p.good)
        g.add_row(left, right)

        tabs = Text()
        for i, (key, label) in enumerate((("search", "Search"), ("files", "Files"))):
            if key == self.screen:
                tabs.append(f"[{label}]", style=f"bold {p.accent}")
            else:
                tabs.append(f" {label} ", style=p.text if (key == "search" or self.repo) else p.dim)
            if i == 0:
                tabs.append(" › ", style=p.dim)
        g.add_row(tabs, Text(""))
        return g

    def panel(self, body: RenderableType, title: str, *, focused: bool = True, height: int | None = None,
              subtitle: str = "") -> Panel:
        p = self.p
        return Panel(body, title=Text(f" {title} ", style=f"bold {p.accent}" if focused else p.text),
                     title_align="left", subtitle=Text(subtitle, style=p.dim) if subtitle else None,
                     subtitle_align="right", border_style=p.accent if focused else p.dim, box=box.ROUNDED,
                     padding=(0, 1), height=height)

    def search_box(self) -> Panel:
        p = self.p
        t = Text()
        t.append("⌕ ", style=p.accent)
        if self.query or self.editing:
            t.append(self.query, style=f"bold {p.text}")
        else:
            t.append("all GGUF repos", style=p.dim)
        if self.editing:
            t.append("▌", style=p.accent)
        sort = hub.SORTS[self.sort][1]
        return self.panel(t, "Search Hugging Face", focused=self.editing, height=3,
                          subtitle=f" filter: gguf · sort: {sort} ")

    def view_search(self, height: int) -> RenderableType:
        p = self.p
        rows = max(height - 3 - 4, 1)
        focused = self.focus == "main" and not self.editing
        if self.error:
            body: RenderableType = Text(self.error, style=p.bad)
        elif self.searching and not self.hits:
            body = Text("Searching…", style=p.info)
        elif not self.hits:
            body = Text("No GGUF repos found. Try another search (press /).", style=p.dim) if self.searched or \
                not self.editing else Text("Type a model name and press Enter — e.g. qwen3 8b, gemma, unsloth/",
                                           style=p.dim)
        else:
            t = Table(box=box.SIMPLE_HEAD, border_style=p.dim, header_style=f"bold {p.accent}", expand=True,
                      padding=(0, 1), show_edge=False)
            t.add_column("", width=1, no_wrap=True)
            t.add_column("REPO", ratio=1, no_wrap=True, overflow="ellipsis")
            t.add_column("ARCH", no_wrap=True, max_width=12, overflow="ellipsis")
            t.add_column("PARAMS", justify="right", no_wrap=True)
            t.add_column("CTX", justify="right", no_wrap=True)
            t.add_column("DOWNLOADS", justify="right", no_wrap=True)
            t.add_column("♥", justify="right", no_wrap=True)
            t.add_column("UPDATED", justify="right", no_wrap=True)
            a, b = window(self.sel, len(self.hits), rows)
            for i in range(a, b):
                hit = self.hits[i]
                on = i == self.sel
                owner, _, name = hit.id.partition("/")
                repo = Text()
                repo.append(owner + "/", style=p.dim)
                repo.append(name, style=f"bold {p.accent2}" if on else p.text)
                if hit.gated:
                    repo.append(" 🔒", style=p.warn)
                if hit.pipeline == "image-text-to-text":
                    repo.append(" vision", style=p.info)
                t.add_row(Text("▸" if on else " ", style=p.accent), repo, Text(hit.arch, style=p.dim),
                          params(hit.params), k(hit.ctx) if hit.ctx else "—", k(hit.downloads),
                          k(hit.likes), ago(hit.modified),
                          style=f"on {p.track}" if on and focused else (p.text if not on else ""))
            body = t
        count = f" {self.sel + 1}/{len(self.hits)} " if self.hits else ""
        results = self.panel(body, "Results", focused=focused, height=height - 3, subtitle=count)
        return Group(self.search_box(), results)

    def view_files(self, height: int) -> RenderableType:
        p = self.p
        hit = self.repo
        assert hit is not None
        focused = self.focus == "main"
        info = Text()
        for label, value in (("arch", hit.arch or "—"), ("params", params(hit.params)),
                             ("context", k(hit.ctx) if hit.ctx else "—"), ("downloads", k(hit.downloads)),
                             ("likes", k(hit.likes)), ("updated", ago(hit.modified))):
            info.append(f"{label} ", style=p.dim)
            info.append(f"{value}   ", style=f"bold {p.text}")
        if hit.gated:
            info.append("gated: accept the license on the website and set hf_token", style=p.warn)
        proj = self.projector()
        mm = Text()
        if proj:
            mm.append("vision projector  ", style=p.dim)
            mm.append("[x] " if self.with_mmproj else "[ ] ", style=f"bold {p.accent}")
            mm.append(f"{proj.name}  {hub.human(proj.size)}", style=p.text if self.with_mmproj else p.dim)
            mm.append("   m toggle", style=p.dim)
        top = self.panel(Group(info, mm) if proj else info, hit.id, focused=False, height=4 if proj else 3)
        top_h = 4 if proj else 3

        rows = max(height - top_h - 4, 1)
        if self.error:
            body: RenderableType = Text(self.error, style=p.bad)
        elif self.loading_files:
            body = Text("Listing files…", style=p.info)
        elif not self.groups:
            body = Text("No .gguf model files in this repo.", style=p.dim)
        else:
            biggest = max(g.size for g in self.groups) or 1
            t = Table(box=box.SIMPLE_HEAD, border_style=p.dim, header_style=f"bold {p.accent}", expand=True,
                      padding=(0, 1), show_edge=False)
            t.add_column("", width=1, no_wrap=True)
            t.add_column("QUANT", no_wrap=True)
            t.add_column("FILE", ratio=3, no_wrap=True, overflow="ellipsis")
            t.add_column("SIZE", justify="right", no_wrap=True)
            t.add_column("", ratio=1, min_width=6, no_wrap=True)
            t.add_column("PARTS", justify="right", no_wrap=True)
            t.add_column("LOCAL", no_wrap=True)
            a, b = window(self.fsel, len(self.groups), rows)
            for i in range(a, b):
                g = self.groups[i]
                on = i == self.fsel
                state = hub.local_state(hit.id, g, self.cfg)
                queued = any(j.active and j.repo == hit.id and j.group.name == g.name for j in self.dl.jobs)
                local = Text("✔ on disk", style=p.good) if state == "done" else \
                    Text("↓ queued", style=p.accent) if queued else \
                    Text("◐ partial", style=p.warn) if state == "partial" else Text("")
                t.add_row(Text("▸" if on else " ", style=p.accent),
                          Text(g.quant, style=f"bold {p.accent2}" if on else p.accent2),
                          Text(g.name, style=f"bold {p.text}" if on else p.text),
                          hub.human(g.size), Bar(g.size / biggest, p.accent if on else p.dim, p.track),
                          str(len(g.files)) if len(g.files) > 1 else "", local,
                          style=f"on {p.track}" if on and focused else "")
            body = t
        sub = f" {len(self.groups)} files · → {hub.local_dir_for(hit.id, self.cfg)} "
        files = self.panel(body, "Files", focused=focused, height=height - top_h, subtitle=sub)
        return Group(top, files)

    def view_downloads(self, height: int) -> RenderableType:
        p = self.p
        focused = self.focus == "downloads"
        jobs = self.dl.jobs
        a, b = window(self.jsel if focused else len(jobs) - 1, len(jobs), height - 2)
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(width=1, no_wrap=True)
        t.add_column(width=1, no_wrap=True)
        t.add_column(ratio=2, no_wrap=True, overflow="ellipsis")
        t.add_column(ratio=2, no_wrap=True)
        t.add_column(no_wrap=True, justify="right", width=5)
        t.add_column(no_wrap=True, justify="right", width=20)
        t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        for i in range(a, b):
            j = jobs[i]
            on = focused and i == self.jsel
            icon, color = {"queued": ("·", p.dim), "downloading": ("↓", p.accent), "done": ("✔", p.good),
                           "failed": ("✖", p.bad), "cancelled": ("■", p.warn)}[j.status]
            name = Text()
            name.append(j.group.name, style=f"bold {p.text}" if on else p.text)
            if j.mmproj:
                name.append(" +mmproj", style=p.info)
            if j.status == "downloading" and j.speed:
                eta = (j.total - j.done) / j.speed
                note = Text(f"{hub.human(j.speed)}/s · {dur(eta)} left", style=p.accent2)
            elif j.status == "done":
                note = Text(f"profile: {j.profile}" if j.profile else "p → make profile",
                            style=p.good if j.profile else p.accent2)
            elif j.status == "failed":
                note = Text(j.error, style=p.bad)
            else:
                note = Text(j.status, style=color)
            t.add_row(Text("▸" if on else " ", style=p.accent), Text(icon, style=color), name,
                      Bar(j.frac, p.good if j.status == "done" else color, p.track),
                      Text(f"{j.frac * 100:.0f}%", style=color),
                      Text(f"{hub.human(j.done)} / {hub.human(j.total)}", style=p.dim), note)
        done = sum(1 for j in jobs if j.status == "done")
        return self.panel(t, "Downloads", focused=focused, height=height,
                          subtitle=f" {done}/{len(jobs)} done · Tab to select ")

    def hints(self) -> RenderableType:
        p = self.p
        if self.naming is not None:
            return Text.assemble(("Profile name: ", f"bold {p.accent}"), (self.name_buf, p.text), ("▌", p.accent),
                                 ("   Enter save · Esc cancel", p.dim))
        if self.confirm_quit:
            return Text.assemble(("Downloads are still running. Quit and cancel them?  ", f"bold {p.warn}"),
                                 ("y", f"bold {p.accent}"), (" quit (partial files resume next time)   ", p.text),
                                 ("any other key", f"bold {p.accent}"), (" keep going", p.text))
        if self.editing:
            items = [("type", "query"), ("Enter", "search"), ("Esc", "back to results"), ("Ctrl+U", "clear")]
        elif self.focus == "downloads":
            items = [("↑↓", "select"), ("p", "make profile"), ("x", "cancel"), ("Tab", "back"), ("q", "quit")]
        elif self.screen == "search":
            items = [("↑↓", "select"), ("Enter", "open"), ("/", "search"), ("o", "sort"), ("b", "browser"),
                     ("t", "theme"), ("q", "quit")]
        else:
            items = [("↑↓", "select"), ("Enter", "download"), ("m", "mmproj"), ("Esc", "back"), ("b", "browser"),
                     ("q", "quit")]
        if self.dl.jobs and self.focus == "main" and not self.editing:
            items.insert(-1, ("Tab", "downloads"))
        t = Text()
        for key, label in items:
            t.append(key, style=f"bold {p.accent}")
            t.append(f" {label}    ", style=p.dim)
        return t

    def statusbar(self) -> RenderableType:
        p = self.p
        t = Text()
        if self.flash and time.monotonic() < self.flash[1]:
            t.append(self.flash[0], style=f"bold {p.accent2}")
            return t
        t.append("Hub", style=f"bold {p.accent}")
        t.append(f"   |   models → {self.cfg.models_path()}", style=p.text)
        t.append("   |   token " + ("set" if hub._token(self.cfg) else "none (public repos only)"), style=p.text)
        if self.hits and self.screen == "search":
            t.append(f"   |   {len(self.hits)} repos", style=p.text)
        return t
