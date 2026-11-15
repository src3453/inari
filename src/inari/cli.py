"""Command line entry point: ``inari <command>``."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_traceback

# installed before anything else runs, so even a broken config.toml gets a readable traceback.
# INARI_DEBUG=1 also shows local variables in each frame.
install_traceback(console=Console(stderr=True), width=None, extra_lines=1,
                  show_locals=os.environ.get("INARI_DEBUG", "") not in ("", "0"))

from inari import __version__  # noqa: E402
from inari import hfcache, hub, server  # noqa: E402
from inari import theme as th  # noqa: E402
from inari.config import Config, config_path, home  # noqa: E402
from inari.profiles import Profile, list_profiles, new_profile  # noqa: E402

cfg = Config.load()
console = th.make_console(cfg.theme)




def die(msg: str, code: int = 1) -> None:
    console.print(f"[bad]✖[/] {msg}")
    sys.exit(code)


def rel_model(p: Path) -> str:
    return hub.model_ref(cfg, p)


# ───────────────────────── pull / models ─────────────────────────
def cmd_hub(a: argparse.Namespace):
    from inari.hubtui import HubApp

    app = HubApp(cfg, console, query=" ".join(getattr(a, "query", None) or []), theme_name=cfg.theme)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    if app.theme_name != cfg.theme or app.dl.backend != cfg.download_backend:
        cfg.theme, cfg.download_backend = app.theme_name, app.dl.backend
        cfg.save()
    for j in app.dl.jobs:
        if j.status == "done":
            extra = f"  → profile [accent]{j.profile}[/]" if j.profile else ""
            console.print(f"[good]✔[/] [text]{j.path}[/]{extra}")
        elif j.status in ("cancelled", "queued", "downloading"):
            console.print(f"[warn]■[/] [text]{j.group.name}[/] [dim]not finished — re-download resumes it[/]")
        elif j.status == "failed":
            console.print(f"[bad]✖[/] [text]{j.group.name}[/] [dim]{j.error}[/]")
    return app


def cmd_pull(a: argparse.Namespace) -> None:
    if not a.repo:
        return cmd_hub(argparse.Namespace(query=[]))
    try:
        model, mmproj = hub.pull(a.repo, cfg, console, yes=a.yes, backend=a.via or "")
    except Exception as e:  # network / auth / 404
        die(f"download failed: {e}")
    if model is None:
        return
    name = a.profile
    if not name and not a.yes and Confirm.ask("[accent]Create a launch profile for it?[/]", default=True, console=console):
        default = model.stem.split("-0000")[0].lower()
        name = Prompt.ask("[accent]Profile name[/]", default=default, console=console)
    if name:
        prof = new_profile(name, model=rel_model(model), description=f"from {a.repo}")
        if mmproj:
            prof.args["mmproj"] = rel_model(mmproj)
        prof.save()
        console.print(f"[good]✔ Profile[/] [accent]{name}[/] saved → [dim]{Profile.path_for(name)}[/]")
        console.print(f"  run it: [accent]inari serve {name}[/]")


def cmd_models(a: argparse.Namespace) -> None:
    if getattr(a, "action", None) == "migrate":
        return cmd_migrate(a)
    items = hub.local_models(cfg)
    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="accent", title="[accent]Local models[/]")
    t.add_column("#", style="accent2", justify="right")
    t.add_column("MODEL", style="text")
    t.add_column("SIZE", justify="right", style="text")
    t.add_column("IN", style="dim")
    for i, m in enumerate(items, 1):
        t.add_row(str(i), m.ref, hub.human(m.size), m.source)
    if not items:
        t.add_row("", "[dim]none yet — try: inari pull unsloth/Qwen3-8B-GGUF:Q4_K_M[/]", "", "")
    console.print(t)
    console.print(f"[dim]  hf  → {hfcache.cache_dir(cfg)}  (shared with llama-server -hf)[/]", soft_wrap=True)
    console.print(f"[dim]  dir → {cfg.models_path()}[/]", soft_wrap=True)
    if any(m.source == "dir" and "__" in Path(m.ref).parts[0] for m in items):
        console.print("[dim]Old inari downloads found in models_dir — [accent]inari models migrate[/] moves them "
                      "into the HF cache so llama-server -hf can use them too.[/]")


def cmd_migrate(a: argparse.Namespace) -> None:
    moved = hub.migrate(cfg, console, yes=a.yes)
    if not moved:
        return
    # profiles may point at the old files by relative or absolute path
    old = {}
    for path, ref in moved.items():
        old[path] = ref
        old[rel_model(Path(path))] = ref  # still relative to models_dir (the file is gone, the string is not)
        try:
            old[Path(path).relative_to(cfg.models_path()).as_posix()] = ref
        except ValueError:
            pass
    norm = {k.replace("\\", "/"): v for k, v in old.items()}

    def swap(v):
        if isinstance(v, str):
            return norm.get(v.replace("\\", "/"), v)
        if isinstance(v, list):
            return [swap(x) for x in v]
        return v

    for n in list_profiles():
        try:
            p = Profile.load(n)
        except Exception:
            continue
        before = (p.model, dict(p.args))
        p.model = swap(p.model)
        p.args = {k_: swap(v) for k_, v in p.args.items()}
        if (p.model, p.args) != before:
            p.save()
            console.print(f"[good]✔ profile[/] [accent]{n}[/] [dim]now uses {p.model}[/]")


# ───────────────────────── profiles ─────────────────────────
def load_profile(name: str) -> Profile:
    try:
        return Profile.load(name)
    except FileNotFoundError as e:
        die(str(e))
        raise


def show_profile(p: Profile) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="text")
    if p.description:
        t.add_row("description", p.description)
    mp = p.model_path(cfg)
    exists = mp.exists() if mp else False
    t.add_row("model", (p.model or "[dim]—[/]") + ("" if exists or not mp else "  [bad](missing)[/]"))
    t.add_row("listen", f"{p.host}:{p.port}")
    if p.server_bin:
        t.add_row("server_bin", p.server_bin)
    for k_, v in p.args.items():
        t.add_row(f"[accent2]{k_}[/]", str(v))
    if p.extra:
        t.add_row("extra", " ".join(p.extra))
    for k_, v in p.env.items():
        t.add_row(f"env.{k_}", v)
    try:
        cmd = subprocess.list2cmdline(p.build_command(cfg, binary=cfg.server_binary(p.server_bin) or "llama-server"))
    except Exception as e:
        cmd = f"(error: {e})"
    console.print(Panel(t, title=f"[accent] {p.name} [/]", title_align="left", border_style="dim", box=box.ROUNDED))
    console.print(Text(cmd, style="accent2"), soft_wrap=True)


def pick_model_interactive() -> str:
    items = hub.local_models(cfg)
    if not items:
        return Prompt.ask("[accent]Model path (.gguf)[/]", console=console)
    cmd_models(argparse.Namespace())
    idx = Prompt.ask("[accent]Model #[/] (or a path)", console=console)
    if idx.isdigit() and 1 <= int(idx) <= len(items):
        return items[int(idx) - 1].ref
    return idx


def cmd_profile(a: argparse.Namespace) -> None:
    act = a.action
    if act in (None, "list", "ls"):
        names = list_profiles()
        runs = {r.name for r in server.active_runs()}
        t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="accent", title="[accent]Profiles[/]")
        for c in ("NAME", "MODEL", "LISTEN", "ARGS", "STATUS"):
            t.add_column(c, style="text", overflow="fold")
        for n in names:
            try:
                p = Profile.load(n)
            except Exception as e:
                t.add_row(n, f"[bad]{e}[/]", "", "", "")
                continue
            args = " ".join(f"{k_}={v}" for k_, v in p.args.items())
            t.add_row(f"[accent]{n}[/]", p.model, f"{p.host}:{p.port}", args,
                      "[good]● serving[/]" if n in runs else "[dim]stopped[/]")
        if not names:
            t.add_row("[dim]no profiles — inari profile new <name>[/]", "", "", "", "")
        console.print(t)
        return

    if act == "new":
        if Profile.path_for(a.name).exists() and not a.force:
            die(f"profile '{a.name}' exists (use --force to overwrite)")
        model = a.model or pick_model_interactive()
        p = new_profile(a.name, model=model)
        if a.port:
            p.port = a.port
        for kv in a.set or []:
            k_, _, v = kv.partition("=")
            p.set(k_, v)
        p.save()
        console.print(f"[good]✔ created[/] [accent]{a.name}[/]")
        show_profile(p)
        return

    p = load_profile(a.name)
    if act == "show":
        show_profile(p)
    elif act == "set":
        for kv in a.pairs:
            if "=" not in kv:
                die(f"expected key=value, got {kv!r}")
            k_, _, v = kv.partition("=")
            p.set(k_.strip(), v)
        p.save()
        show_profile(p)
    elif act == "unset":
        for k_ in a.keys:
            if not p.unset(k_):
                console.print(f"[warn]· {k_} was not set[/]")
        p.save()
        show_profile(p)
    elif act == "copy":
        p.name = a.new_name
        p.save()
        console.print(f"[good]✔ copied →[/] [accent]{a.new_name}[/]")
    elif act == "rm":
        if a.yes or Confirm.ask(f"Delete profile [accent]{p.name}[/]?", default=False, console=console):
            Profile.path_for(p.name).unlink()
            console.print(f"[good]✔ deleted[/] {p.name}")
    elif act == "edit":
        editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
        subprocess.call([editor, str(Profile.path_for(p.name))])
        show_profile(Profile.load(p.name))
    elif act == "cmd":
        try:
            console.print(subprocess.list2cmdline(p.build_command(cfg)), soft_wrap=True, markup=False, highlight=False)
        except FileNotFoundError as e:
            die(str(e))


# ───────────────────────── setup wizard ─────────────────────────
def cmd_tune(a: argparse.Namespace) -> None:
    """Benchmark candidate settings for a profile with the real llama-server and keep the fastest."""
    import time

    from rich.live import Live

    from inari import argspec, autotune, tuning
    from inari.gguf import read_info

    prof = load_profile(a.profile)
    binary = cfg.server_binary(prof.server_bin)
    if not binary:
        die("llama-server not found — run: inari config set llama_server <path>")
    mp = prof.model_path(cfg)
    if not mp or not mp.exists():
        die(f"model not found: {prof.model}")
    if server.find_run(prof.name):
        die(f"{prof.name} is running — stop it first (inari stop {prof.name}) so the GPU memory is free")
    others = server.active_runs()
    if others:
        console.print(f"[warn]⚠ other servers are running ({', '.join(r.name for r in others)}) — they use GPU "
                      "memory and CPU, so the results will be off. Stop them for best results.[/]")
    with console.status("[accent]Reading llama-server settings and devices…[/]"):
        cat = argspec.load(cfg, binary)
        hw = tuning.detect(binary)
    size = next((m.size for m in hub.local_models(cfg) if m.ref == prof.model), 0) or mp.stat().st_size
    tuner = autotune.Tuner(cfg, cat, hw, read_info(mp), size, prof, binary=binary, goal=a.goal,
                           depth="thorough" if a.thorough else "quick")
    console.print(f"[dim]devices: {hw.describe()} · up to {tuner.max_trials} server starts · "
                  f"log: {tuner.log_path}[/]")
    p = th.get(cfg.theme)
    tuner.start()
    try:
        with Live(autotune.render(tuner, p, rows=40), console=console, refresh_per_second=4) as live:
            while not tuner.done:
                time.sleep(0.25)
                live.update(autotune.render(tuner, p, rows=40))
            live.update(autotune.render(tuner, p, rows=40))
    except KeyboardInterrupt:
        tuner.cancel()
        console.print("[dim]cancelling…[/]")
        while not tuner.done:
            time.sleep(0.2)
    console.print()
    ch = tuner.changes()
    if tuner.best is None or not ch:
        console.print("[dim]nothing to change — the profile is already the fastest of what was tried[/]"
                      if tuner.best else "[bad]no candidate ran successfully[/]")
        return
    console.print("changes: " + "  ".join(f"[accent2]--{k}[/] {'(default)' if v is None else v}"
                                          for k, v in ch.items()))
    if a.yes or Confirm.ask(f"[accent]Save these to profile {prof.name}?[/]", console=console, default=True):
        vals = argspec.from_args(prof.args, cat)
        for k, v in ch.items():
            if v is None:
                vals.pop(k, None)
            else:
                vals[k] = v
        prof.args = argspec.to_args(vals, cat)
        prof.save()
        console.print(f"[good]✔ saved[/] [accent]{prof.name}[/]")


def cmd_setup(a: argparse.Namespace) -> None:
    from inari.wizard import run_wizard

    res = run_wizard(cfg, console, name=a.profile or "", start="advanced" if a.advanced else "")
    if res is None:
        console.print("[dim]setup cancelled — nothing saved[/]")
        return
    console.print(f"[good]✔ Profile[/] [accent]{res.name}[/] saved → [dim]{res.path}[/]")
    show_profile(Profile.load(res.name))
    if res.serve:
        cmd_serve(argparse.Namespace(profile=res.name, port=None, no_tui=False))
    else:
        console.print(f"  run it: [accent]inari serve {res.name}[/]   ·   change it: [accent]inari setup {res.name}[/]")


# ───────────────────────── import a llama-server command ─────────────────────────
def read_command_interactive() -> str:
    console.print("[accent]Paste your llama-server command[/] [dim](line continuations \\ ^ ` are fine; "
                  "empty line to finish)[/]")
    lines: list[str] = []
    while True:
        try:
            line = console.input("[accent2]› [/]" if not lines else "[dim]… [/]")
        except EOFError:
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


def import_command(text: str, *, name: str = "", yes: bool = False, via: str = "", wizard: bool = True):
    """Parse ``text``, download the model if needed, then open the setup wizard on Review (or save directly).
    Returns the wizard Result, the saved Profile (``wizard=False``) or None."""
    from inari import argspec, importer
    from inari.wizard import free_port, run_wizard

    if not text.strip():
        console.print("[dim]nothing pasted — nothing imported[/]")
        return None
    tokens, _ = importer.split_statements(text)
    binary = ""
    src_bin, program, _ = importer.program_of(tokens)
    if src_bin and Path(src_bin).stem.lower().startswith("llama-server"):  # only llama-server itself runs a profile
        cand = Path(src_bin).expanduser()
        binary = str(cand) if cand.is_file() else (shutil.which(src_bin) or "")
    with console.status("[accent]Reading llama-server settings…[/]"):
        cat = argspec.load(cfg, binary or None)
    imp = importer.parse(text, cat, cfg)

    # ── what we understood ──
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", no_wrap=True)
    t.add_column(style="text", overflow="fold")
    t.add_row("from", (imp.binary + (f" {imp.program}" if Path(imp.binary).stem.lower() == "llama" else ""))
              if imp.binary else "—")
    if imp.hf:
        target = imp.hf.repo + (f":{imp.hf.quant}" if imp.hf.quant else "") + (f"  {imp.hf.file}" if imp.hf.file else "")
        t.add_row("model", f"{imp.model or target}" + ("" if imp.model else "  [warn](not downloaded yet)[/]"))
    else:
        t.add_row("model", (imp.model or "[bad]none[/]") +
                  ("" if not imp.model or imp.model_path else "  [bad](missing)[/]"))
    if imp.host or imp.port:
        t.add_row("listen", f"{imp.host or '127.0.0.1'}:{imp.port or '(free port)'}")
    for k_, v in imp.args.items():
        t.add_row(f"[accent2]{k_}[/]", str(v))
    if imp.extra:
        t.add_row("extra", " ".join(imp.extra))
    for k_, v in imp.env.items():
        t.add_row(f"env.{k_}", v)
    console.print(Panel(t, title="[accent] imported [/]", title_align="left", border_style="dim", box=box.ROUNDED))
    for n in imp.notes:
        console.print(f"  [warn]·[/] [dim]{n}[/]", soft_wrap=True)

    # ── fetch the model when the command used -hf / a Hugging Face URL ──
    if imp.needs_fetch:
        try:
            with console.status(f"[accent]Looking up[/] [text]{imp.hf.repo}[/] …"):
                pl = importer.plan(cfg, imp)
        except Exception as e:  # network / auth / 404 / no match
            console.print(f"[bad]✖[/] could not look up the model: {e}")
            pl = None
        if pl and len(pl.choices) > 1 and not yes:
            for i, g in enumerate(pl.choices, 1):
                console.print(f"  [accent2]{i:>2}[/] {g.name}  [dim]{g.quant} · {hub.human(g.size)}[/]")
            idx = Prompt.ask("[accent]Several files match — which #[/]",
                             choices=[str(i) for i in range(1, len(pl.choices) + 1)], default="1",
                             show_choices=False, console=console)
            pl.model = pl.choices[int(idx) - 1]
        if pl and importer.link_existing(cfg, imp, pl):
            console.print(f"[good]✔[/] already in the HF cache: [text]{imp.model}[/]")
        elif pl:
            what = pl.model.name + (f" + {pl.mmproj.name}" if pl.mmproj else "")
            console.print(f"  [dim]→ {hfcache.repo_dir(cfg, pl.repo)}[/]", soft_wrap=True)
            if yes or Confirm.ask(f"[accent]Download[/] [text]{what}[/] ([accent2]{hub.human(pl.size)}[/])?",
                                  default=True, console=console):
                try:
                    importer.fetch(cfg, imp, pl, console, backend=via)
                    console.print(f"[good]✔ Downloaded[/] [text]{imp.model}[/]")
                except Exception as e:
                    console.print(f"[bad]✖[/] download failed: {e}")
        if not imp.model:
            console.print("[warn]·[/] [dim]no model yet — pick or download one in the wizard[/]")

    # ── the profile ──
    prof = new_profile(name or imp.suggested_name(), model=imp.model, description="imported from a command")
    prof.args = dict(imp.args)
    prof.extra, prof.env = list(imp.extra), dict(imp.env)
    prof.host = imp.host or prof.host
    prof.port = imp.port or free_port()
    if binary and not (cfg.server_binary() and importer.same_binary(binary, cfg.server_binary())):
        if cfg.server_binary():
            prof.server_bin = binary  # this command used another build: keep using it for this profile
        else:
            cfg.llama_server = binary
            cfg.save()
            console.print(f"[good]✔[/] llama_server set to [text]{binary}[/]")

    if not wizard:
        if not prof.model:
            die("no model — run again without -y to pick one in the wizard")
        base, n = prof.name, 2
        while Profile.path_for(prof.name).exists() and not name:
            prof.name, n = f"{base}-{n}", n + 1
        prof.save()
        console.print(f"[good]✔ Profile[/] [accent]{prof.name}[/] saved → [dim]{Profile.path_for(prof.name)}[/]")
        show_profile(prof)
        return prof
    if not yes:
        console.input("[dim]Enter to review it in the setup wizard[/] ")
    return run_wizard(cfg, console, seed=prof, start="review" if prof.model else "model",
                      origin="imported from a llama-server command")


def cmd_import(a: argparse.Namespace) -> None:
    if a.file:
        try:
            text = Path(a.file).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            die(f"cannot read {a.file}: {e}")
            return
    elif a.command:
        text = a.command[0] if len(a.command) == 1 else subprocess.list2cmdline(a.command)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = read_command_interactive()
    res = import_command(text, name=a.profile or "", yes=a.yes, via=a.via or "", wizard=not a.yes)
    if res is None:
        console.print("[dim]import cancelled — nothing saved[/]")
        return
    if isinstance(res, Profile):
        console.print(f"  run it: [accent]inari serve {res.name}[/]")
        return
    console.print(f"[good]✔ Profile[/] [accent]{res.name}[/] saved → [dim]{res.path}[/]")
    show_profile(Profile.load(res.name))
    if res.serve:
        cmd_serve(argparse.Namespace(profile=res.name, port=None, no_tui=False))
    else:
        console.print(f"  run it: [accent]inari serve {res.name}[/]   ·   change it: [accent]inari setup {res.name}[/]")


# ───────────────────────── serve / monitor ─────────────────────────
def run_monitor(collector, *, managed: bool, run: server.RunInfo | None = None) -> None:
    from inari.tui import App

    collector.start()
    app = App(collector, theme_name=cfg.theme, managed=managed, console=console)
    try:
        result = app.run()
    except KeyboardInterrupt:
        result = "keep" if managed else "quit"
    collector.stop()
    if app.theme_name != cfg.theme:
        cfg.theme = app.theme_name
        cfg.save()
    if run is None:
        return
    label = display_name(run.name)
    if result == "stop":
        with console.status("[accent]Stopping llama-server…[/]"):
            server.stop(run)
        console.print(f"[good]✔ stopped[/] [accent]{label}[/]")
    elif run.alive():
        console.print(f"[accent]●[/] [text]{label}[/] keeps serving at [accent2]{run.url}[/]  "
                      f"[dim](inari monitor {label} · inari stop {label})[/]")


def display_name(run_name: str) -> str:
    from inari import router

    return "router" if run_name == router.NAME else run_name


def run_name(name: str) -> str:
    """CLI spelling -> RunInfo name (`router` -> `@router`, unless a profile is really called that)."""
    from inari import router

    if name in ("router", router.NAME) and not Profile.path_for("router").exists():
        return router.NAME
    return name


def make_collector(run: server.RunInfo, popen=None):
    from inari import router
    from inari.collector import Collector, RouterCollector

    cls = RouterCollector if run.name == router.NAME else Collector
    return cls(run.url, log_path=run.log, pid=run.pid, name=display_name(run.name), cmd=run.cmd,
               started=run.started, popen=popen)


def cmd_serve(a: argparse.Namespace) -> None:
    if not getattr(a, "profile", None):
        return cmd_serve_router(a)
    p = load_profile(a.profile)
    if a.port:
        p.port = a.port
    existing = server.find_run(p.name)
    if existing:
        console.print(f"[accent]●[/] {p.name} is already running (pid {existing.pid}) — attaching")
        return attach(existing, a.no_tui)
    try:
        run, proc = server.launch(p, cfg)
    except (FileNotFoundError, RuntimeError) as e:
        die(str(e))
        return
    console.print(f"[accent]▶[/] started [accent]{p.name}[/] pid {run.pid} → [accent2]{run.url}[/]  [dim]log: {run.log}[/]")
    if a.no_tui:
        return
    run_monitor(make_collector(run, proc), managed=True, run=run)


def cmd_serve_router(a: argparse.Namespace) -> None:
    """`inari serve` with no profile: one llama-server in router mode, every profile a model preset."""
    from inari import router

    existing = server.find_run(router.NAME)
    if existing:
        console.print(f"[accent]●[/] router is already running (pid {existing.pid}) at "
                      f"[accent2]{existing.url}[/] — attaching")
        return attach(existing, getattr(a, "no_tui", False))
    host = getattr(a, "host", None) or "127.0.0.1"
    port = getattr(a, "port", None) or 8080
    try:
        run, proc, pre = router.launch(cfg, host=host, port=port, models_max=getattr(a, "models_max", None),
                                       load=getattr(a, "load", None) or [],
                                       allow_no_profile_models=getattr(a, "allow_no_profile_models", False))
    except (FileNotFoundError, RuntimeError) as e:
        die(str(e))
        return
    for w in pre.warnings:
        console.print(f"[warn]·[/] [dim]{w}[/]")
    allow = getattr(a, "allow_no_profile_models", False)
    names = ", ".join(pre.names) if pre.names else ("none — only models in the HF cache" if allow else "none")
    console.print(f"[accent]▶[/] started [accent]router[/] pid {run.pid} → [accent2]{run.url}[/]  "
                  f"[dim]profiles: {names}[/]")
    console.print(f"  [dim]pick a model per request with \"model\": \"<profile>\" · log: {run.log}[/]", soft_wrap=True)
    if getattr(a, "no_tui", False):
        return
    run_monitor(make_collector(run, proc), managed=True, run=run)


def attach(run: server.RunInfo, no_tui: bool = False) -> None:
    if no_tui:
        return
    run_monitor(make_collector(run), managed=True, run=run)


def cmd_monitor(a: argparse.Namespace) -> None:
    from inari.collector import Collector

    if a.url:
        return run_monitor(Collector(a.url, name=a.url), managed=False)
    runs = server.active_runs()
    if a.profile:
        run = next((r for r in runs if r.name == run_name(a.profile)), None)
        if not run and run_name(a.profile) != a.profile:
            die("the router is not running — start it with: inari serve")
        if not run:
            p = load_profile(a.profile)
            console.print(f"[warn]{a.profile} wasn't started by inari — read-only monitor of {p.url()}[/]")
            return run_monitor(Collector(p.url(), name=p.name), managed=False)
        return attach(run)
    if not runs:
        console.print("[warn]No inari-managed servers running — monitoring http://127.0.0.1:8080 read-only[/]")
        return run_monitor(Collector("http://127.0.0.1:8080"), managed=False)
    if len(runs) == 1:
        return attach(runs[0])
    for i, r in enumerate(runs, 1):
        console.print(f"  [accent2]{i}[/] {display_name(r.name)}  [dim]{r.url}[/]")
    idx = Prompt.ask("[accent]Attach to #[/]", choices=[str(i) for i in range(1, len(runs) + 1)], console=console)
    attach(runs[int(idx) - 1])


# ───────────────────────── cli (chat in the terminal) ─────────────────────────
def pick_profile_interactive(prompt: str = "Chat with #") -> str:
    names = list_profiles()
    if not names:
        die("no profiles yet — make one with: inari setup")
    if len(names) == 1:
        return names[0]
    runs = {r.name for r in server.active_runs()}
    for i, n in enumerate(names, 1):
        console.print(f"  [accent2]{i:>2}[/] {n}" + ("  [good]● serving[/]" if n in runs else ""))
    idx = Prompt.ask(f"[accent]{prompt}[/]", choices=[str(i) for i in range(1, len(names) + 1)],
                     show_choices=False, console=console)
    return names[int(idx) - 1]


def effort_arg(v: str) -> str:
    """argparse type for ``--effort``: validate now, parse again in cmd_cli (chat is imported lazily)."""
    from inari.chat import parse_effort
    try:
        parse_effort(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid value {v!r}: a token count (e.g. 2048, 8k), low/medium/high/max, off or auto") from None
    return v


def cmd_cli(a: argparse.Namespace) -> None:
    """Chat in the terminal like llama-cli. Uses a server that already serves the profile (on its own or via
    the router); otherwise starts llama-server for the chat and stops it again when you leave."""
    from inari import chat, router

    params: dict = {}
    if getattr(a, "temp", None) is not None:
        params["temperature"] = a.temp
    if getattr(a, "n_predict", None):
        params["max_tokens"] = a.n_predict
    prompt = getattr(a, "prompt", None)
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    system = getattr(a, "system", None) or ""

    started: server.RunInfo | None = None
    if getattr(a, "url", None):
        url, model, label, alive, log = a.url, getattr(a, "model", None) or "", a.url, None, None
    else:
        name = getattr(a, "profile", None)
        if not name:
            runs = [r for r in server.active_runs() if r.name != router.NAME]
            name = runs[0].name if len(runs) == 1 and not server.find_run(router.NAME) else pick_profile_interactive()
        p = load_profile(name)
        own, rt = server.find_run(p.name), server.find_run(router.NAME)
        if own:
            url, model, alive, log = own.url, "", own.alive, own.log
            console.print(f"[accent]●[/] {p.name} is already serving at [accent2]{own.url}[/] — using it")
        elif rt and not getattr(a, "no_router", False):
            url, model, alive, log = rt.url, p.name, rt.alive, rt.log
            console.print(f"[accent]●[/] using the running router at [accent2]{rt.url}[/] "
                          f"[dim](model {p.name} loads on the first message)[/]")
        else:
            if getattr(a, "port", None):
                p.port = a.port
            try:
                started, _ = server.launch(p, cfg)
            except (FileNotFoundError, RuntimeError) as e:
                die(str(e))
                return
            url, model, alive, log = started.url, "", started.alive, started.log
            console.print(f"[accent]▶[/] started [accent]{p.name}[/] pid {started.pid} → [accent2]{started.url}[/]  "
                          f"[dim]log: {started.log}[/]", soft_wrap=True)
        label = p.name
    c = chat.Chat(console, url, model=model, label=label, system=system, params=params,
                  # piped one-shot output (`inari cli x -p … > out.txt`) gets the answer only
                  show_reasoning=not getattr(a, "no_think", False) and (prompt is None or sys.stdout.isatty()),
                  markdown=not getattr(a, "plain", False))
    if getattr(a, "effort", None):
        c.effort = chat.parse_effort(a.effort)
    ok = True
    try:
        chat.wait_ready(console, url, alive=alive, log=log, label=label)
        if prompt is not None:
            ok = c.once(prompt)
        else:
            c.loop()
    except chat.ChatError as e:
        console.print(f"[bad]✖[/] {escape(str(e))}")
        ok = False
    except KeyboardInterrupt:
        console.print("[dim]cancelled[/]")
    finally:
        if started is not None:
            if getattr(a, "keep", False) and started.alive():
                console.print(f"[accent]●[/] [text]{started.name}[/] keeps serving at [accent2]{started.url}[/]  "
                              f"[dim](inari monitor {started.name} · inari stop {started.name})[/]")
            else:
                with console.status("[accent]Stopping llama-server…[/]"):
                    server.stop(started)
    if not ok and prompt is not None:
        sys.exit(1)


def cmd_ps(a: argparse.Namespace) -> None:
    import time

    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="accent", title="[accent]Running servers[/]")
    for c in ("PROFILE", "PID", "URL", "UPTIME", "LOG"):
        t.add_column(c, style="text")
    for r in server.active_runs():
        up = int(time.time() - r.started)
        t.add_row(f"[accent]{display_name(r.name)}[/]", str(r.pid), r.url, f"{up // 3600}h{up % 3600 // 60:02d}m", f"[dim]{r.log}[/]")
    console.print(t)


def cmd_stop(a: argparse.Namespace) -> None:
    runs = server.active_runs()
    targets = runs if a.all else [r for r in runs if a.profile and r.name == run_name(a.profile)]
    if not targets:
        die("nothing to stop" if a.all or not a.profile else f"'{a.profile}' is not running")
    for r in targets:
        server.stop(r)
        console.print(f"[good]✔ stopped[/] [accent]{display_name(r.name)}[/] (pid {r.pid})")


def cmd_demo(a: argparse.Namespace) -> None:
    from inari.demo import DemoCollector

    run_monitor(DemoCollector(), managed=False)


def cmd_config(a: argparse.Namespace) -> None:
    if a.action == "set":
        if not hasattr(cfg, a.key):
            die(f"unknown key {a.key} (llama_server, models_dir, hf_cache, download_backend, theme, hf_token)")
        if a.key == "download_backend" and a.value not in hub.BACKENDS:
            die(f"download_backend must be one of {', '.join(hub.BACKENDS)}")
        setattr(cfg, a.key, a.value)
        cfg.save()
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="text")
    t.add_row("home", str(home()))
    t.add_row("config", str(config_path()))
    t.add_row("llama_server", f"{cfg.llama_server or '(PATH)'}  →  {cfg.server_binary() or '[bad]not found[/]'}")
    t.add_row("hf_cache", f"{hfcache.cache_dir(cfg)}  [dim]({'set' if cfg.hf_cache else 'same lookup as llama.cpp'})[/]")
    t.add_row("models_dir", str(cfg.models_path()))
    xet = "hf_xet installed" if hub.xet_available() else "hf_xet not installed → http"
    t.add_row("download_backend", f"{cfg.download_backend}  [dim]({', '.join(hub.BACKENDS)} · {xet})[/]")
    t.add_row("theme", f"{cfg.theme}  [dim]({', '.join(th.ORDER)})[/]")
    t.add_row("hf_token", "set" if cfg.hf_token else ("HF_TOKEN env" if os.environ.get("HF_TOKEN") else "—"))
    console.print(Panel(t, title="[accent] config [/]", title_align="left", border_style="dim", box=box.ROUNDED))


# ───────────────────────── parser ─────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="inari", description="A helpful, swift, golden TUI wrapper for llama.cpp's llama-server")
    ap.add_argument("-V", "--version", action="version", version=f"inari {__version__}")
    ap.add_argument("--nologo", action="store_true", help="show the logo as plain ASCII text")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", aliases=["onboard", "wizard"],
                       help="setup & models wizard: llama-server → model (pick / download / delete) → context / GPU / chat → all args → profile")
    s.add_argument("profile", nargs="?", help="profile to edit (or the name for a new one)")
    s.add_argument("-a", "--advanced", action="store_true", help="open the all-settings table directly")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("tune", aliases=["autotune"],
                       help="benchmark settings for a profile with the real llama-server and keep the fastest")
    s.add_argument("profile")
    s.add_argument("-g", "--goal", choices=["balanced", "gen", "prompt"], default="balanced",
                   help="what to optimize (default: balanced = a typical chat request)")
    s.add_argument("--thorough", action="store_true", help="more candidates and repeats (slower)")
    s.add_argument("-y", "--yes", action="store_true", help="save the result without asking")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("import", help="turn an existing llama-server command into a profile (downloads -hf models)")
    s.add_argument("command", nargs=argparse.REMAINDER,
                   help='the command, quoted or as-is: inari import llama-server -hf unsloth/Qwen3-8B-GGUF -c 32768 '
                        '(omit to paste it, or pipe it in)')
    s.add_argument("-f", "--file", help="read the command from a script (.sh / .bat / .ps1)")
    s.add_argument("-p", "--profile", help="profile name (default: from the model file)")
    s.add_argument("-y", "--yes", action="store_true", help="download without asking and save without the wizard")
    s.add_argument("--via", choices=hub.BACKENDS, help="download backend (default: config download_backend)")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("pull", help="download a GGUF from Hugging Face (repo or repo:QUANT)")
    s.add_argument("repo", nargs="?", help="e.g. unsloth/Qwen3-8B-GGUF or unsloth/Qwen3-8B-GGUF:Q4_K_M "
                   "(omit to browse in the hub TUI)")
    s.add_argument("-p", "--profile", help="create a launch profile with this name")
    s.add_argument("-y", "--yes", action="store_true", help="no prompts (also downloads mmproj)")
    s.add_argument("--via", choices=hub.BACKENDS, help="download backend (default: config download_backend)")
    s.set_defaults(fn=cmd_pull)

    s = sub.add_parser("hub", aliases=["search", "browse"], help="search Hugging Face and download GGUFs in a TUI")
    s.add_argument("query", nargs="*", help="initial search, e.g. qwen3 8b")
    s.set_defaults(fn=cmd_hub)

    s = sub.add_parser("models", aliases=["ls"], help="list local models (HF cache + models_dir)")
    s.add_argument("action", nargs="?", choices=["list", "migrate"],
                   help="migrate: move old inari downloads into the HF cache and update profiles")
    s.add_argument("-y", "--yes", action="store_true")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("profile", aliases=["p"], help="manage launch profiles")
    ps = s.add_subparsers(dest="action")
    ps.add_parser("list", aliases=["ls"])
    x = ps.add_parser("new", help="create a profile")
    x.add_argument("name")
    x.add_argument("-m", "--model")
    x.add_argument("--port", type=int)
    x.add_argument("-s", "--set", action="append", metavar="KEY=VALUE")
    x.add_argument("-f", "--force", action="store_true")
    for name in ("show", "edit", "cmd"):
        ps.add_parser(name).add_argument("name")
    x = ps.add_parser("set", help="set values: ctx-size=32768 port=8081 flash-attn=on extra+=--no-webui")
    x.add_argument("name")
    x.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    x = ps.add_parser("unset")
    x.add_argument("name")
    x.add_argument("keys", nargs="+")
    x = ps.add_parser("copy")
    x.add_argument("name")
    x.add_argument("new_name")
    x = ps.add_parser("rm")
    x.add_argument("name")
    x.add_argument("-y", "--yes", action="store_true")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("serve", aliases=["run"],
                       help="start llama-server and open the monitor (no profile: router mode, all profiles)")
    s.add_argument("profile", nargs="?",
                   help="profile to serve on its own; omit to start llama-server's router with every profile")
    s.add_argument("--port", type=int, help="listen port (router default: 8080)")
    s.add_argument("--host", help="router: listen address (default 127.0.0.1)")
    s.add_argument("-l", "--load", action="append", metavar="PROFILE", help="router: load this model at startup")
    s.add_argument("--models-max", type=int, metavar="N",
                   help="router: max models loaded at once (llama.cpp default 4, 0 = unlimited)")
    s.add_argument("--allow-no-profile-models", action="store_true",
                   help="router: also list models in the HF cache that have no profile")
    s.add_argument("--no-tui", action="store_true", help="start in background and return")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("cli", aliases=["chat"],
                       help="chat with a profile's model in the terminal, like llama-cli")
    s.add_argument("profile", nargs="?", help="profile to chat with (omit to pick one)")
    s.add_argument("-p", "--prompt", help="answer this one prompt and exit (stdin works too)")
    s.add_argument("-s", "--system", help="system prompt")
    s.add_argument("--temp", type=float, help="temperature for this chat (default: the profile's)")
    s.add_argument("-n", "--n-predict", type=int, metavar="N", help="max tokens per reply")
    s.add_argument("--no-think", action="store_true", help="hide the model's reasoning (/think toggles it)")
    s.add_argument("-e", "--effort", type=effort_arg, metavar="N|LEVEL",
                   help="thinking budget: tokens (2048, 8k), low/medium/high/max, off or auto (/effort changes it)")
    s.add_argument("--plain", action="store_true",
                   help="show replies as plain text instead of rendered Markdown (/md toggles it)")
    s.add_argument("--keep", action="store_true",
                   help="keep the llama-server started for the chat running afterwards")
    s.add_argument("--port", type=int, help="listen port when a server is started for the chat")
    s.add_argument("--no-router", action="store_true",
                   help="start the profile on its own even if the router is running")
    s.add_argument("--url", help="chat with any running llama-server, e.g. http://127.0.0.1:8080")
    s.add_argument("-m", "--model", help="with --url: the model name to request (router)")
    s.set_defaults(fn=cmd_cli)

    s = sub.add_parser("monitor", aliases=["mon", "top"], help="open the monitor for a running server")
    s.add_argument("profile", nargs="?")
    s.add_argument("--url", help="monitor any llama-server read-only, e.g. http://127.0.0.1:8080")
    s.set_defaults(fn=cmd_monitor)

    s = sub.add_parser("ps", help="list servers started by inari")
    s.set_defaults(fn=cmd_ps)

    s = sub.add_parser("stop", help="stop a server started by inari")
    s.add_argument("profile", nargs="?", help="profile name, or `router`")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("demo", help="explore the monitor with a simulated server")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("config", help="show or change global settings")
    cs = s.add_subparsers(dest="action")
    cs.add_parser("show")
    x = cs.add_parser("set")
    x.add_argument("key")
    x.add_argument("value")
    s.set_defaults(fn=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> None:
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    ap = build_parser()
    a, unknown = ap.parse_known_args(argv)
    if unknown:
        # allow raw llama.cpp flags as keys: `inari profile set x -ot=exps=CPU`
        if getattr(a, "cmd", None) in ("profile", "p") and getattr(a, "action", None) == "set":
            a.pairs += unknown
        else:
            ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    if a.nologo:
        from inari import inari_logo

        inari_logo.ascii_only = True
    if not getattr(a, "fn", None):
        if sys.stdin.isatty() and sys.stdout.isatty():
            return main_menu()
        print_help()
        return
    try:
        a.fn(a)
    except KeyboardInterrupt:
        console.print("[dim]interrupted[/]")


# ───────────────────────── main menu (bare `inari`) ─────────────────────────
def main_menu() -> None:
    from inari.menu import Menu

    select, flash = "", ""
    while True:
        try:
            action, name = Menu(cfg, console, select=select, flash=flash).run()
        except KeyboardInterrupt:
            return
        select, flash = name, ""
        if action == "quit":
            return
        try:
            if action == "serve":
                cmd_serve(argparse.Namespace(profile=name, port=None, no_tui=False))
            elif action == "chat":
                console.clear()
                cmd_cli(argparse.Namespace(profile=name))
            elif action.startswith("tune:"):
                mode = action.split(":", 1)[1]
                console.clear()
                cmd_tune(argparse.Namespace(profile=name, goal="balanced" if mode == "thorough" else mode,
                                            thorough=mode == "thorough", yes=False))
                console.input("[dim]Enter to return to the menu[/] ")
            elif action == "router":
                cmd_serve(argparse.Namespace(profile=None, port=None, no_tui=False))
            elif action in ("new", "edit"):
                from inari.wizard import run_wizard

                res = run_wizard(cfg, console, name=name)
                if res is None:
                    flash = "setup cancelled — nothing saved"
                else:
                    select, flash = res.name, f"✔ profile {res.name} saved"
                    if res.serve:
                        cmd_serve(argparse.Namespace(profile=res.name, port=None, no_tui=False))
            elif action == "import":
                text = read_command_interactive()
                res = import_command(text) if text.strip() else None
                if res is None:
                    flash = "import cancelled — nothing saved"
                else:
                    select, flash = res.name, f"✔ profile {res.name} imported"
                    if res.serve:
                        cmd_serve(argparse.Namespace(profile=res.name, port=None, no_tui=False))
            elif action == "stop":
                run = server.find_run(name)
                if run:
                    with console.status(f"[accent]Stopping {name}…[/]"):
                        server.stop(run)
                    flash = f"✔ stopped {name}"
            elif action == "hub":
                app = cmd_hub(argparse.Namespace(query=[]))
                done = [j for j in app.dl.jobs if j.status == "done"]
                made = [j.profile for j in done if j.profile]
                if made:
                    select, flash = made[-1], f"✔ downloaded — profile {made[-1]} created"
                elif done:
                    flash = f"✔ downloaded {len(done)} model(s) — n to set up a profile for it"
            elif action == "demo":
                cmd_demo(argparse.Namespace())
        except KeyboardInterrupt:
            pass
        except SystemExit:  # die() — the error is already printed above
            try:
                console.input("[dim]Enter to return to the menu[/] ")
            except (KeyboardInterrupt, EOFError):
                return


def print_help() -> None:
    from inari.inari_logo import banner

    console.print(banner(th.get(cfg.theme), console))
    console.print()
    for c, d in (("", "main menu: pick a profile to serve or chat with, or set one up"),
                 ("setup [PROFILE]", "setup & models wizard: new profile, manage local models (start here)"),
                 ("import [COMMAND]", "llama-server command → profile (+ model download)"), ("hub [QUERY]", "search Hugging Face + download (TUI)"),
                 ("pull REPO[:QUANT]", "download a GGUF from Hugging Face"), ("models", "list local models"),
                 ("profile new|set|show|ls", "save llama-server arguments"),
                 ("tune PROFILE", "benchmark & pick the fastest GPU/batch/thread settings"), ("serve [PROFILE]", "launch + monitor (no profile: router, all models)"),
                 ("cli [PROFILE]", "chat with a model in the terminal, like llama-cli"),
                 ("monitor [PROFILE]", "re-attach to a running server"), ("ps / stop", "manage servers"),
                 ("demo", "try the monitor with a simulated server"), ("config", "settings")):
        console.print(f"  [accent]inari {c:<26}[/][dim]{d}[/]")
    if not list_profiles():
        console.print()
        console.print("  [accent2]No profiles yet — run[/] [accent]inari setup[/] [accent2]to make one.[/]")
