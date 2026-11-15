"""Command line entry point: ``inari <command>``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.console import Console
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

BANNER = r"""[accent]  ╦╔╗╔╔═╗╦═╗╦[/]
[accent]  ║║║║╠═╣╠╦╝║[/]   [dim]llama.cpp, served golden[/]
[accent]  ╩╝╚╝╩ ╩╩╚═╩[/]   [dim]v{v}[/]"""


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
                                       load=getattr(a, "load", None) or [])
    except (FileNotFoundError, RuntimeError) as e:
        die(str(e))
        return
    for w in pre.warnings:
        console.print(f"[warn]·[/] [dim]{w}[/]")
    names = ", ".join(pre.names) if pre.names else "none — only models in the HF cache"
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
    ap = argparse.ArgumentParser(prog="inari", description="Golden TUI wrapper for llama.cpp's llama-server")
    ap.add_argument("-V", "--version", action="version", version=f"inari {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", aliases=["onboard", "wizard"],
                       help="guided setup: llama-server → model → context / GPU / chat → all args → profile")
    s.add_argument("profile", nargs="?", help="profile to edit (or the name for a new one)")
    s.add_argument("-a", "--advanced", action="store_true", help="open the all-settings table directly")
    s.set_defaults(fn=cmd_setup)

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
    s.add_argument("--no-tui", action="store_true", help="start in background and return")
    s.set_defaults(fn=cmd_serve)

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
    console.print(BANNER.format(v=__version__))
    console.print()
    for c, d in (("", "main menu: pick a profile to serve, or set one up"),
                 ("setup [PROFILE]", "guided setup wizard (start here)"), ("hub [QUERY]", "search Hugging Face + download (TUI)"),
                 ("pull REPO[:QUANT]", "download a GGUF from Hugging Face"), ("models", "list local models"),
                 ("profile new|set|show|ls", "save llama-server arguments"), ("serve [PROFILE]", "launch + monitor (no profile: router, all models)"),
                 ("monitor [PROFILE]", "re-attach to a running server"), ("ps / stop", "manage servers"),
                 ("demo", "try the monitor with a simulated server"), ("config", "settings")):
        console.print(f"  [accent]inari {c:<26}[/][dim]{d}[/]")
    if not list_profiles():
        console.print()
        console.print("  [accent2]No profiles yet — run[/] [accent]inari setup[/] [accent2]to make one.[/]")
