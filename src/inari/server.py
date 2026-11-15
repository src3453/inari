"""Starting, tracking and stopping llama-server processes.

The server runs detached with its output written to a log file, so the
monitor can be closed (``keep serving``) and re-attached later with
``inari monitor``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from inari import hfcache
from inari.config import Config, logs_dir, runs_dir
from inari.profiles import Profile


@dataclass
class RunInfo:
    name: str
    pid: int
    url: str
    log: str
    cmd: list[str]
    started: float
    create_time: float = 0.0

    @property
    def path(self) -> Path:
        return runs_dir() / f"{self.name}.json"

    def save(self) -> None:
        self.path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)

    def alive(self) -> bool:
        try:
            p = psutil.Process(self.pid)
            if self.create_time and abs(p.create_time() - self.create_time) > 2:
                return False  # pid was reused
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @classmethod
    def load(cls, path: Path) -> "RunInfo":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def active_runs(prune: bool = True) -> list[RunInfo]:
    out = []
    for f in sorted(runs_dir().glob("*.json")):
        try:
            r = RunInfo.load(f)
        except Exception:
            continue
        if r.alive():
            out.append(r)
        elif prune:
            f.unlink(missing_ok=True)
    return out


def find_run(name: str) -> RunInfo | None:
    for r in active_runs():
        if r.name == name:
            return r
    return None


def port_in_use(host: str, port: int) -> bool:
    import socket

    h = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((h, port)) == 0


def launch(profile: Profile, cfg: Config) -> tuple[RunInfo, subprocess.Popen]:
    cmd = profile.build_command(cfg)
    mp = profile.model_path(cfg)
    if mp is not None and not mp.exists() and "-hf" not in profile.args and "hf-repo" not in profile.args:
        parsed = hfcache.parse_ref(profile.model)
        if parsed:
            raise FileNotFoundError(f"{profile.model} is not in the HF cache ({hfcache.cache_dir(cfg)}) — "
                                    f"run: inari pull {parsed[0]}")
        raise FileNotFoundError(f"model not found: {mp}")
    if port_in_use(profile.host, profile.port):
        raise RuntimeError(f"port {profile.port} is already in use")
    return spawn(profile.name, cmd, profile.url(), cfg, env=profile.env)


def spawn(name: str, cmd: list[str], url: str, cfg: Config, *, env: dict[str, str] | None = None,
          log_name: str = "") -> tuple[RunInfo, subprocess.Popen]:
    """Start ``cmd`` detached, logging to ~/.inari/logs, and record it in ~/.inari/runs."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir() / f"{log_name or name}-{stamp}.log"
    log_f = open(log_path, "wb")
    log_f.write(("$ " + subprocess.list2cmdline(cmd) + "\n").encode())
    log_f.flush()

    full_env = dict(os.environ)
    if cfg.hf_cache and "LLAMA_CACHE" not in full_env:
        full_env["LLAMA_CACHE"] = str(hfcache.cache_dir(cfg))  # so -hf in llama-server shares inari's cache
    full_env.update(env or {})
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=full_env, **kwargs)
    log_f.close()  # child keeps its own handle
    try:
        ct = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        ct = 0.0
    run = RunInfo(name=name, pid=proc.pid, url=url, log=str(log_path), cmd=cmd,
                  started=time.time(), create_time=ct)
    run.save()
    _prune_logs()
    return run, proc


def stop(run: RunInfo, timeout: float = 8.0) -> bool:
    try:
        p = psutil.Process(run.pid)
        # router mode: one llama-server per loaded model. On Windows terminate() is a hard kill, so the
        # router could not clean them up itself — stop them first.
        procs = p.children(recursive=True) + [p]
    except psutil.NoSuchProcess:
        procs = []
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)
    run.remove()
    return True


def _prune_logs(keep: int = 30) -> None:
    logs = sorted(logs_dir().glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in logs[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
