"""Keep llama.cpp up to date with the official prebuilt releases from GitHub.

Builds are installed side by side under ``~/.inari/llama.cpp/<tag>-<variant>/`` (e.g. ``b12420-cuda``),
so a running llama-server keeps its files while the next build is unpacked. Only binaries inside
that folder are "managed": inari switches ``llama_server`` (and profiles' ``server_bin``) to a new
managed build by itself, but never replaces a build you pointed it at without asking.

The variant (cuda / rocm / vulkan / sycl / cpu …) follows the backend of the build in use, or
``update_variant`` in the config.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from inari.config import Config, home

REPO = "ggml-org/llama.cpp"
API = f"https://api.github.com/repos/{REPO}/releases"
CHECK_EVERY = 24 * 3600  # automatic checks: at most once a day
KEEP = 2  # managed builds kept on disk: the current one and the one before it
MODES = ("off", "notify", "install")
VARIANTS = ("cuda", "rocm", "vulkan", "sycl", "cpu")  # or an exact one: cuda-12.4, rocm-10.0, …
ASSET_RE = re.compile(r"^llama-(b\d+)-bin-(.+)\.(zip|tar\.gz)$")
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


# ───────────────────────── paths / state ─────────────────────────
def install_root() -> Path:
    p = home() / "llama.cpp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path() -> Path:
    return install_root() / "state.json"


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(st: dict) -> None:
    try:
        _state_path().write_text(json.dumps(st, indent=2), encoding="utf-8")
    except OSError:
        pass


def is_managed(binary: str | None) -> bool:
    if not binary:
        return False
    try:
        Path(binary).resolve().relative_to(install_root().resolve())
        return True
    except (ValueError, OSError):
        return False


# ───────────────────────── what's installed ─────────────────────────
def _run(args: list[str], timeout: float = 15) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def build_number(binary: str | None) -> int | None:
    """``build 12340`` from ``llama-server --version`` (None when it can't be read)."""
    if not binary:
        return None
    m = re.search(r"build[ :=]+(\d+)", _run([binary, "--version"]))
    return int(m.group(1)) if m else None


def _os_arch() -> tuple[str, str]:
    osk = {"win32": "win", "darwin": "macos"}.get(sys.platform, "ubuntu")
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    return osk, arch


def cuda_driver_version() -> tuple[int, int] | None:
    """Highest CUDA version the NVIDIA driver supports (from nvidia-smi), or None without one."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run([smi]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def detect_variant(cfg: Config) -> str:
    """Backend of the build in use (from its ggml-* libraries), else a guess from the hardware."""
    if cfg.update_variant:
        return cfg.update_variant
    osk, _ = _os_arch()
    if osk == "macos":
        return "metal"
    binary = cfg.server_binary()
    if binary:
        try:
            names = " ".join(f.name.lower() for f in Path(binary).resolve().parent.iterdir())
        except OSError:
            names = ""
        for lib, v in (("ggml-cuda", "cuda"), ("ggml-hip", "rocm"), ("ggml-vulkan", "vulkan"), ("ggml-sycl", "sycl")):
            if lib in names:
                return v
        if "ggml-cpu" in names:
            return "cpu"
    return "cuda" if cuda_driver_version() else "vulkan"


# ───────────────────────── releases ─────────────────────────
@dataclass
class Asset:
    name: str
    url: str
    size: int


@dataclass
class UpdateInfo:
    tag: str
    build: int
    variant: str  # the requested variant, e.g. "cuda"
    flavor: str  # what the asset is, e.g. "win-cuda-12.4-x64"
    assets: list[Asset]  # the build itself first, then runtime libraries (cudart) if needed
    published: str = ""
    current_build: int | None = None
    current_binary: str = ""

    @property
    def available(self) -> bool:
        return self.current_build is None or self.build > self.current_build

    @property
    def managed(self) -> bool:
        return is_managed(self.current_binary)

    @property
    def size(self) -> int:
        return sum(a.size for a in self.assets)

    @property
    def dirname(self) -> str:
        return f"{self.tag}-{self.variant}"


def _vtuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(".") if x.isdigit())


def pick_assets(release: dict, variant: str) -> tuple[str, list[Asset]] | None:
    """The archive(s) of ``release`` for this machine and ``variant``; None if it has none (yet)."""
    osk, arch = _os_arch()
    by_name = {a["name"]: Asset(a["name"], a["browser_download_url"], int(a.get("size", 0)))
               for a in release.get("assets", [])}
    flavors: dict[str, Asset] = {}
    for name, a in by_name.items():
        m = ASSET_RE.match(name)
        if m:
            flavors[m.group(2)] = a
    v, _, ver = variant.partition("-")
    if osk == "macos":
        want = [f"macos-{arch}"]
    elif v == "cpu":
        want = [f"win-cpu-{arch}" if osk == "win" else f"ubuntu-{arch}"]
    elif v == "vulkan":
        want = [f"{osk}-vulkan-{arch}"]
    elif v == "sycl":
        want = ["win-sycl-x64"] if osk == "win" else [f"ubuntu-sycl-{ver or 'fp16'}-x64"]
    elif v in ("rocm", "cuda"):
        rx = re.compile(rf"^{osk}-{v}-([\d.]+)-{arch}$")
        found = [(m.group(1), f) for f in flavors if (m := rx.match(f))]
        if ver:
            found = [x for x in found if x[0] == ver]
        elif v == "cuda":
            drv = cuda_driver_version()
            if drv and drv != (0, 0):
                found = [x for x in found if _vtuple(x[0])[:2] <= drv] or found
        want = [f for _, f in sorted(found, key=lambda x: _vtuple(x[0]), reverse=True)][:1]
    else:
        want = [variant]  # an exact flavor, e.g. "ubuntu-openvino-2026.4.1-x64"
    for f in want:
        if f in flavors:
            assets = [flavors[f]]
            if osk == "win" and v == "cuda":  # the Windows CUDA builds need the CUDA runtime DLLs beside them
                cudart = by_name.get(f"cudart-llama-bin-{f}.zip")
                if not cudart:
                    return None
                assets.append(cudart)
            return f, assets
    return None


def fetch_releases(limit: int = 15) -> list[dict]:
    import httpx

    headers = {"accept": "application/vnd.github+json", "user-agent": "inari"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["authorization"] = f"Bearer {tok}"
    r = httpx.get(API, params={"per_page": limit}, headers=headers, timeout=20, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def latest(cfg: Config, variant: str = "") -> UpdateInfo:
    """Newest release that has a complete build for this machine (fresh ones upload for a while)."""
    variant = variant or detect_variant(cfg)
    for rel in fetch_releases():
        tag = rel.get("tag_name", "")
        if rel.get("draft") or not re.fullmatch(r"b\d+", tag):
            continue
        got = pick_assets(rel, variant)
        if got:
            binary = cfg.server_binary() or ""
            return UpdateInfo(tag=tag, build=int(tag[1:]), variant=variant, flavor=got[0], assets=got[1],
                              published=rel.get("published_at", ""), current_build=build_number(binary),
                              current_binary=binary)
    raise RuntimeError(f"no recent llama.cpp release has a {variant} build for {'-'.join(_os_arch())}")


def check(cfg: Config, *, force: bool = False) -> UpdateInfo | None:
    """``latest()``, but at most once per CHECK_EVERY unless forced (the answer is cached in state.json)."""
    st = load_state()
    variant = detect_variant(cfg)
    cached = st.get("latest")
    if not force and cached and cached.get("variant") == variant and time.time() - st.get("checked", 0) < CHECK_EVERY:
        info = UpdateInfo(**{**cached, "assets": [Asset(**a) for a in cached["assets"]]})
        info.current_binary = cfg.server_binary() or ""
        info.current_build = build_number(info.current_binary)
        return info
    info = latest(cfg, variant)
    st["checked"] = time.time()
    st["latest"] = asdict(info)
    save_state(st)
    return info


# ───────────────────────── install ─────────────────────────
Progress = Callable[[str, int, int], None]  # (label, bytes done, bytes total)


class Cancelled(Exception):
    pass


def _download(a: Asset, dest: Path, progress: Progress, cancel: threading.Event | None) -> None:
    import httpx

    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", a.url, follow_redirects=True, headers={"user-agent": "inari"},
                      timeout=httpx.Timeout(30.0, read=120.0)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or a.size or 0)
        done = 0
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                f.write(chunk)
                done += len(chunk)
                progress(a.name, done, total)
    part.replace(dest)


def _extract(archive: Path, dest: Path) -> None:
    """Unpack, refusing members that would land outside ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                target = (dest / m.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe path in {archive.name}: {m.filename}")
            z.extractall(dest)
        return
    with tarfile.open(archive) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
            return
        def inside(p: Path) -> bool:
            p = p.resolve()
            return p == root or root in p.parents

        for m in t.getmembers():
            target = dest / m.name
            ok = inside(target) and not (m.isdev() or m.isfifo())
            if m.issym():
                ok = ok and inside(target.parent / m.linkname)
            elif m.islnk():
                ok = ok and inside(dest / m.linkname)
            if not ok:
                raise RuntimeError(f"unsafe path in {archive.name}: {m.name}")
        t.extractall(dest)


def install(info: UpdateInfo, progress: Progress | None = None, cancel: threading.Event | None = None) -> str:
    """Download and unpack ``info`` into its own folder; returns the new llama-server path."""
    progress = progress or (lambda *_: None)
    root = install_root()
    final = root / info.dirname
    if final.is_dir():
        existing = next(final.rglob(EXE), None)
        if existing and build_number(str(existing)):
            return str(existing)
        shutil.rmtree(final, ignore_errors=True)
    dl = root / ".download"
    dl.mkdir(exist_ok=True)
    tmp = root / f".{info.dirname}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        files = []
        for a in info.assets:
            f = dl / a.name
            if not (f.exists() and (not a.size or f.stat().st_size == a.size)):
                _download(a, f, progress, cancel)
            files.append(f)
        progress("unpacking", 0, 0)
        _extract(files[0], tmp)
        exe = next(tmp.rglob(EXE), None)
        if exe is None:
            raise RuntimeError(f"{info.assets[0].name} has no {EXE}")
        for extra in files[1:]:  # runtime libraries go next to the binary
            _extract(extra, exe.parent)
        if os.name != "nt":
            exe.chmod(exe.stat().st_mode | 0o111)
        exe.parent.rename(final)  # the zips are flat (= tmp), the tarballs have a llama-bNNNN/ folder
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    binary = next(final.rglob(EXE))
    if build_number(str(binary)) is None:
        out = _run([str(binary), "--version"]).strip().splitlines()
        raise RuntimeError(f"the new llama-server doesn't run: {out[-1] if out else 'no output'}"
                           f" (left in {final})")
    for f in files:
        f.unlink(missing_ok=True)
    st = load_state()
    st["installed"] = info.dirname
    save_state(st)
    return str(binary)


def switch(cfg: Config, binary: str, *, force: bool = False) -> list[str]:
    """Point ``llama_server`` (if it is unset, managed, or ``force``) and managed profile overrides at
    ``binary``. Returns what changed."""
    from inari.profiles import Profile, list_profiles

    changed = []
    if force or not cfg.llama_server or is_managed(cfg.llama_server) or not cfg.server_binary():
        if cfg.llama_server != binary:
            cfg.llama_server = binary
            cfg.save()
            changed.append("llama_server")
    for n in list_profiles():
        try:
            p = Profile.load(n)
        except Exception:
            continue
        if p.server_bin and is_managed(p.server_bin) and p.server_bin != binary:
            p.server_bin = binary
            p.save()
            changed.append(f"profile {n}")
    return changed


def prune(keep_binary: str, keep: int = KEEP) -> list[str]:
    """Delete older managed builds, keeping the newest ``keep`` and any that a server is running from."""
    from inari import server

    in_use = set()
    for r in server.active_runs():
        if r.cmd:
            try:
                in_use.add(Path(r.cmd[0]).resolve())
            except OSError:
                pass
    builds = sorted((d for d in install_root().iterdir() if d.is_dir() and re.match(r"b\d+-", d.name)),
                    key=lambda d: int(d.name[1:].split("-")[0]), reverse=True)
    keep_dir = Path(keep_binary).resolve()
    removed = []
    for d in builds[keep:]:
        dr = d.resolve()
        if dr in keep_dir.parents or any(dr in b.parents for b in in_use):
            continue
        shutil.rmtree(d, ignore_errors=True)
        if not d.exists():
            removed.append(d.name)
    return removed


# ───────────────────────── background (main menu) ─────────────────────────
@dataclass
class Background:
    """A once-per-process check (and, with ``auto_update = "install"``, install) the menu shows."""

    info: UpdateInfo | None = None
    busy: bool = False
    label: str = ""
    done: int = 0
    total: int = 0
    message: str = ""  # one-line result for the status bar
    error: str = ""
    shown: str = ""  # the last message the menu has flashed
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def offer(self) -> bool:
        """An update is waiting for the user (u in the menu)."""
        return bool(self.info and self.info.available and not self.busy and not self.message.startswith("✔"))

    def status(self) -> str:
        if self.busy:
            if self.total:
                return f"↓ llama.cpp {self.info.tag if self.info else ''}  {self.done / self.total:4.0%}  " \
                       f"{self.done / 1e6:,.0f} / {self.total / 1e6:,.0f} MB"
            return f"llama.cpp: {self.label or 'checking'}…"
        return self.message

    def _progress(self, label: str, done: int, total: int) -> None:
        self.label, self.done, self.total = label, done, total

    def start_install(self, cfg: Config, *, force: bool = False) -> None:
        if self.busy or not self.info:
            return
        self.busy, self.message, self.error = True, "", ""
        threading.Thread(target=self._install, args=(cfg, force), daemon=True).start()

    def _install(self, cfg: Config, force: bool) -> None:
        info = self.info
        assert info
        try:
            binary = install(info, self._progress, self.cancel)
            changed = switch(cfg, binary, force=force)
            prune(binary)
            in_use = cfg.server_binary() or ""
            if not info.current_binary:
                self.message = f"✔ llama.cpp {info.tag} installed ({info.flavor}) — ready to serve"
            elif changed or (in_use and Path(in_use).resolve() == Path(binary).resolve()):
                self.message = f"✔ llama.cpp updated to {info.tag} — servers already running keep the old build"
            else:
                self.message = f"✔ llama.cpp {info.tag} installed (llama_server unchanged — inari update switches)"
            info.current_build, info.current_binary = info.build, binary
        except Cancelled:
            self.message = "llama.cpp update cancelled"
        except Exception as e:
            self.error = self.message = f"llama.cpp update failed: {e}"
        finally:
            self.busy = False

    def start_check(self, cfg: Config, *, force: bool = False, install_after: bool = False) -> None:
        if self.busy:
            return
        self.busy, self.label, self.total, self.message, self.error = True, "checking for updates", 0, "", ""
        threading.Thread(target=self._check, args=(cfg, force, install_after), daemon=True).start()

    def _check(self, cfg: Config, force: bool = False, install_after: bool = False) -> None:
        try:
            self.info = check(cfg, force=force)
        except Exception as e:
            self.error = self.message = f"llama.cpp update check failed: {e}" if force else ""
            self.busy = False
            return
        self.busy = False
        info = self.info
        if not info.available:
            if force:
                self.message = f"✔ llama.cpp is up to date (b{info.current_build}, latest {info.tag})"
            return
        # no llama-server at all: nothing to lose, so notify installs too; install mode also updates managed builds
        auto = (cfg.auto_update in ("notify", "install") and not info.current_binary) or             (cfg.auto_update == "install" and info.managed)
        if install_after or auto:
            self.start_install(cfg)
        elif info.current_build is None:
            self.message = f"llama.cpp {info.tag} ({info.flavor}) can be installed — u to install"
        else:
            self.message = f"llama.cpp {info.tag} is out (you have b{info.current_build}) — u to update"


_bg: Background | None = None


def background(cfg: Config, *, create: bool = False) -> Background | None:
    """The process-wide updater; the first call starts the automatic check (unless ``auto_update = "off"``),
    which also installs llama.cpp right away when no llama-server is found.
    ``create`` returns one even when automatic checks are off (for u in the menu)."""
    global _bg
    if _bg is None and (create or cfg.auto_update in ("notify", "install")):
        _bg = Background()
        if cfg.auto_update in ("notify", "install"):
            _bg.start_check(cfg)
    return _bg
