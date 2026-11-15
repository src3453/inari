"""Hugging Face GGUF discovery and download.

Downloads go into the Hugging Face cache (see :mod:`inari.hfcache`), so llama.cpp's
``-hf owner/repo:QUANT`` finds what inari pulled and inari lists what llama.cpp pulled.
Files can be fetched over plain HTTP (Range-resumable) or through Xet (``hf_xet``).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from inari import hfcache
from inari.config import Config

if TYPE_CHECKING:
    from rich.progress import Progress

SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
QUANT_RE = re.compile(
    r"(UD-)?(I?Q\d(?:_[0-9A-Z]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d)", re.IGNORECASE
)


@dataclass
class GgufGroup:
    """One logical model file (possibly split into several shards)."""

    name: str
    files: list[str] = field(default_factory=list)
    size: int = 0
    commit: str = ""  # repo revision the listing came from
    oids: dict[str, str] = field(default_factory=dict)  # file -> blob id (LFS sha256 / git sha1)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def is_mmproj(self) -> bool:
        return "mmproj" in self.name.lower()

    @property
    def quant(self) -> str:
        stem = Path(self.name).name
        m = None
        for m in QUANT_RE.finditer(stem):
            pass
        return m.group(0).upper() if m else "?"

    @property
    def entry(self) -> str:
        """The file llama.cpp should be pointed at (first shard)."""
        return sorted(self.files)[0]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _token(cfg: Config) -> str | None:
    return cfg.hf_token or os.environ.get("HF_TOKEN") or None


def list_groups(repo: str, cfg: Config) -> list[GgufGroup]:
    from huggingface_hub import HfApi

    info = HfApi(token=_token(cfg)).model_info(repo, files_metadata=True)
    groups: dict[str, GgufGroup] = {}
    for s in info.siblings or []:
        fn = s.rfilename
        if not fn.lower().endswith(".gguf"):
            continue
        m = SPLIT_RE.match(fn)
        key = m.group("base") + ".gguf" if m else fn
        g = groups.setdefault(key, GgufGroup(name=key, commit=info.sha or ""))
        g.files.append(fn)
        g.size += s.size or 0
        g.sizes[fn] = s.size or 0
        oid = (s.lfs.sha256 if s.lfs else None) or s.blob_id
        if oid:
            g.oids[fn] = oid
    return sorted(groups.values(), key=lambda g: (g.is_mmproj, g.name.lower()))


def split_spec(spec: str) -> tuple[str, str | None]:
    """``owner/repo:Q4_K_M`` -> (``owner/repo``, ``Q4_K_M``)."""
    if spec.startswith("https://huggingface.co/"):
        spec = spec.removeprefix("https://huggingface.co/").strip("/")
        spec = "/".join(spec.split("/")[:2])
    if ":" in spec:
        repo, quant = spec.split(":", 1)
        return repo, quant or None
    return spec, None


def local_dir_for(repo: str, cfg: Config) -> Path:
    """Where older inari versions stored downloads (still read, no longer written)."""
    return cfg.models_path() / repo.replace("/", "__")


# ───────────────────────── fetching into the HF cache ─────────────────────────
BACKENDS = ("auto", "xet", "http")


class Cancelled(Exception):
    pass


def _noop(*_a: object) -> None:
    return None


class RateMeter:
    """Bytes/s over a sliding window. Thread-safe; decays to 0 when bytes stop arriving."""

    def __init__(self, window: float = 4.0) -> None:
        import threading
        from collections import deque

        self.window = window
        self.total = 0
        self._samples: deque[tuple[float, int]] = deque()  # (time, cumulative bytes)
        self._base: tuple[float, int] | None = None  # last sample that slid out of the window
        self._lock = threading.Lock()

        self._last = 0.0  # time of the last add

    def start(self) -> None:
        """A new transfer: measure from its first byte, so setup time (metadata, connecting) isn't averaged in."""
        with self._lock:
            self._samples.clear()
            self._base = None

    def add(self, n: int) -> None:
        if n <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self.total += n
            self._last = now
            if self._base is None:  # first chunk only starts the clock (it took unknown time to arrive)
                self._base = (now, self.total)
                return
            self._samples.append((now, self.total))
            self._prune(now)

    def idle(self) -> float:
        """Seconds since bytes last arrived (inf if none yet)."""
        return time.monotonic() - self._last if self._last else float("inf")

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window:
            self._base = self._samples.popleft()

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if self._base is None:
                return 0.0
            t0, b0 = self._base
            span = now - t0
            return (self.total - b0) / span if span > 0.3 else 0.0


class Speed:
    """Download speed: network bytes when the backend reports them, else bytes written."""

    def __init__(self) -> None:
        self.net = RateMeter()  # steady (HTTP chunks, Xet transfer reports)
        self.disk = RateMeter()  # Xet flushes these in bursts; only a fallback for old huggingface_hub
        self._base = self._size = self._net0 = 0
        self._started = (0, 0)

    def start(self, base: int = 0, size: int = 0) -> None:
        """A file's transfer begins. ``base``: bytes done before it; ``size``: its size."""
        self.net.start()
        self.disk.start()
        self._base, self._size, self._net0 = base, size, self.net.total
        self._started = (self.net.total, self.disk.total)

    def rate(self) -> float:
        return self.net.rate() if self.net.total else self.disk.rate()

    def stalled(self, after: float = 3.0) -> bool:
        """Bytes were flowing for the current file, then none for ``after`` seconds."""
        flowing = (self.net.total, self.disk.total) != self._started
        return flowing and min(self.net.idle(), self.disk.idle()) > after

    def shown(self, done: int) -> int:
        """Bytes to display. Xet writes to disk well after receiving, so count received bytes too."""
        got = self.net.total - self._net0
        if self._size:
            got = min(got, self._size)
        return max(done, self._base + got)


def _dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


@dataclass
class FetchCtl:
    """Hooks a fetch reports through (TUI job, rich progress bar, ...)."""

    progress: Callable[[int], None] = _noop  # file bytes completed by this fetch
    cancelled: Callable[[], bool] = lambda: False
    status: Callable[[str], None] = _noop  # "xet" | "http" | "waiting"
    transfer: Callable[[int], None] = _noop  # bytes received over the network (drives the speed)
    skip: Callable[[int], None] | None = None  # bytes already on disk (progress, but not speed)

    def have(self, n: int) -> None:
        (self.skip or self.progress)(n)


def xet_available() -> bool:
    try:
        from huggingface_hub.file_download import is_xet_available
        return bool(is_xet_available())
    except Exception:
        return False


def snapshot_path(cfg: Config, repo: str, group: GgufGroup, fn: str) -> Path:
    return hfcache.repo_dir(cfg, repo) / "snapshots" / group.commit / fn


def pick_backend(cfg: Config, repo: str, oid: str, backend: str) -> str:
    backend = backend if backend in BACKENDS else "auto"
    if backend == "auto":
        # a partial HTTP download is worth resuming; otherwise Xet when it is installed
        if hfcache.partial_path(cfg, repo, oid).exists() or not xet_available():
            return "http"
        return "xet"
    if backend == "xet" and not xet_available():
        raise RuntimeError("the xet backend needs the hf_xet package (pip install hf_xet) — or use http")
    return backend


def fetch_file(cfg: Config, repo: str, group: GgufGroup, fn: str, backend: str, ctl: FetchCtl) -> Path:
    """Put one file in the HF cache and return its snapshot path."""
    if not group.commit or fn not in group.oids:
        raise RuntimeError(f"no revision / blob id for {fn} — reopen the repo")
    oid, size = group.oids[fn], group.sizes.get(fn, 0)
    snap = snapshot_path(cfg, repo, group, fn)
    if snap.is_file():
        ctl.have(snap.stat().st_size)
        return snap
    legacy = local_dir_for(repo, cfg) / fn
    if legacy.is_file():
        ctl.have(legacy.stat().st_size)
        return legacy

    rdir = hfcache.repo_dir(cfg, repo)
    try:
        with hfcache.BlobLock(cfg, repo, oid, keep_waiting=lambda: not ctl.cancelled(),
                              on_wait=lambda: ctl.status("waiting")):
            if snap.is_file():  # another process finished it while we waited
                ctl.have(snap.stat().st_size)
            else:
                blob = hfcache.blob_path(cfg, repo, oid)
                if blob.is_file():
                    ctl.have(blob.stat().st_size)
                else:
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    via = pick_backend(cfg, repo, oid, backend)
                    if via == "xet" and not _fetch_xet(cfg, repo, group, fn, blob, ctl):
                        via = "http"  # this file isn't stored on Xet
                    if via == "http":
                        _fetch_http(cfg, repo, group, fn, blob, ctl)
                    got = blob.stat().st_size
                    if size and got != size:
                        blob.unlink()
                        raise RuntimeError(f"{fn}: got {got} bytes, expected {size}")
                hfcache.link_snapshot(blob, snap)
            if hfcache.read_ref(rdir) != group.commit:
                hfcache.write_ref(rdir, group.commit)  # the snapshot llama.cpp's -hf reads
    except hfcache.LockWaitAborted:
        raise Cancelled() from None
    return snap


def _fetch_http(cfg: Config, repo: str, group: GgufGroup, fn: str, blob: Path, ctl: FetchCtl) -> None:
    """Range-resumable download into ``blobs/<oid>.downloadInProgress`` (the name llama.cpp resumes too)."""
    import httpx
    from huggingface_hub import hf_hub_url

    ctl.status("http")
    part = hfcache.partial_path(cfg, repo, group.oids[fn])
    have = part.stat().st_size if part.exists() else 0
    headers = {"user-agent": "inari"}
    tok = _token(cfg)
    if tok:
        headers["authorization"] = f"Bearer {tok}"  # httpx drops it on cross-origin redirects
    if have:
        headers["range"] = f"bytes={have}-"
    url = hf_hub_url(repo, fn, revision=group.commit)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(30.0, read=120.0)) as r:
        if have and r.status_code == 416:  # already complete
            os.replace(part, blob)
            ctl.have(have)
            return
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored Range -> start over
            have = 0
        ctl.have(have)
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                if ctl.cancelled():
                    raise Cancelled()
                f.write(chunk)
                ctl.transfer(len(chunk))
                ctl.progress(len(chunk))
    os.replace(part, blob)


def _fetch_xet(cfg: Config, repo: str, group: GgufGroup, fn: str, blob: Path, ctl: FetchCtl) -> bool:
    """Download through hf_xet. Returns False if the file is not Xet-backed.

    Xet writes out of order, so it gets its own temp file (never ``.downloadInProgress``, which
    llama.cpp would resume with a Range request); it is dropped on failure or cancel.
    """
    import inspect

    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    from tqdm.auto import tqdm

    try:  # internals of recent huggingface_hub; older versions -> plain HTTP
        from huggingface_hub.file_download import xet_get
        from huggingface_hub.utils._xet import abort_xet_session
    except ImportError:
        return False
    if "_tqdm_bar" not in inspect.signature(xet_get).parameters:
        return False

    tok = _token(cfg)
    meta = get_hf_file_metadata(hf_hub_url(repo, fn, revision=group.commit), token=tok)
    if getattr(meta, "xet_file_data", None) is None:
        return False
    ctl.status("xet")

    aborted = [False]

    def _check_cancel() -> bool:
        # runs inside hf_xet's progress callback: raising here only prints a traceback to stderr,
        # so cancel by aborting the session and let xet_get fail on its own
        if ctl.cancelled():
            if not aborted[0]:
                aborted[0] = True
                abort_xet_session()
            return True
        return False

    class _Bar(tqdm):  # receives xet's progress updates; never draws
        def __init__(self, **kw: object) -> None:
            kw["disable"] = True
            super().__init__(**kw)  # type: ignore[arg-type]

        def update(self, n: float | None = 1) -> bool | None:
            # bytes written to the file: flushed in bursts, so it only drives the % done
            if not _check_cancel():
                ctl.progress(int(n or 0))
            return None

        # Having these makes huggingface_hub route network bytes here too (instead of dropping them).
        # They arrive steadily, so they drive the speed.
        def update_transfer(self, n: float | None = 1) -> None:
            if not _check_cancel():
                ctl.transfer(int(n or 0))

        def set_transfer_postfix_str(self, *_a: object, **_kw: object) -> None:
            return None

    tmp = blob.with_name(f"{blob.name}.xet.{os.getpid()}.incomplete")
    bar = _Bar(total=group.sizes.get(fn) or None)
    try:
        xet_get(incomplete_path=tmp, xet_file_data=meta.xet_file_data, headers=build_hf_headers(token=tok),
                expected_size=meta.size, displayed_filename=fn, _tqdm_bar=bar)
        if ctl.cancelled():
            raise Cancelled()
        os.replace(tmp, blob)
    except BaseException as e:
        tmp.unlink(missing_ok=True)
        if ctl.cancelled() and not isinstance(e, Cancelled):
            raise Cancelled() from None
        raise
    return True


def download_group(repo: str, group: GgufGroup, cfg: Config, console: Console, backend: str = "") -> Path:
    from rich.progress import BarColumn, DownloadColumn, Progress, ProgressColumn, Task, TextColumn
    from rich.text import Text

    meters: dict[int, Speed] = {}

    class SpeedColumn(ProgressColumn):  # rich's own speed counts resumed bytes as if just downloaded
        def render(self, task: Task) -> Text:
            m = meters.get(task.id)
            if m is None or task.finished:
                return Text("")
            r = m.rate()
            left = f" · {_dur(task.remaining / r)}" if r and task.remaining is not None else ""
            return Text(f"{human(r)}/s{left}" if r else "—", style="progress.data.speed")

    backend = backend or cfg.download_backend
    first: Path | None = None
    files = sorted(group.files)
    with Progress(TextColumn("[accent]↓[/] [text]{task.description}[/]"), BarColumn(),
                  DownloadColumn(), SpeedColumn(),
                  TextColumn("[dim]{task.fields[via]}[/]"), console=console) as prog:
        for i, fn in enumerate(files, 1):
            tag = f" [{i}/{len(files)}]" if len(files) > 1 else ""
            task = prog.add_task(fn + tag, total=group.sizes.get(fn) or None, via="")
            meters[task] = speed = Speed()
            ctl = _bar_ctl(prog, task, speed, group.sizes.get(fn, 0))
            path = fetch_file(cfg, repo, group, fn, backend, ctl)
            first = first or path
    assert first is not None
    return first


def _bar_ctl(prog: Progress, task: int, speed: Speed, size: int) -> FetchCtl:
    """FetchCtl driving one rich progress task (one file)."""
    done = 0

    def show() -> None:
        prog.update(task, completed=speed.shown(done))

    def status(st: str) -> None:
        if st in ("xet", "http"):
            speed.start(size=size)
        prog.update(task, via="waiting for another download…" if st == "waiting" else st)

    def progress(n: int) -> None:
        nonlocal done
        speed.disk.add(n)
        done += n
        show()

    def skip(n: int) -> None:
        nonlocal done
        done += n
        show()

    def transfer(n: int) -> None:
        speed.net.add(n)
        show()

    return FetchCtl(progress=progress, status=status, transfer=transfer, skip=skip)


def pull(spec: str, cfg: Config, console: Console, *, yes: bool = False,
         backend: str = "") -> tuple[Path | None, Path | None]:
    """Interactive download. Returns (model_path, mmproj_path)."""
    repo, quant = split_spec(spec)
    with console.status(f"[accent]Looking up[/] [text]{repo}[/] …", spinner="dots"):
        groups = list_groups(repo, cfg)
    if not groups:
        console.print(f"[bad]No .gguf files in {repo}[/]")
        return None, None

    models = [g for g in groups if not g.is_mmproj]
    projs = [g for g in groups if g.is_mmproj]

    chosen: GgufGroup | None = None
    if quant:
        q = quant.lower()
        hits = [g for g in models if q in g.name.lower()]
        if len(hits) == 1:
            chosen = hits[0]
        elif hits:
            models = hits
        else:
            console.print(f"[warn]No file matches '{quant}', pick one:[/]")

    if chosen is None:
        t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="accent", title=f"[accent]{repo}[/]")
        t.add_column("#", justify="right", style="accent2")
        t.add_column("FILE", style="text")
        t.add_column("QUANT", style="accent2")
        t.add_column("SIZE", justify="right", style="text")
        t.add_column("PARTS", justify="right", style="dim")
        for i, g in enumerate(models, 1):
            t.add_row(str(i), g.name, g.quant, human(g.size), str(len(g.files)))
        console.print(t)
        idx = Prompt.ask("[accent]Download which #[/]", choices=[str(i) for i in range(1, len(models) + 1)],
                         show_choices=False, console=console)
        chosen = models[int(idx) - 1]

    console.print(f"[dim]→ {hfcache.repo_dir(cfg, repo)}[/]  [text]{chosen.name}[/] [accent2]{human(chosen.size)}[/]")
    model_path = download_group(repo, chosen, cfg, console, backend)

    mmproj_path: Path | None = None
    if projs:
        pick = projs[0]
        for p in projs:  # prefer F16 projector
            if "f16" in p.name.lower():
                pick = p
                break
        if yes or Confirm.ask(f"[accent]Vision projector found[/] ({pick.name}, {human(pick.size)}). Download it?",
                              default=False, console=console):
            mmproj_path = download_group(repo, pick, cfg, console, backend)

    console.print(f"[good]✔ Downloaded[/] [text]{model_path}[/]")
    return model_path, mmproj_path


# ───────────────────────── local models ─────────────────────────
@dataclass
class LocalModel:
    path: Path
    size: int
    ref: str  # what a profile stores: hf://owner/repo/file.gguf, or a path relative to models_dir
    source: str  # "hf" (HF cache, shared with llama.cpp -hf) | "dir" (models_dir)


def model_ref(cfg: Config, path: Path) -> str:
    """Profile string for a file on disk."""
    ref = hfcache.ref_from_path(cfg, path)
    if ref:
        return ref
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(cfg.models_path())).as_posix()
    except ValueError:
        return str(path)


def local_models(cfg: Config) -> list[LocalModel]:
    out = [LocalModel(path=c.path, size=c.size, ref=c.ref, source="hf") for c in hfcache.scan(cfg)]
    root = cfg.models_path()
    cache = os.path.abspath(hfcache.cache_dir(cfg))
    seen: set[str] = set()
    for p in sorted(root.rglob("*.gguf")):
        if ".cache" in p.parts or os.path.abspath(p).startswith(cache + os.sep):
            continue
        m = SPLIT_RE.match(p.name)
        if m:
            key = str(p.parent / m.group("base"))
            if key in seen:
                continue
            seen.add(key)
            size = sum(q.stat().st_size for q in p.parent.glob(m.group("base") + "-*-of-*.gguf"))
        else:
            size = p.stat().st_size
        out.append(LocalModel(path=p, size=size, ref=model_ref(cfg, p), source="dir"))
    return out


def model_files(m: LocalModel) -> list[Path]:
    """Every file on disk that makes up ``m`` (all shards of a split model)."""
    sm = SPLIT_RE.match(m.path.name)
    if not sm:
        return [m.path]
    return sorted(m.path.parent.glob(sm.group("base") + "-*-of-*.gguf"))


def delete_model(cfg: Config, m: LocalModel) -> int:
    """Delete a local model (all shards). HF-cache entries also drop their unshared blobs. Returns bytes freed."""
    if m.source == "hf":
        parsed = hfcache.parse_ref(m.ref)
        if parsed:
            return hfcache.delete(cfg, *parsed)
    freed = 0
    for p in model_files(m):
        size = p.stat().st_size
        p.unlink()
        freed += size
    return freed


# ───────────────────────── migrating old downloads ─────────────────────────
@dataclass
class _Legacy:
    repo: str
    rel: str
    path: Path


def legacy_files(cfg: Config) -> tuple[list[_Legacy], list[Path]]:
    """GGUFs under ``models_dir/owner__repo/`` (old download layout) and leftover ``.part`` files."""
    root = cfg.models_path()
    files: list[_Legacy] = []
    parts: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        repo = d.name.replace("__", "/", 1)
        for p in sorted(d.rglob("*")):
            if p.name.endswith(".gguf.part"):
                parts.append(p)
            elif p.is_file() and p.suffix == ".gguf":
                files.append(_Legacy(repo=repo, rel=p.relative_to(d).as_posix(), path=p))
    return files, parts


def migrate(cfg: Config, console: Console, *, yes: bool = False) -> dict[str, str]:
    """Move old ``models_dir/owner__repo`` downloads into the HF cache. Returns {old path: hf ref}."""
    from huggingface_hub import HfApi

    files, parts = legacy_files(cfg)
    if not files:
        console.print(f"[dim]Nothing to migrate in {cfg.models_path()}[/]")
        return {}
    total = sum(f.path.stat().st_size for f in files)
    console.print(f"[accent]{len(files)}[/] files ([accent2]{human(total)}[/]) in [dim]{cfg.models_path()}[/]"
                  f" → [dim]{hfcache.cache_dir(cfg)}[/]")
    if not yes and not Confirm.ask("[accent]Move them into the Hugging Face cache?[/]", default=True,
                                   console=console):
        return {}

    api = HfApi(token=_token(cfg))
    infos: dict[str, object] = {}
    moved: dict[str, str] = {}
    for f in files:
        if f.repo not in infos:
            try:
                infos[f.repo] = api.model_info(f.repo, files_metadata=True)
            except Exception as e:
                infos[f.repo] = e
        info = infos[f.repo]
        if isinstance(info, Exception):
            console.print(f"[warn]· skip[/] {f.repo}/{f.rel} [dim]({info})[/]")
            continue
        sib = next((s for s in info.siblings or [] if s.rfilename == f.rel), None)  # type: ignore[attr-defined]
        if sib is None:
            console.print(f"[warn]· skip[/] {f.repo}/{f.rel} [dim](no longer on the Hub)[/]")
            continue
        size = f.path.stat().st_size
        oid = (sib.lfs.sha256 if sib.lfs else None) or sib.blob_id
        if not oid or (sib.size and sib.size != size):
            console.print(f"[warn]· skip[/] {f.repo}/{f.rel} [dim](differs from the Hub copy)[/]")
            continue
        commit = info.sha  # type: ignore[attr-defined]
        rdir = hfcache.repo_dir(cfg, f.repo)
        snap = rdir / "snapshots" / commit / f.rel
        blob = hfcache.blob_path(cfg, f.repo, oid)
        with hfcache.BlobLock(cfg, f.repo, oid):
            note = "moved"
            if snap.is_file():
                note = f"already cached — duplicate left at {f.path}"
            else:
                if blob.is_file():
                    note = f"blob already cached — duplicate left at {f.path}"
                else:
                    hfcache.move_into_blob(f.path, blob)
                hfcache.link_snapshot(blob, snap)
            hfcache.write_ref(rdir, commit)
        moved[str(f.path)] = hfcache.make_ref(f.repo, f.rel)
        console.print(f"[good]✔[/] {f.repo}/{f.rel} [dim]{note}[/]")

    for d in sorted({f.path.parent for f in files}, key=lambda p: len(p.parts), reverse=True):
        try:
            while d != cfg.models_path() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
        except OSError:
            pass
    if parts:
        console.print(f"[warn]{len(parts)} unfinished .part file(s) left in place[/] [dim](re-download to the cache"
                      " and delete them)[/]")
    return moved


# ───────────────────────── search + background downloads (used by the hub TUI) ─────────────────────────
SORTS = [("downloads", "downloads"), ("trending_score", "trending"), ("likes", "likes"),
         ("last_modified", "updated")]


@dataclass
class RepoHit:
    id: str
    downloads: int = 0
    likes: int = 0
    modified: float | None = None  # unix time
    params: int = 0
    arch: str = ""
    ctx: int = 0
    pipeline: str = ""
    gated: bool = False


def search_repos(query: str, cfg: Config, *, sort: str = "downloads", limit: int = 60) -> list[RepoHit]:
    """GGUF repos on the Hub matching ``query`` (empty = most popular)."""
    from huggingface_hub import HfApi

    api = HfApi(token=_token(cfg))
    out: list[RepoHit] = []
    for m in api.list_models(filter="gguf", search=query.strip() or None, sort=sort, limit=limit,
                             expand=["downloads", "likes", "lastModified", "gguf", "pipeline_tag", "gated"]):
        g = getattr(m, "gguf", None) or {}
        lm = getattr(m, "last_modified", None)
        out.append(RepoHit(
            id=m.id, downloads=m.downloads or 0, likes=m.likes or 0,
            modified=lm.timestamp() if lm else None,
            params=int(g.get("total") or 0), arch=str(g.get("architecture") or ""),
            ctx=int(g.get("context_length") or 0), pipeline=m.pipeline_tag or "",
            gated=bool(getattr(m, "gated", False)),
        ))
    return out


def local_state(repo: str, group: GgufGroup, cfg: Config) -> str:
    """'done' if every shard is on disk, 'partial' if a resumable download exists, else ''."""
    legacy = local_dir_for(repo, cfg)

    def have(fn: str) -> bool:
        return (bool(group.commit) and snapshot_path(cfg, repo, group, fn).is_file()) or (legacy / fn).is_file()

    if all(have(fn) for fn in group.files):
        return "done"
    if any(fn in group.oids and hfcache.partial_path(cfg, repo, group.oids[fn]).exists() for fn in group.files):
        return "partial"
    return ""


@dataclass
class Job:
    repo: str
    group: GgufGroup
    mmproj: GgufGroup | None = None
    status: str = "queued"  # queued | downloading | done | failed | cancelled
    done: int = 0
    error: str = ""
    path: Path | None = None
    mmproj_path: Path | None = None
    current: str = ""
    profile: str = ""  # profile created from this job
    backend: str = "auto"  # requested: auto | xet | http
    via: str = ""  # what the current file is using: xet | http | waiting
    _cancel: bool = field(default=False, repr=False)
    _speed: Speed = field(default_factory=Speed, repr=False)

    @property
    def speed(self) -> float:
        """Bytes/s over the last few seconds (0 when stalled or not downloading)."""
        return self._speed.rate() if self.status == "downloading" and self.via != "waiting" else 0.0

    @property
    def total(self) -> int:
        return self.group.size + (self.mmproj.size if self.mmproj else 0)

    @property
    def frac(self) -> float:
        return self.shown / self.total if self.total else 0.0

    @property
    def shown(self) -> int:
        """Bytes to display: on disk, or received but not yet written (Xet), whichever is further."""
        return min(self.total, self._speed.shown(self.done)) if self.status == "downloading" else self.done

    @property
    def stalled(self) -> bool:
        return self.status == "downloading" and self.via != "waiting" and self._speed.stalled()

    @property
    def active(self) -> bool:
        return self.status in ("queued", "downloading")

    def cancel(self) -> None:
        self._cancel = True


class Downloader:
    """Sequential download queue on a worker thread, writing into the HF cache."""

    def __init__(self, cfg: Config) -> None:
        import queue
        import threading

        self.cfg = cfg
        self.backend = cfg.download_backend if cfg.download_backend in BACKENDS else "auto"
        self.jobs: list[Job] = []
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, repo: str, group: GgufGroup, mmproj: GgufGroup | None = None) -> Job:
        job = Job(repo=repo, group=group, mmproj=mmproj, backend=self.backend)
        self.jobs.append(job)
        self._q.put(job)
        return job

    def busy(self) -> bool:
        return any(j.active for j in self.jobs)

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job._cancel:
                job.status = "cancelled"
                continue
            job.status = "downloading"
            try:
                job.path = self._group(job, job.group)
                if job.mmproj:
                    job.mmproj_path = self._group(job, job.mmproj)
                job.status, job.current, job.via = "done", "", ""
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # network / auth / disk
                job.status, job.error = "failed", str(e) or type(e).__name__

    def _group(self, job: Job, group: GgufGroup) -> Path:
        first: Path | None = None
        for fn in sorted(group.files):
            path = self._file(job, group, fn)
            first = first or path
        assert first is not None
        return first

    def _file(self, job: Job, group: GgufGroup, fn: str) -> Path:
        job.current = fn

        def progress(n: int) -> None:
            job.done += n
            job._speed.disk.add(n)

        def skip(n: int) -> None:
            job.done += n

        def status(st: str) -> None:
            if st in ("xet", "http"):  # don't average in metadata / lock-wait time
                job._speed.start(base=job.done, size=group.sizes.get(fn, 0))
            job.via = st

        return fetch_file(self.cfg, job.repo, group, fn, job.backend,
                          FetchCtl(progress=progress, cancelled=lambda: job._cancel, status=status,
                                   transfer=job._speed.net.add, skip=skip))
