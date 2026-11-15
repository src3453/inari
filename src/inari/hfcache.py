"""The Hugging Face hub cache, shared with llama.cpp's ``-hf`` and ``huggingface_hub``.

Layout (https://huggingface.co/docs/hub/local-cache)::

    <cache>/models--owner--repo/
        blobs/<oid>                         file contents (LFS sha256 or git blob sha1)
        blobs/<oid>.downloadInProgress      partial download (same name llama.cpp uses -> either can resume)
        refs/main                           commit hash of the snapshot llama.cpp reads
        snapshots/<commit>/<path>           symlink -> ../../blobs/<oid> (hardlink / real file as fallbacks)
    <cache>/.locks/models--owner--repo/<oid>.lock   held while a blob is downloaded or linked

Profiles refer to files in it as ``hf://owner/repo/path/to/file.gguf`` so a snapshot update
never leaves a stale absolute path behind.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from inari.config import Config

REF_PREFIX = "hf://"
PARTIAL_SUFFIX = ".downloadInProgress"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def cache_dir(cfg: "Config | None" = None) -> Path:
    """Same lookup order as llama.cpp (common/hf-cache.cpp), with inari's own setting first."""
    if cfg is not None and cfg.hf_cache:
        return Path(cfg.hf_cache).expanduser()
    for var, sub in (("LLAMA_CACHE", None), ("HF_HUB_CACHE", None), ("HUGGINGFACE_HUB_CACHE", None),
                     ("HF_HOME", ("hub",)), ("XDG_CACHE_HOME", ("huggingface", "hub"))):
        base = os.environ.get(var)
        if base:
            return Path(base).expanduser().joinpath(*(sub or ()))
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def folder_repo(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    return name.removeprefix("models--").replace("--", "/")


def repo_dir(cfg: "Config", repo: str) -> Path:
    return cache_dir(cfg) / repo_folder(repo)


def make_ref(repo: str, filename: str) -> str:
    return f"{REF_PREFIX}{repo}/{filename}"


def parse_ref(ref: str) -> tuple[str, str] | None:
    """``hf://owner/repo/a/b.gguf`` -> (``owner/repo``, ``a/b.gguf``)."""
    if not ref.startswith(REF_PREFIX):
        return None
    parts = ref.removeprefix(REF_PREFIX).split("/", 2)
    if len(parts) < 3 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}", parts[2]


def read_ref(rdir: Path, name: str = "main") -> str | None:
    try:
        commit = (rdir / "refs" / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return commit if _COMMIT_RE.match(commit) else None


def write_ref(rdir: Path, commit: str, name: str = "main") -> None:
    p = rdir / "refs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(commit, encoding="utf-8")


def snapshot_dirs(rdir: Path) -> list[Path]:
    """Snapshots of a repo, the one ``refs/main`` points at first, then newest first."""
    snaps = rdir / "snapshots"
    if not snaps.is_dir():
        return []
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    main = read_ref(rdir)

    def key(d: Path) -> tuple[int, float]:
        try:
            mt = d.stat().st_mtime
        except OSError:
            mt = 0.0
        return (0 if d.name == main else 1, -mt)

    return sorted(dirs, key=key)


def _usable(p: Path) -> bool:
    try:
        return p.is_file()  # follows symlinks: a dangling link is not usable
    except OSError:
        return False


def find(cfg: "Config", repo: str, filename: str) -> Path | None:
    """Path of ``filename`` in any snapshot of ``repo`` (refs/main preferred)."""
    for d in snapshot_dirs(repo_dir(cfg, repo)):
        p = d / filename
        if _usable(p):
            return p
    return None


def resolve(cfg: "Config", ref: str) -> Path | None:
    """``hf://`` ref -> a file on disk (HF cache, then inari's legacy models dir)."""
    parsed = parse_ref(ref)
    if not parsed:
        return None
    repo, filename = parsed
    hit = find(cfg, repo, filename)
    if hit:
        return hit
    legacy = cfg.models_path() / repo.replace("/", "__") / filename
    return legacy if legacy.is_file() else None


def ref_from_path(cfg: "Config", path: Path) -> str | None:
    """Inverse of :func:`resolve` for paths inside the cache (symlinks are *not* followed)."""
    try:
        rel = Path(os.path.abspath(path)).relative_to(os.path.abspath(cache_dir(cfg)))
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4 or parts[1] != "snapshots":
        return None
    repo = folder_repo(parts[0])
    return make_ref(repo, "/".join(parts[3:])) if repo else None


# ───────────────────────── writing ─────────────────────────
def blob_path(cfg: "Config", repo: str, oid: str) -> Path:
    return repo_dir(cfg, repo) / "blobs" / oid


def partial_path(cfg: "Config", repo: str, oid: str) -> Path:
    return blob_path(cfg, repo, oid).with_name(oid + PARTIAL_SUFFIX)


def lock_path(cfg: "Config", repo: str, oid: str) -> Path:
    return cache_dir(cfg) / ".locks" / repo_folder(repo) / f"{oid}.lock"


class LockWaitAborted(Exception):
    pass


class BlobLock:
    """Exclusive lock on one blob, compatible with llama.cpp (LockFileEx / flock on byte 0) and huggingface_hub."""

    def __init__(self, cfg: "Config", repo: str, oid: str,
                 keep_waiting: Callable[[], bool] | None = None, on_wait: Callable[[], None] | None = None) -> None:
        from filelock import FileLock

        p = lock_path(cfg, repo, oid)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(p))
        self._keep_waiting = keep_waiting
        self._on_wait = on_wait

    def __enter__(self) -> "BlobLock":
        import time

        from filelock import Timeout

        waited = False
        while True:
            try:
                self._lock.acquire(timeout=0.5)
                return self
            except Timeout:
                if not waited and self._on_wait:
                    self._on_wait()
                waited = True
                if self._keep_waiting and not self._keep_waiting():
                    raise LockWaitAborted()
                time.sleep(0.1)

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def link_snapshot(blob: Path, snap: Path) -> None:
    """Expose ``blob`` at ``snap``: relative symlink, else hardlink, else move (like llama.cpp's degraded mode)."""
    snap.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.relpath(blob, snap.parent), snap)
        if _usable(snap):
            return
        snap.unlink()
    except OSError:
        pass
    try:
        os.link(blob, snap)
        return
    except OSError:
        pass
    os.replace(blob, snap)


def move_into_blob(src: Path, blob: Path) -> None:
    blob.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, blob)
    except OSError:  # different volume
        shutil.move(str(src), str(blob))


# ───────────────────────── scanning ─────────────────────────
@dataclass
class CachedFile:
    repo: str
    filename: str  # path inside the snapshot (first shard for split models)
    path: Path
    size: int  # all shards

    @property
    def ref(self) -> str:
        return make_ref(self.repo, self.filename)


def scan(cfg: "Config", suffix: str = ".gguf") -> list[CachedFile]:
    """Every ``*.gguf`` in the cache, one entry per logical model (split shards folded together)."""
    root = cache_dir(cfg)
    if not root.is_dir():
        return []
    out: list[CachedFile] = []
    for rdir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        repo = folder_repo(rdir.name)
        if not repo or not rdir.is_dir():
            continue
        seen: dict[str, tuple[Path, int]] = {}
        for snap in snapshot_dirs(rdir):
            for p in snap.rglob("*" + suffix):
                rel = p.relative_to(snap).as_posix()
                if rel in seen or not _usable(p):
                    continue
                try:
                    seen[rel] = (p, p.stat().st_size)
                except OSError:
                    continue
        groups: dict[str, CachedFile] = {}
        for rel in sorted(seen):
            p, size = seen[rel]
            m = SPLIT_RE.match(rel)
            key = m.group("base") if m else rel
            if key in groups:
                groups[key].size += size
            else:
                groups[key] = CachedFile(repo=repo, filename=rel, path=p, size=size)
        out.extend(groups.values())
    return out


# ───────────────────────── deleting ─────────────────────────
def shard_names(filename: str) -> Callable[[str], bool]:
    """Matcher for every snapshot path that belongs to the logical model ``filename`` (all split shards)."""
    m = SPLIT_RE.match(filename)
    if not m:
        return lambda rel: rel == filename
    base = m.group("base")
    return lambda rel: (sm := SPLIT_RE.match(rel)) is not None and sm.group("base") == base


def _blob_of(link: Path, blobs: list[Path]) -> Path | None:
    """The blob a snapshot entry stands for: symlink target, or the blob it is hard-linked with."""
    try:
        if link.is_symlink():
            target = Path(os.path.abspath(link.parent / os.readlink(link)))
            return target if target.exists() else None
        return next((b for b in blobs if os.path.samefile(b, link)), None)
    except OSError:
        return None


def delete(cfg: "Config", repo: str, filename: str) -> int:
    """Remove ``filename`` (and its split shards) from every snapshot of ``repo``, plus the blobs nothing else
    links to. Empty snapshots go too, and the whole repo folder once no snapshot is left. Returns bytes freed."""
    rdir = repo_dir(cfg, repo)
    bdir = rdir / "blobs"
    blobs = [b for b in bdir.iterdir() if b.is_file()] if bdir.is_dir() else []
    match = shard_names(filename)
    freed = 0
    dropped: set[Path] = set()
    for snap in snapshot_dirs(rdir):
        for p in list(snap.rglob("*")):
            rel = p.relative_to(snap).as_posix()
            if not match(rel) or not (p.is_symlink() or p.is_file()):
                continue
            blob = _blob_of(p, blobs)
            if blob is not None:
                dropped.add(blob)
            elif p.is_file() and not p.is_symlink():  # real file (no blob behind it)
                freed += p.stat().st_size
            p.unlink()
    # a blob is still needed if any remaining snapshot entry points at it
    keep: set[Path] = set()
    for snap in snapshot_dirs(rdir):
        for p in snap.rglob("*"):
            if p.is_symlink() or p.is_file():
                b = _blob_of(p, blobs)
                if b is not None:
                    keep.add(b)
    for b in dropped - keep:
        try:
            size = b.stat().st_size
            b.unlink()
            freed += size
        except FileNotFoundError:
            pass
    for snap in snapshot_dirs(rdir):  # prune empty folders, deepest first
        for d in sorted((q for q in snap.rglob("*") if q.is_dir() and not q.is_symlink()),
                        key=lambda q: len(q.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        try:
            snap.rmdir()
        except OSError:
            pass
    if not snapshot_dirs(rdir) and not any(b.name.endswith(PARTIAL_SUFFIX) for b in
                                           (bdir.iterdir() if bdir.is_dir() else [])):
        shutil.rmtree(rdir, ignore_errors=True)
    return freed
