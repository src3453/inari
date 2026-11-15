"""Hugging Face GGUF discovery and download."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from inari.config import Config

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
        g = groups.setdefault(key, GgufGroup(name=key))
        g.files.append(fn)
        g.size += s.size or 0
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
    return cfg.models_path() / repo.replace("/", "__")


def download_group(repo: str, group: GgufGroup, cfg: Config, console: Console) -> Path:
    from huggingface_hub import hf_hub_download

    dest = local_dir_for(repo, cfg)
    first: Path | None = None
    for i, fn in enumerate(sorted(group.files), 1):
        tag = f" [{i}/{len(group.files)}]" if len(group.files) > 1 else ""
        console.print(f"[accent]↓[/] [text]{fn}[/]{tag}")
        path = Path(hf_hub_download(repo_id=repo, filename=fn, local_dir=dest, token=_token(cfg)))
        first = first or path
    assert first is not None
    return first


def pull(spec: str, cfg: Config, console: Console, *, yes: bool = False) -> tuple[Path | None, Path | None]:
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

    console.print(f"[dim]→ {local_dir_for(repo, cfg)}[/]  [text]{chosen.name}[/] [accent2]{human(chosen.size)}[/]")
    model_path = download_group(repo, chosen, cfg, console)

    mmproj_path: Path | None = None
    if projs:
        pick = projs[0]
        for p in projs:  # prefer F16 projector
            if "f16" in p.name.lower():
                pick = p
                break
        if yes or Confirm.ask(f"[accent]Vision projector found[/] ({pick.name}, {human(pick.size)}). Download it?",
                              default=False, console=console):
            mmproj_path = download_group(repo, pick, cfg, console)

    console.print(f"[good]✔ Downloaded[/] [text]{model_path}[/]")
    return model_path, mmproj_path


def local_models(cfg: Config) -> list[tuple[Path, int]]:
    root = cfg.models_path()
    out: list[tuple[Path, int]] = []
    seen: set[str] = set()
    for p in sorted(root.rglob("*.gguf")):
        if ".cache" in p.parts:
            continue
        m = SPLIT_RE.match(p.name)
        if m:
            key = str(p.parent / m.group("base"))
            if key in seen:
                continue
            seen.add(key)
            size = sum(q.stat().st_size for q in p.parent.glob(m.group("base") + "-*-of-*.gguf"))
            out.append((p, size))
        else:
            out.append((p, p.stat().st_size))
    return out


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
    """'done' if every shard is on disk, 'partial' if a .part exists, else ''."""
    d = local_dir_for(repo, cfg)
    if all((d / fn).exists() for fn in group.files):
        return "done"
    if any((d / (fn + ".part")).exists() for fn in group.files):
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
    speed: float = 0.0  # bytes/s (EWMA)
    current: str = ""
    profile: str = ""  # profile created from this job
    _cancel: bool = field(default=False, repr=False)

    @property
    def total(self) -> int:
        return self.group.size + (self.mmproj.size if self.mmproj else 0)

    @property
    def frac(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def active(self) -> bool:
        return self.status in ("queued", "downloading")

    def cancel(self) -> None:
        self._cancel = True


class Cancelled(Exception):
    pass


class Downloader:
    """Sequential download queue on a worker thread (resumable via ``.part`` files)."""

    def __init__(self, cfg: Config) -> None:
        import queue
        import threading

        self.cfg = cfg
        self.jobs: list[Job] = []
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, repo: str, group: GgufGroup, mmproj: GgufGroup | None = None) -> Job:
        job = Job(repo=repo, group=group, mmproj=mmproj)
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
                job.status, job.speed, job.current = "done", 0.0, ""
            except Cancelled:
                job.status, job.speed = "cancelled", 0.0
            except Exception as e:  # network / auth / disk
                job.status, job.error, job.speed = "failed", str(e) or type(e).__name__, 0.0

    def _group(self, job: Job, group: GgufGroup) -> Path:
        dest = local_dir_for(job.repo, self.cfg)
        first: Path | None = None
        for fn in sorted(group.files):
            path = self._file(job, fn, dest)
            first = first or path
        assert first is not None
        return first

    def _file(self, job: Job, fn: str, dest: Path) -> Path:
        import time

        import httpx
        from huggingface_hub import hf_hub_url

        target = dest / fn
        job.current = fn
        if target.exists():
            job.done += target.stat().st_size
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"user-agent": "inari"}
        tok = _token(self.cfg)
        if tok:
            headers["authorization"] = f"Bearer {tok}"  # httpx drops it on cross-origin redirects
        if have:
            headers["range"] = f"bytes={have}-"
        url = hf_hub_url(job.repo, fn)
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=120.0)) as r:
            if have and r.status_code == 416:  # already complete
                part.replace(target)
                job.done += have
                return target
            r.raise_for_status()
            if have and r.status_code != 206:  # server ignored Range -> start over
                have = 0
            job.done += have
            last_t, last_b = time.monotonic(), job.done
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    if job._cancel:
                        raise Cancelled()
                    f.write(chunk)
                    job.done += len(chunk)
                    now = time.monotonic()
                    if now - last_t >= 0.5:
                        rate = (job.done - last_b) / (now - last_t)
                        job.speed = rate if not job.speed else job.speed * 0.6 + rate * 0.4
                        last_t, last_b = now, job.done
        part.replace(target)
        return target
