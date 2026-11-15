"""The catalog of llama-server arguments, read from ``llama-server --help``.

Each build of llama.cpp has its own set of flags, so the setup wizard asks the
configured binary for its help text, parses it once and caches the result in
``~/.inari/cache``. Without a binary a small built-in catalog is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inari.config import Config, home

PARSER_VERSION = 3
DESC_COL = 40  # llama.cpp prints descriptions from this column

# managed by inari itself (Profile.model / host / port, --metrics) or meaningless in a profile
HIDDEN = {"help", "usage", "version", "cache-list", "completion-bash", "list-devices", "model", "host", "port", "metrics",
          "license"}
SECTIONS = {"common": "common", "sampling": "sampling", "speculative": "speculative",
            "example-specific": "server"}


@dataclass
class ArgSpec:
    key: str  # canonical name used in profiles: the long flag without dashes
    names: list[str] = field(default_factory=list)  # every positive spelling, with dashes
    neg: str = ""  # e.g. "no-mmap" — set for on/off pairs
    neg_names: list[str] = field(default_factory=list)
    metavar: str = ""
    choices: list[str] = field(default_factory=list)
    default: str = ""
    help: str = ""
    env: str = ""
    section: str = "common"
    experimental: bool = False

    @property
    def kind(self) -> str:
        """flag (bare switch) | toggle (on/off pair) | choice | value."""
        if self.neg:
            return "toggle"
        if self.choices:
            return "choice"
        if not self.metavar:
            return "flag"
        return "value"

    @property
    def flag(self) -> str:
        return "--" + self.key if len(self.key) > 1 else "-" + self.key

    def short(self) -> str:
        s = [n for n in self.names if not n.startswith("--")]
        return s[0] if s else ""


class Catalog:
    def __init__(self, specs: list[ArgSpec], *, version: str = "", source: str = "") -> None:
        self.specs = specs
        self.version = version
        self.source = source  # path of the binary, or "built-in"
        self.by_key = {s.key: s for s in specs}
        self._alias: dict[str, tuple[ArgSpec, bool]] = {}
        for s in specs:
            for n in s.names:
                self._alias.setdefault(n.lstrip("-"), (s, False))
            for n in s.neg_names:
                self._alias.setdefault(n.lstrip("-"), (s, True))

    def __contains__(self, key: str) -> bool:
        return key in self.by_key

    def get(self, key: str) -> ArgSpec | None:
        return self.by_key.get(key)

    def lookup(self, name: str) -> tuple[ArgSpec | None, bool]:
        """Any spelling (``-c``, ``ctx_size``, ``--no-webui``…) → (spec, is_negative_form)."""
        n = name.strip().lstrip("-")
        hit = self._alias.get(n) or self._alias.get(n.replace("_", "-"))
        return hit if hit else (None, False)

    @property
    def visible(self) -> list[ArgSpec]:
        return [s for s in self.specs if s.key not in HIDDEN]


# ───────────────────────── parsing ─────────────────────────
_SECTION_RE = re.compile(r"^-{3,}\s*(.+?)\s+params\s*-{3,}")
_DEFAULT_RE = re.compile(r"default:\s*([^)\n]*)", re.I)
_ENV_RE = re.compile(r"\(env:\s*([A-Z0-9_]+)\)")
_LINK_RE = re.compile(r"\[\(more\s*info\)\]\([^)]*\)", re.I)


def _clean_default(v: str) -> str:
    v = v.split(" (", 1)[0].strip().rstrip(".")
    m = re.match(r"^['\"]([^'\"]*)['\"]?", v)
    if m:  # 'auto' (detect from template) -> auto
        return m.group(1).strip()
    # "0, 0 = loaded from model" -> "0" ; "'auto'" -> "auto"
    if re.match(r"^[-\w.'\"]+,\s", v):
        v = v.split(",", 1)[0]
    return v.strip().strip("'\"")


def _split_opt(opt: str) -> tuple[list[str], str]:
    names, metavar = [], ""
    for seg in re.split(r",\s+", opt.strip()):
        seg = seg.strip()
        if not seg.startswith("-"):
            metavar = (metavar + " " + seg).strip()
            continue
        name, _, rest = seg.partition(" ")
        names.append(name)
        if rest.strip():
            metavar = rest.strip()
    return names, metavar


def _choices(metavar: str, desc: str) -> list[str]:
    m = re.fullmatch(r"[\[{<]([^\]}>]+)[\]}>]", metavar)
    if m and ("|" in m.group(1) or "," in m.group(1)):
        return [c.strip() for c in re.split(r"[|,]", m.group(1)) if c.strip() and c.strip() != "..."]
    m = re.search(r"allowed values:\s*([\w.,\s]+?)(?:\(|$)", desc)
    if m:
        return [c.strip() for c in m.group(1).split(",") if c.strip()]
    return []


def parse_help(text: str) -> list[ArgSpec]:
    entries: list[tuple[str, list[str], str]] = []  # (option column, description lines, section)
    section = "common"
    for raw in text.replace("\t", "    ").splitlines():
        line = raw.rstrip()
        sm = _SECTION_RE.match(line)
        if sm:
            section = SECTIONS.get(sm.group(1).strip().lower(), sm.group(1).strip().lower())
            continue
        if line.startswith("-"):
            if len(line) > DESC_COL and line[DESC_COL - 2:DESC_COL] == "  " and line[DESC_COL] != " ":
                entries.append((line[:DESC_COL], [line[DESC_COL:]], section))
            else:
                entries.append((line, [], section))
        elif line.startswith(" ") and line.strip() and entries:
            entries[-1][1].append(line.strip())

    out: list[ArgSpec] = []
    seen: set[str] = set()
    for opt, lines, section in entries:
        names, metavar = _split_opt(opt)
        if not names:
            continue
        neg_names = [n for i, n in enumerate(names) if n.startswith("--no-")
                     or (not n.startswith("--") and i + 1 < len(names) and names[i + 1].startswith("--no-")
                         and any(x.startswith("--") and not x.startswith("--no-") for x in names[:i]))]
        pos = [n for n in names if n not in neg_names]
        longs = [n for n in pos if n.startswith("--")]
        if pos:
            key = (longs[-1] if longs else pos[0]).lstrip("-")  # --gpu-layers, --n-gpu-layers -> n-gpu-layers
            neg = next((n.lstrip("-") for n in reversed(neg_names) if n.startswith("--")), "")
        else:  # only negative spellings, e.g. --no-host: a plain switch
            key, neg, pos, neg_names = neg_names[-1].lstrip("-"), "", neg_names, []
        if key in seen:
            continue
        seen.add(key)
        joined = " ".join(lines)
        dm = _DEFAULT_RE.search(joined)
        em = _ENV_RE.search(joined)
        desc = _ENV_RE.sub("", _LINK_RE.sub("", joined))
        desc = re.sub(r"\(\s*default:[^)]*\)", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        out.append(ArgSpec(key=key, names=pos, neg=neg, neg_names=neg_names, metavar=metavar,
                           choices=_choices(metavar, joined), default=_clean_default(dm.group(1)) if dm else "",
                           help=desc, env=em.group(1) if em else "", section=section,
                           experimental="experimental" in joined.lower()))
    return out


# ───────────────────────── loading ─────────────────────────
def _cache_file(binary: str) -> Path:
    try:
        st = Path(binary).stat()
        sig = f"{Path(binary).resolve()}|{st.st_size}|{int(st.st_mtime)}|{PARSER_VERSION}"
    except OSError:
        sig = f"{binary}|{PARSER_VERSION}"
    d = home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"llama-args-{hashlib.sha1(sig.encode()).hexdigest()[:16]}.json"


def _run(binary: str, *args: str) -> str:
    r = subprocess.run([binary, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=20, stdin=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return (r.stdout or "") + "\n" + (r.stderr or "")


def load(cfg: Config, binary: str | None = None) -> Catalog:
    """Catalog for the configured llama-server (cached per binary build)."""
    binary = binary or cfg.server_binary()
    if binary:
        cf = _cache_file(binary)
        if cf.exists():
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                return Catalog([ArgSpec(**d) for d in data["specs"]], version=data.get("version", ""),
                               source=binary)
            except Exception:
                pass
        try:
            specs = parse_help(_run(binary, "--help"))
            ver = next((ln.strip() for ln in _run(binary, "--version").splitlines()
                        if ln.strip().startswith("version")), "")
        except Exception:
            specs, ver = [], ""
        if len(specs) > 20:
            try:
                cf.write_text(json.dumps({"version": ver, "specs": [asdict(s) for s in specs]}), encoding="utf-8")
            except OSError:
                pass
            return Catalog(specs, version=ver, source=binary)
    return Catalog(parse_help(FALLBACK_HELP), source="built-in")


# Used when no llama-server is available: the commonly tuned options, in --help format.
FALLBACK_HELP = """\
----- common params -----
-t,    --threads N                      number of CPU threads to use during generation (default: -1)
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-n,    --predict, --n-predict N         number of tokens to predict (default: -1, -1 = infinity)
-b,    --batch-size N                   logical maximum batch size (default: 2048)
-ub,   --ubatch-size N                  physical maximum batch size (default: 512)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
-ctk,  --cache-type-k TYPE              KV cache data type for K allowed values: f32, f16, bf16, q8_0, q4_0,
                                        q4_1, iq4_nl, q5_0, q5_1 (default: f16)
-ctv,  --cache-type-v TYPE              KV cache data type for V allowed values: f32, f16, bf16, q8_0, q4_0,
                                        q4_1, iq4_nl, q5_0, q5_1 (default: f16)
--mlock                                 force system to keep model in RAM rather than swapping or compressing
--mmap, --no-mmap                       whether to memory-map model (default: enabled)
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type
-cmoe, --cpu-moe                        keep all Mixture of Experts (MoE) weights in the CPU
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the
                                        CPU
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM (default: auto)
-sm,   --split-mode {none,layer,row}    how to split the model across multiple GPUs (default: layer)
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU, e.g. 3,1
-mg,   --main-gpu INDEX                 the GPU to use for the model (default: 0)
-m,    --model FNAME                    model path to load

----- sampling params -----
--temp, --temperature N                 temperature (default: 0.80)
--top-k N                               top-k sampling (default: 40, 0 = disabled)
--top-p N                               top-p sampling (default: 0.95, 1.0 = disabled)
--min-p N                               min-p sampling (default: 0.05, 0.0 = disabled)
--repeat-penalty N                      penalize repeat sequence of tokens (default: 1.00, 1.0 = disabled)

----- example-specific params -----
-np,   --parallel N                     number of server slots (default: -1, -1 = auto)
-cb,   --cont-batching, -nocb, --no-cont-batching
                                        whether to enable continuous batching (default: enabled)
-mm,   --mmproj FILE                    path to a multimodal projector file
-a,    --alias STRING                   set model name aliases, comma-separated (to be used by API)
--host HOST                             ip address to listen (default: 127.0.0.1)
--port PORT                             port to listen (default: 8080)
--webui, --no-webui                     whether to enable the Web UI (default: enabled)
--api-key KEY                           API key to use for authentication (default: none)
--metrics                               enable prometheus compatible metrics endpoint (default: disabled)
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: disabled)
-rea,  --reasoning [on|off|auto]        use reasoning/thinking in the chat (default: auto)
--reasoning-budget N                    token budget for thinking: -1 for unrestricted (default: -1)
"""


def to_args(values: dict[str, Any], cat: Catalog) -> dict[str, Any]:
    """Wizard values (canonical keys; toggles as True/False) → Profile.args."""
    out: dict[str, Any] = {}
    for key, v in values.items():
        if v is None:
            continue
        spec = cat.get(key)
        if spec and spec.kind == "toggle":
            if v is True:
                out[key] = True
            elif v is False:
                out[spec.neg] = True
            continue
        if spec and spec.kind == "flag":
            if v is True:
                out[key] = True
            continue
        out[key] = v
    return out


def from_args(args: dict[str, Any], cat: Catalog) -> dict[str, Any]:
    """Profile.args (any spelling) → canonical wizard values. Unknown keys are kept as-is."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        spec, negated = cat.lookup(k)
        if spec is None:
            out[k] = v
            continue
        if negated:
            if v is True:
                out[spec.key] = False
            continue
        if spec.kind in ("toggle", "flag") and v is False:
            continue  # false = omitted = llama.cpp default
        out[spec.key] = v
    return out
