"""Turn an existing ``llama-server …`` command line into a launch profile.

``parse()`` reads the command with the catalog from ``llama-server --help`` (so every
spelling — ``-c``, ``--ctx-size``, ``--no-mmap``, ``--flash-attn=on`` — lands on the same
profile key) and finds the model it points at:

* ``-m PATH`` → that file (an ``hf://`` ref when it lives in the HF cache)
* ``-hf owner/repo[:QUANT]`` / ``-hff FILE`` → the file llama.cpp would pick, looked up in the
  shared HF cache first; if it is not there yet, ``fetch()`` downloads it (plus the mmproj that
  ``-hf`` would have fetched, unless ``--no-mmproj``)
* ``-mu https://huggingface.co/owner/repo/resolve/REV/FILE`` → treated like ``-hf`` with that file
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inari import hfcache, hub
from inari.argspec import ArgSpec, Catalog, parse_help
from inari.config import Config

DEFAULT_QUANT = "Q4_K_M"  # what llama.cpp's -hf picks when no :QUANT is given

# handled by the importer itself rather than copied into [args]
MODEL_KEYS = {"model", "hf-repo", "hf-file", "model-url", "hf-token", "host", "port", "metrics"}
SECRET_KEYS = {"hf-token"}
PATH_KEYS = {"mmproj", "model-draft", "lora", "lora-scaled", "control-vector", "control-vector-scaled",
             "chat-template-file", "grammar-file", "json-schema-file", "slot-save-path", "model-vocoder",
             "path", "media-path"}
SERVER_PROGRAMS = ("", "server", "serve")  # llama-server, `llama server`, `llama serve`
# options that only mean something to llama-cli / the interactive tools — dropped when importing those
CLI_ONLY = {"prompt", "file", "binary-file", "system-prompt", "system-prompt-file", "conversation",
            "no-conversation", "single-turn", "interactive", "interactive-first", "multiline-input", "in-prefix",
            "in-suffix", "in-prefix-bos", "reverse-prompt", "escape", "no-escape", "no-display-prompt", "color",
            "simple-io", "prompt-cache", "prompt-cache-all", "prompt-cache-ro", "show-timings", "no-show-timings",
            "image", "audio", "verbose-prompt", "no-warmup-prompt"}
# always understood, whatever the llama-server build (or the built-in fallback catalog) knows
_IMPORT_HELP = """----- common params -----
-m,    --model FNAME                    model path
-mu,   --model-url MODEL_URL            model download url
-hf,   -hfr, --hf-repo <user>/<model>[:quant]
                                        Hugging Face model repository
-hff,  --hf-file FILE                   Hugging Face model file
-hft,  --hf-token TOKEN                 Hugging Face access token
--mmproj-auto, --no-mmproj, --no-mmproj-auto
                                        whether to use a multimodal projector file
-mm,   --mmproj FILE                    path to a multimodal projector file
-mmu,  --mmproj-url URL                 URL to a multimodal projector file
--host HOST                             ip address to listen
--port PORT                             port to listen
--metrics                               enable prometheus compatible metrics endpoint
-p,    --prompt PROMPT                  prompt to start generation with
-f,    --file FNAME                     a file containing the prompt
-sys,  --system-prompt PROMPT           system prompt
-sysf, --system-prompt-file FNAME       system prompt file
-cnv,  --conversation, -no-cnv, --no-conversation
                                        conversation mode
-st,   --single-turn                    run conversation for a single turn only
-i,    --interactive                    run in interactive mode
-if,   --interactive-first              run in interactive mode and wait for input right away
-mli,  --multiline-input                allows you to write or paste multiple lines
-r,    --reverse-prompt PROMPT          halt generation at PROMPT
--color                                 colorise output
--image, --audio FILE                   path to an image or audio file
"""
_BASE = Catalog(parse_help(_IMPORT_HELP), source="built-in")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PS_ENV_RE = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", re.I)
_SET_ENV_RE = re.compile(r"^set\s+\"?([A-Za-z_][A-Za-z0-9_]*)=(.*?)\"?$", re.I)
_HF_URL_RE = re.compile(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class HfTarget:
    repo: str
    quant: str = ""  # from repo:QUANT
    file: str = ""  # from -hff / a resolve URL


@dataclass
class Imported:
    binary: str = ""
    program: str = ""  # "server", "cli", … — from llama-server / llama-cli / `llama cli`
    model: str = ""  # profile value: hf:// ref, path, or "" when it still has to be fetched
    model_path: Path | None = None
    hf: HfTarget | None = None  # set when the model comes from Hugging Face
    want_mmproj: bool = True  # -hf fetches the repo's mmproj unless --no-mmproj / an explicit --mmproj
    host: str = ""
    port: int = 0
    args: dict[str, Any] = field(default_factory=dict)  # Profile.args (canonical keys)
    extra: list[str] = field(default_factory=list)  # unknown flags, verbatim
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # things the user should know (dropped / kept as-is)
    recognized: list[tuple[str, str]] = field(default_factory=list)  # (as written, profile key=value) for display

    @property
    def is_server(self) -> bool:
        return self.program in SERVER_PROGRAMS

    @property
    def needs_fetch(self) -> bool:
        return self.hf is not None and not self.model

    def suggested_name(self) -> str:
        if self.model_path or (self.model and not self.model.startswith(hfcache.REF_PREFIX)):
            stem = self.model_path.name if self.model_path else self.model.replace("\\", "/")
        elif self.model:
            stem = self.model.rsplit("/", 1)[-1]
        elif self.hf:
            stem = (self.hf.file or self.hf.repo.split("/")[-1] + ("-" + self.hf.quant if self.hf.quant else ""))
        else:
            return "imported"
        stem = Path(stem).name
        stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
        stem = re.sub(r"-GGUF\b", "", stem, flags=re.I)
        return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.").lower() or "imported"


# ───────────────────────── tokenizing ─────────────────────────
def join_lines(text: str) -> str:
    """Undo shell line continuations (bash ``\\``, cmd ``^``, PowerShell backtick) and drop comments."""
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.lstrip().startswith(("#", "::", "REM ", "rem ")) and not out:
            continue
        cont = line.endswith(("\\", "^", "`"))
        if cont:
            line = line[:-1]
        out.append(line + (" " if cont else "\n"))
    return "".join(out).strip()


def tokenize(text: str) -> list[str]:
    """Shell-ish split that keeps Windows backslashes (``C:\\models\\x.gguf``) intact."""
    lex = shlex.shlex(text, posix=True)
    lex.whitespace_split = True
    lex.escape = ""  # backslash is a path separator on Windows, not an escape
    lex.commenters = ""
    return list(lex)


def split_statements(text: str) -> tuple[list[str], dict[str, str]]:
    """The llama-server invocation's tokens, plus env vars set by earlier statements on the same paste."""
    env: dict[str, str] = {}
    stmts = [s.strip() for s in re.split(r"\n|&&|;(?=\s)|;$", join_lines(text)) if s.strip()]
    cmd: list[str] = []
    for s in stmts:
        m = _PS_ENV_RE.match(s) or _SET_ENV_RE.match(s)
        if m:
            env[m.group(1)] = m.group(2).strip().strip("'\"")
            continue
        if s.startswith("export "):
            for tok in tokenize(s[7:]):
                k, _, v = tok.partition("=")
                env[k] = v
            continue
        toks = tokenize(s)
        if not cmd or any("llama" in Path(t).name.lower() for t in toks[:3]):
            cmd = toks
    while cmd and cmd[0] in ("&", ".", "call", "exec", "time", "nohup", "sudo"):
        cmd = cmd[1:]
    while cmd and _ENV_ASSIGN_RE.match(cmd[0]):  # FOO=1 llama-server …
        k, _, v = cmd.pop(0).partition("=")
        env[k] = v
    return cmd, env


# ───────────────────────── parsing ─────────────────────────
def program_of(tokens: list[str]) -> tuple[str, str, int]:
    """(binary, program, tokens used): ``llama-server`` → server, ``llama-cli.exe`` → cli,
    ``llama cli`` / ``llama server`` (the unified binary + subcommand) → cli / server."""
    if not tokens or tokens[0].startswith("-"):
        return "", "", 0
    binary = tokens[0]
    stem = re.sub(r"\.exe$", "", Path(binary.replace("\\", "/")).name.lower())
    if stem == "llama" and len(tokens) > 1 and re.fullmatch(r"[a-z][a-z0-9-]*", tokens[1]):
        return binary, tokens[1].lower(), 2
    if stem.startswith("llama-"):
        return binary, stem[6:], 1
    return binary, "", 1


def _num(v: str) -> Any:
    if _NUM_RE.match(v):
        return float(v) if "." in v else int(v)
    return v


def _takes_value(spec: ArgSpec) -> str:
    """"no" | "yes" | "optional" (``--flash-attn [on|off|auto]`` — bare on old builds)."""
    if spec.kind in ("flag", "toggle"):
        return "no"
    if spec.metavar.startswith("["):
        return "optional"
    return "yes"


def _looks_like_value(tok: str) -> bool:
    return not tok.startswith("-") or bool(_NUM_RE.match(tok))


def _add(args: dict[str, Any], key: str, value: Any) -> None:
    if key in args and args[key] is not True and value is not True:  # repeatable: -ot, --lora, …
        cur = args[key] if isinstance(args[key], list) else [args[key]]
        args[key] = cur + [value]
    else:
        args[key] = value


def parse(text: str, cat: Catalog, cfg: Config, *, cwd: Path | None = None) -> Imported:
    tokens, env = split_statements(text)
    imp = Imported(env=env)
    imp.binary, imp.program, used = program_of(tokens)
    tokens = tokens[used:]
    if not imp.is_server:
        what = f"llama {imp.program}" if Path(imp.binary).stem.lower() == "llama" else Path(imp.binary).name
        imp.notes.append(f"converted from {what} — options llama-server doesn't have are dropped")
    cwd = cwd or Path.cwd()
    i = 0
    model_arg = ""
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok.startswith("-") or tok in ("-", "--"):
            imp.extra.append(tok)
            imp.notes.append(f"stray argument kept as-is: {tok}")
            continue
        flag, eq, inline = tok.partition("=")
        spec, negated = cat.lookup(flag)
        from_base = spec is None
        if spec is None:
            spec, negated = _BASE.lookup(flag)
            if spec is not None and spec.key not in MODEL_KEYS | CLI_ONLY | {"mmproj-auto", "mmproj", "mmproj-url"}:
                spec = None
        if spec is None:
            # unknown to this build: keep verbatim (with its value if one follows) — unless the command
            # was for another tool, where it would only make llama-server refuse to start
            val = [] if eq or i >= len(tokens) or not _looks_like_value(tokens[i]) else [tokens[i]]
            i += len(val)
            if imp.is_server or cat.source == "built-in":
                imp.extra += [tok, *val]
                imp.notes.append(f"{flag} is not in this llama-server's --help — kept verbatim in extra"
                                 if cat.source != "built-in" else
                                 f"{flag} kept verbatim in extra (no llama-server set up to check it against)")
            else:
                imp.notes.append(f"{' '.join([tok, *val])} dropped — not a llama-server option")
            continue
        takes = "no" if negated else _takes_value(spec)
        value: Any = True
        if eq:
            value = inline
        elif takes == "yes" and i < len(tokens):
            value = tokens[i]
            i += 1
        elif takes == "optional" and i < len(tokens) and (tokens[i] in spec.choices or
                                                          (not spec.choices and _looks_like_value(tokens[i]))):
            value = tokens[i]
            i += 1
        elif takes == "yes":
            imp.notes.append(f"{flag} has no value — ignored")
            continue
        elif takes == "optional" and "on" in spec.choices:
            value = "on"  # old builds: bare -fa meant "on"; new ones want a value
        if isinstance(value, str):
            value = _num(value)

        key = spec.key
        if key in CLI_ONLY and (not imp.is_server or cat.get(key) is None):
            imp.notes.append(f"{flag} dropped — only meaningful for llama-cli")
            continue
        if key in SECRET_KEYS:
            imp.notes.append(f"{flag} dropped — secrets don't go into profiles (inari config set hf_token …)")
            continue
        if key == "metrics":
            continue  # inari always adds it
        if key == "model":
            model_arg = str(value)
        elif key == "hf-repo":
            repo, _, quant = str(value).partition(":")
            imp.hf = HfTarget(repo=repo, quant=quant, file=imp.hf.file if imp.hf else "")
        elif key == "hf-file":
            imp.hf = imp.hf or HfTarget(repo="")
            imp.hf.file = str(value)
        elif key == "model-url":
            m = _HF_URL_RE.match(str(value))
            if m:
                imp.hf = HfTarget(repo=m.group(1), file=m.group(2))
            else:
                _add(imp.args, key, value)  # llama-server downloads it itself
                imp.notes.append("--model-url is not a Hugging Face link — llama-server will download it at start")
        elif key == "host":
            imp.host = str(value)
        elif key == "port":
            imp.port = int(value) if isinstance(value, int) else 0
        elif negated:
            if flag.lstrip("-") in ("no-mmproj", "no-mmproj-auto"):
                imp.want_mmproj = False
            if not from_base:  # a spelling only our table knows may not exist in this build
                _add(imp.args, spec.neg or key, True)
        else:
            if key in PATH_KEYS and isinstance(value, str):
                value = _path_value(cfg, value, cwd)
            if key == "mmproj":
                imp.want_mmproj = False
            if key == "mmproj-url" and isinstance(value, str):
                m = _HF_URL_RE.match(value)
                if m:
                    ref = hfcache.make_ref(m.group(1), m.group(2))
                    if hfcache.resolve(cfg, ref):
                        key, value = "mmproj", ref
                        imp.want_mmproj = False
            _add(imp.args, key, value)
        imp.recognized.append((tok if eq or value is True else f"{tok} {value}", key))

    if model_arg:
        p = Path(model_arg).expanduser()
        if not p.is_absolute():
            p = cwd / p
        if p.is_file():
            imp.model_path = p
            imp.model = hub.model_ref(cfg, p)
        else:
            hit = _find_local_by_name(cfg, Path(model_arg).name)
            if hit:
                imp.model_path, imp.model = hit.path, hit.ref
                imp.notes.append(f"{model_arg} not found here — using the local copy {hit.ref}")
            else:
                imp.model = model_arg
                imp.notes.append(f"model file not found: {model_arg}")
        if imp.hf:
            imp.notes.append("both -m and -hf given — using -m")
            imp.hf = None
    elif imp.hf:
        if not imp.hf.repo:
            imp.notes.append("--hf-file without --hf-repo — ignored")
            imp.hf = None
        else:
            _resolve_hf_locally(cfg, imp)
    return imp


def _path_value(cfg: Config, value: str, cwd: Path) -> str:
    if value.startswith(hfcache.REF_PREFIX) or "://" in value:
        return value
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = cwd / p
    if p.is_file():
        return hub.model_ref(cfg, p) if p.suffix.lower() == ".gguf" else str(p)
    return value


def _find_local_by_name(cfg: Config, filename: str) -> hub.LocalModel | None:
    low = filename.lower()
    return next((m for m in hub.local_models(cfg) if m.path.name.lower() == low), None)


# ───────────────────────── Hugging Face ─────────────────────────
def _quant_match(name: str, quant: str) -> bool:
    q = quant.lower()
    stem = Path(name).name.lower()
    return bool(re.search(rf"(^|[-_.]){re.escape(q)}([-_.]|$)", stem))


def _resolve_hf_locally(cfg: Config, imp: Imported) -> None:
    """Use the HF cache when llama.cpp (or inari) already downloaded the file."""
    hf = imp.hf
    assert hf is not None
    cached = [c for c in hfcache.scan(cfg) if c.repo.lower() == hf.repo.lower()]
    models = [c for c in cached if "mmproj" not in c.filename.lower()]
    hit = None
    if hf.file:
        hit = next((c for c in models if c.filename == hf.file or hfcache.shard_names(c.filename)(hf.file)), None)
    else:
        quant = hf.quant or DEFAULT_QUANT
        hits = [c for c in models if _quant_match(c.filename, quant)]
        hit = hits[0] if len(hits) == 1 else None
    if hit:
        imp.model, imp.model_path = hit.ref, hit.path
        if imp.want_mmproj:
            proj = pick_mmproj([c.filename for c in cached if "mmproj" in c.filename.lower()])
            if proj:
                imp.args.setdefault("mmproj", hfcache.make_ref(hit.repo, proj))
                imp.want_mmproj = False


def pick_mmproj(names: list[str]) -> str:
    if not names:
        return ""
    return next((n for n in names if "f16" in n.lower()), sorted(names)[0])


@dataclass
class Plan:
    """What ``fetch()`` would download."""

    repo: str
    model: hub.GgufGroup
    mmproj: hub.GgufGroup | None = None
    choices: list[hub.GgufGroup] = field(default_factory=list)  # set when the quant was ambiguous

    @property
    def size(self) -> int:
        return self.model.size + (self.mmproj.size if self.mmproj else 0)


def plan(cfg: Config, imp: Imported) -> Plan:
    """Look the repo up on Hugging Face and pick the file llama.cpp would pick."""
    hf = imp.hf
    assert hf is not None
    groups = hub.list_groups(hf.repo, cfg)
    models = [g for g in groups if not g.is_mmproj]
    projs = [g for g in groups if g.is_mmproj]
    if not models:
        raise RuntimeError(f"no .gguf files in {hf.repo}")
    chosen: hub.GgufGroup | None = None
    choices: list[hub.GgufGroup] = []
    if hf.file:
        chosen = next((g for g in models if hf.file in g.files or g.name == hf.file), None)
        if chosen is None:
            raise RuntimeError(f"{hf.file} is not in {hf.repo}")
    else:
        quant = hf.quant or DEFAULT_QUANT
        hits = [g for g in models if g.quant.lower() == quant.lower()] or \
               [g for g in models if _quant_match(g.name, quant)] or \
               [g for g in models if quant.lower() in g.name.lower()]
        if len(hits) == 1:
            chosen = hits[0]
        elif hits:
            chosen, choices = hits[0], hits
        elif not hf.quant:  # llama.cpp falls back to the first file when Q4_K_M is missing
            chosen, choices = models[0], models
        else:
            raise RuntimeError(f"no file in {hf.repo} matches :{hf.quant}")
    proj = None
    if imp.want_mmproj and projs:
        pick = pick_mmproj([g.name for g in projs])
        proj = next(g for g in projs if g.name == pick)
    return Plan(repo=hf.repo, model=chosen, mmproj=proj, choices=choices)


def fetch(cfg: Config, imp: Imported, pl: Plan, console, *, backend: str = "") -> None:
    """Download what ``plan()`` chose into the HF cache and point the import at it."""
    path = hub.download_group(pl.repo, pl.model, cfg, console, backend)
    imp.model_path = path
    imp.model = hub.model_ref(cfg, path)
    if pl.mmproj:
        proj = hub.download_group(pl.repo, pl.mmproj, cfg, console, backend)
        imp.args.setdefault("mmproj", hub.model_ref(cfg, proj))
    imp.want_mmproj = False


def link_existing(cfg: Config, imp: Imported, pl: Plan) -> bool:
    """The planned files may already be in the cache under the name we only learn from the listing."""
    p = hfcache.find(cfg, pl.repo, pl.model.entry)
    if not p:
        return False
    imp.model_path, imp.model = p, hfcache.make_ref(pl.repo, pl.model.entry)
    if pl.mmproj:
        mp = hfcache.find(cfg, pl.repo, pl.mmproj.entry)
        if not mp:
            return False
        imp.args.setdefault("mmproj", hfcache.make_ref(pl.repo, pl.mmproj.entry))
    imp.want_mmproj = False
    return True


def same_binary(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return Path(a).name.lower() == Path(b).name.lower()
