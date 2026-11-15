"""Launch profiles: named, saved sets of llama-server arguments.

A profile is a TOML file in ``~/.inari/profiles/<name>.toml``::

    description = "Qwen3 8B, full GPU offload"
    model = "hf://unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"   # HF cache ref, path relative to models dir, or absolute
    host = "127.0.0.1"
    port = 8080
    server_bin = ""          # optional per-profile llama-server override
    extra = ["--no-webui"]   # raw args appended verbatim

    [args]                   # --key value ; true = bare flag ; false = omitted
    ctx-size = 32768
    n-gpu-layers = 99
    jinja = true
    flash-attn = "on"
    "-ot" = "exps=CPU"       # keys starting with '-' are used verbatim

    [env]                    # extra environment variables for the server
    GGML_VK_VISIBLE_DEVICES = "0"
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from inari import hfcache
from inari.config import Config, profiles_dir

TOP_LEVEL = ("description", "model", "host", "port", "server_bin")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_ARGS: dict[str, Any] = {
    "ctx-size": 8192,
    "n-gpu-layers": 99,
    "jinja": True,
}


@dataclass
class Profile:
    name: str
    model: str = ""
    description: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    server_bin: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # ---------- persistence ----------
    @staticmethod
    def path_for(name: str) -> Path:
        return profiles_dir() / f"{name}.toml"

    @classmethod
    def load(cls, name: str) -> "Profile":
        p = cls.path_for(name)
        if not p.exists():
            raise FileNotFoundError(f"profile '{name}' not found ({p})")
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        return cls(
            name=name,
            model=str(data.get("model", "")),
            description=str(data.get("description", "")),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 8080)),
            server_bin=str(data.get("server_bin", "")),
            args=dict(data.get("args", {})),
            extra=[str(x) for x in data.get("extra", [])],
            env={k: str(v) for k, v in data.get("env", {}).items()},
        )

    def save(self) -> Path:
        if not NAME_RE.match(self.name):
            raise ValueError(f"invalid profile name: {self.name!r}")
        data: dict[str, Any] = {
            "description": self.description,
            "model": self.model,
            "host": self.host,
            "port": self.port,
        }
        if self.server_bin:
            data["server_bin"] = self.server_bin
        if self.extra:
            data["extra"] = self.extra
        data["args"] = self.args
        if self.env:
            data["env"] = self.env
        p = self.path_for(self.name)
        p.write_text(tomli_w.dumps(data), encoding="utf-8")
        return p

    # ---------- editing ----------
    def set(self, key: str, raw: str) -> None:
        append = key.endswith("+")
        key = key.rstrip("+")
        value = parse_value(raw)
        if key in TOP_LEVEL:
            if key == "port":
                value = int(value)
            setattr(self, key, str(value) if key != "port" else value)
        elif key == "extra":
            self.extra = self.extra + [raw] if append else raw.split()
        elif key.startswith("env."):
            self.env[key[4:]] = raw
        else:
            key = normalize_key(key)
            if append:
                cur = self.args.get(key)
                cur = [] if cur is None else (cur if isinstance(cur, list) else [cur])
                self.args[key] = cur + [value]
            else:
                self.args[key] = value

    def unset(self, key: str) -> bool:
        if key.startswith("env."):
            return self.env.pop(key[4:], None) is not None
        if key == "extra":
            had = bool(self.extra)
            self.extra = []
            return had
        return self.args.pop(normalize_key(key), None) is not None

    # ---------- command line ----------
    def model_path(self, cfg: Config) -> Path | None:
        if not self.model:
            return None
        parsed = hfcache.parse_ref(self.model)
        if parsed:  # unresolved -> a path that doesn't exist, so callers report it missing
            return hfcache.resolve(cfg, self.model) or hfcache.repo_dir(cfg, parsed[0]) / parsed[1]
        p = Path(self.model).expanduser()
        if not p.is_absolute():
            p = cfg.models_path() / p
        return p

    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}"

    def build_command(self, cfg: Config, binary: str | None = None) -> list[str]:
        binary = binary or cfg.server_binary(self.server_bin)
        if not binary:
            raise FileNotFoundError(
                "llama-server not found. Install the official build with: inari update, or run: "
                "inari config set llama_server <path-to-llama-server>"
            )
        cmd = [binary]
        mp = self.model_path(cfg)
        if mp is not None:
            cmd += ["--model", str(mp)]
        cmd += ["--host", self.host, "--port", str(self.port)]
        for key, value in self.args.items():
            cmd += render_arg(key, _resolve_refs(value, cfg))
        cmd += self.extra
        if "--metrics" not in cmd:
            cmd.append("--metrics")  # the monitor reads /metrics
        if not any(f in cmd for f in ("-lv", "--log-verbosity", "-v", "--verbose", "--log-verbose")):
            # recent builds hide device / buffer / per-request lines below verbosity 4
            cmd += ["-lv", "4"]
        return cmd


def _resolve_refs(value: Any, cfg: Config) -> Any:
    """``hf://owner/repo/file`` values (e.g. mmproj) -> local paths in the HF cache."""
    if isinstance(value, list):
        return [_resolve_refs(v, cfg) for v in value]
    if isinstance(value, str) and value.startswith(hfcache.REF_PREFIX):
        p = hfcache.resolve(cfg, value)
        return str(p) if p else value
    return value


def normalize_key(key: str) -> str:
    if key.startswith("-"):
        return key
    return key.strip().replace("_", "-")


def render_arg(key: str, value: Any) -> list[str]:
    flag = key if key.startswith("-") else ("-" + key if len(key) == 1 else "--" + key)
    if value is True:
        return [flag]
    if value is False or value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out += render_arg(key, v)
        return out
    return [flag, str(value)]


def parse_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def list_profiles() -> list[str]:
    return sorted(p.stem for p in profiles_dir().glob("*.toml"))


def new_profile(name: str, model: str = "", **kw: Any) -> Profile:
    p = Profile(name=name, model=model, args=dict(DEFAULT_ARGS))
    for k, v in kw.items():
        setattr(p, k, v)
    return p
