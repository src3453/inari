"""Paths and global settings.

Everything lives under ``$INARI_HOME`` (default: ``~/.inari``)::

    ~/.inari/
      config.toml      global settings (llama-server path, models dir, theme)
      profiles/*.toml  launch profiles
      models/          downloaded GGUF files
      runs/*.json      records of servers started by inari
      logs/*.log       llama-server output
"""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w


def home() -> Path:
    p = Path(os.environ.get("INARI_HOME", Path.home() / ".inari")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def profiles_dir() -> Path:
    return _sub("profiles")


def runs_dir() -> Path:
    return _sub("runs")


def logs_dir() -> Path:
    return _sub("logs")


def config_path() -> Path:
    return home() / "config.toml"


@dataclass
class Config:
    llama_server: str = ""  # path to llama-server(.exe); empty = search PATH
    models_dir: str = ""  # empty = ~/.inari/models
    theme: str = "inari"
    hf_token: str = ""  # optional; HF_TOKEN env var also works

    @classmethod
    def load(cls) -> "Config":
        p = config_path()
        if not p.exists():
            return cls()
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        config_path().write_text(tomli_w.dumps(asdict(self)), encoding="utf-8")

    def models_path(self) -> Path:
        env = os.environ.get("INARI_MODELS")
        p = Path(env or self.models_dir or (home() / "models")).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def server_binary(self, override: str = "") -> str | None:
        for cand in (override, self.llama_server, os.environ.get("LLAMA_SERVER", "")):
            if cand:
                p = Path(cand).expanduser()
                if p.is_dir():
                    for name in ("llama-server.exe", "llama-server"):
                        if (p / name).exists():
                            return str(p / name)
                if p.exists():
                    return str(p)
                found = shutil.which(cand)
                if found:
                    return found
        return shutil.which("llama-server")
