"""Read the few GGUF header fields the setup wizard needs (no external deps).

Only the metadata key/value section is read, and reading stops at the tokenizer
block, so even huge models are inspected in milliseconds.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class ModelInfo:
    arch: str = ""
    name: str = ""
    size_label: str = ""
    ctx_train: int = 0
    layers: int = 0
    experts: int = 0  # 0 = dense
    experts_used: int = 0
    embd: int = 0
    heads: int = 0
    heads_kv: int = 0
    key_len: int = 0
    value_len: int = 0
    file_type: int = -1
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def moe(self) -> bool:
        return self.experts > 1

    def kv_bytes(self, ctx: int, k_type: str = "f16", v_type: str = "f16") -> int:
        """Rough KV-cache size for ``ctx`` tokens (full attention on every layer)."""
        if not (self.layers and (self.heads_kv or self.heads)):
            return 0
        hk = self.heads_kv or self.heads
        head = self.embd // self.heads if self.heads and self.embd else 128
        kl, vl = self.key_len or head, self.value_len or head
        return int(ctx * self.layers * hk * (kl * bpe(k_type) + vl * bpe(v_type)))


_BPE = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "f8_e4m3": 1.0, "q8_0": 34 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32,
        "q4_0": 18 / 32, "q4_1": 20 / 32, "iq4_nl": 18 / 32}


def bpe(t: str) -> float:
    return _BPE.get(str(t).lower(), 2.0)


def _read(f: BinaryIO, fmt: str) -> Any:
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) < n:
        raise EOFError
    return struct.unpack(fmt, b)[0]


def _str(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, t: int) -> Any:
    if t in _SCALAR:
        return _read(f, _SCALAR[t])
    if t == 8:
        return _str(f)
    if t == 9:
        it, n = _read(f, "<I"), _read(f, "<Q")
        if it == 8:
            _skip_strings(f, n)
            return None
        if it in _SCALAR and n > 4096:  # big tables are not needed; skip them
            f.seek(n * struct.calcsize(_SCALAR[it]), 1)
            return None
        return [_value(f, it) for _ in range(n)]  # e.g. per-layer head counts
    raise ValueError(f"unknown gguf type {t}")


def _skip_strings(f: BinaryIO, n: int) -> None:  # noqa: D401
    for _ in range(n):
        f.seek(_read(f, "<Q"), 1)


def read_info(path: str | Path) -> ModelInfo | None:
    """Metadata of a .gguf (for split models pass the first part). None if unreadable."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version = _read(f, "<I")
            if version < 2:
                return None
            _read(f, "<Q")  # tensors
            n_kv = _read(f, "<Q")
            meta: dict[str, Any] = {}
            for _ in range(n_kv):
                key = _str(f)
                if key.startswith("tokenizer."):
                    break
                meta[key] = _value(f, _read(f, "<I"))
    except (OSError, EOFError, ValueError, struct.error):
        return None
    arch = str(meta.get("general.architecture", ""))

    def a(name: str) -> int:
        v = meta.get(f"{arch}.{name}", 0)
        if isinstance(v, list):
            v = max((x for x in v if isinstance(x, (int, float))), default=0)
        return int(v) if isinstance(v, (int, float)) else 0

    return ModelInfo(arch=arch, name=str(meta.get("general.name", "")),
                     size_label=str(meta.get("general.size_label", "")), ctx_train=a("context_length"),
                     layers=a("block_count"), experts=a("expert_count"), experts_used=a("expert_used_count"),
                     embd=a("embedding_length"), heads=a("attention.head_count"),
                     heads_kv=a("attention.head_count_kv"), key_len=a("attention.key_length"),
                     value_len=a("attention.value_length"), file_type=int(meta.get("general.file_type", -1) or -1),
                     meta=meta)
