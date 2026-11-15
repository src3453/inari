"""Hardware-aware starting values for a new profile (the wizard's "Tuned" column)."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

import psutil

from inari.argspec import Catalog
from inari.gguf import ModelInfo

MiB = 1024 ** 2
GiB = 1024 ** 3
CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
_DEV_RE = re.compile(r"^\s*(\S+?):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)")


@dataclass
class Device:
    id: str  # e.g. CUDA0, ROCm0, Vulkan0
    name: str
    total: int  # bytes
    free: int


@dataclass
class Hardware:
    devices: list[Device] = field(default_factory=list)
    ram: int = 0
    cores: int = 0

    @property
    def vram(self) -> int:
        return sum(d.free for d in self.devices)

    def describe(self) -> str:
        if not self.devices:
            return "no GPU found by llama-server — CPU only"
        return " + ".join(f"{d.name} ({d.total / GiB:.0f} GB)" for d in self.devices)


_HW: dict[str, Hardware] = {}


def detect(binary: str | None) -> Hardware:
    """GPUs as llama-server sees them (works for CUDA, ROCm, Vulkan, Metal, SYCL)."""
    key = binary or ""
    if key in _HW:
        return _HW[key]
    hw = Hardware(ram=psutil.virtual_memory().total, cores=psutil.cpu_count(logical=False) or 0)
    if binary:
        try:
            r = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30, stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in (r.stdout + "\n" + r.stderr).splitlines():
                m = _DEV_RE.match(line)
                if m and not m.group(1).upper().startswith("CPU"):
                    hw.devices.append(Device(m.group(1), m.group(2), int(m.group(3)) * MiB, int(m.group(4)) * MiB))
        except (OSError, subprocess.SubprocessError):
            pass
    _HW[key] = hw
    return hw


def gb(n: float) -> str:
    return f"{n / GiB:.1f} GB" if n >= GiB else f"{n / MiB:.0f} MB"


def tune(cat: Catalog, hw: Hardware, info: ModelInfo | None, size: int) -> tuple[dict[str, Any], dict[str, str]]:
    """(values, reasons). Only settings worth changing from llama.cpp's own defaults are returned."""
    vals: dict[str, Any] = {}
    why: dict[str, str] = {}
    fit = "fit" in cat and cat.get("fit").default in ("on", "")  # type: ignore[union-attr]
    vram = hw.vram
    budget = vram * 0.9 if vram else 0

    # ── context ──
    if info and info.ctx_train and "ctx-size" in cat:
        cap = info.ctx_train
        if budget and size:
            room = budget - size
            ctx = 4096
            for c in CTX_STEPS:
                if c <= cap and info.kv_bytes(c) <= max(room, 0):
                    ctx = c
            if room <= 0:  # weights alone don't fit: keep the KV cache modest
                ctx = min(cap, 16384)
                why["ctx-size"] = "the weights already exceed GPU memory, so the context is kept small"
            else:
                why["ctx-size"] = (f"largest step that fits next to the weights in {gb(vram)} free VRAM "
                                   f"(model supports {cap:,})")
        else:
            ctx = min(cap, 32768)
            why["ctx-size"] = f"model supports {cap:,}; 32K keeps memory use moderate"
        vals["ctx-size"] = min(ctx, cap)

    # ── GPU offload ──
    if hw.devices and size:
        if fit:
            why["fit"] = "llama.cpp fits layers / MoE experts into VRAM automatically (--fit on)"
        elif "n-gpu-layers" in cat:
            if size * 1.05 < budget:
                vals["n-gpu-layers"] = 999
                why["n-gpu-layers"] = f"the whole model ({gb(size)}) fits in {gb(vram)} VRAM"
            elif info and info.moe and "n-cpu-moe" in cat and info.layers:
                vals["n-gpu-layers"] = 999
                n = min(info.layers, math.ceil(info.layers * (size - budget * 0.85) / size))
                vals["n-cpu-moe"] = n
                why["n-cpu-moe"] = f"keep experts of {n}/{info.layers} layers in RAM so the rest fits in VRAM"
            elif info and info.layers:
                n = max(0, int(info.layers * budget * 0.85 / size))
                vals["n-gpu-layers"] = n
                why["n-gpu-layers"] = f"≈{n}/{info.layers} layers fit in {gb(vram)} VRAM"
    elif not hw.devices and "n-gpu-layers" in cat and hw.cores:
        vals["n-gpu-layers"] = 0
        why["n-gpu-layers"] = "no GPU detected"
        if "threads" in cat:
            vals["threads"] = hw.cores
            why["threads"] = f"one thread per physical core ({hw.cores})"

    # ── KV cache: quantize when it would eat a big part of VRAM ──
    ctx = vals.get("ctx-size", 0)
    if info and ctx and vram and "cache-type-k" in cat and info.kv_bytes(ctx) > vram * 0.25:
        vals["cache-type-k"] = vals["cache-type-v"] = "q8_0"
        why["cache-type-k"] = why["cache-type-v"] = (f"f16 KV for {ctx:,} tokens ≈ {gb(info.kv_bytes(ctx))}; "
                                                      "q8_0 halves it with little quality loss")

    # ── chat ──
    jinja = cat.get("jinja")
    if jinja and jinja.default not in ("enabled", "true"):
        vals["jinja"] = True
        why["jinja"] = "use the model's own chat template (needed for tool calls)"
    return vals, why
