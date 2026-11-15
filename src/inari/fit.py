"""Rough "will this GGUF run here?" estimates for the Hub, based on total RAM + VRAM."""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass

import psutil

from inari import tuning

GiB = 1024 ** 3
# Extra memory beyond the weights: KV cache for a modest context + compute buffers.
OVERHEAD_FIXED = int(0.6 * GiB)
OVERHEAD_FRAC = 0.08
VRAM_USABLE = 0.92  # leave room for the desktop / driver
RAM_RESERVE = 3 * GiB  # OS + apps
RAM_RESERVE_FRAC = 0.15
GOOD_BPW = 3.4  # below this, quality usually drops noticeably
_FULL_RE = re.compile(r"^(F32|F16|BF16)$")


@dataclass
class Spec:
    ram: int = 0  # bytes, total
    vram: int = 0  # bytes, total over all GPUs
    gpus: str = ""
    unified: bool = False  # Apple Silicon: GPU memory *is* RAM
    source: str = ""  # "llama-server" | "nvml" | ""

    @property
    def ram_usable(self) -> int:
        return max(0, self.ram - max(RAM_RESERVE, int(self.ram * RAM_RESERVE_FRAC)))

    @property
    def vram_usable(self) -> int:
        return int(self.vram * VRAM_USABLE)

    @property
    def total_usable(self) -> int:
        if self.unified:
            return max(self.ram_usable, self.vram_usable)
        return self.ram_usable + self.vram_usable

    def describe(self) -> str:
        parts = [f"RAM {self.ram / GiB:.0f} GB"]
        if self.vram:
            parts.append(f"VRAM {self.vram / GiB:.0f} GB" + (f" ({self.gpus})" if self.gpus else ""))
        else:
            parts.append("no GPU detected" if self.source else "GPU: detecting…")
        return " · ".join(parts)


def need(size: int) -> int:
    """Memory a model of ``size`` bytes needs to run with a modest context."""
    return int(size * (1 + OVERHEAD_FRAC)) + OVERHEAD_FIXED


@dataclass
class Fit:
    level: str  # gpu | split | cpu | tight | no
    label: str
    note: str


def judge(size: int, spec: Spec) -> Fit:
    n = need(size)
    if spec.vram and n <= spec.vram_usable:
        return Fit("gpu", "◆ GPU", "fits entirely in VRAM — fastest")
    if n <= spec.total_usable:
        if spec.vram and not spec.unified:
            pct = int(100 * spec.vram_usable / n)
            return Fit("split", "◐ GPU+RAM", f"~{pct}% on GPU, rest in RAM — slower")
        return Fit("cpu", "○ CPU", "runs from RAM on the CPU — slow")
    if n <= spec.total_usable + spec.ram * RAM_RESERVE_FRAC + RAM_RESERVE:
        return Fit("tight", "△ tight", "barely fits — close other apps, keep the context small")
    return Fit("no", "✖ too big", f"needs ~{n / GiB:.0f} GB, have ~{spec.total_usable / GiB:.0f} GB")


def _bpw(size: int, quant: str, params: int) -> float:
    if params:
        return size * 8 / params
    m = re.search(r"(\d+)", quant)
    return float(m.group(1)) + 0.5 if m else 4.5


def recommend(groups, spec: Spec, params: int = 0, extra: int = 0) -> int | None:
    """Index of the quant to suggest for this machine (None if nothing fits).

    Prefers the biggest (best quality) quant that runs fully on the GPU, as long as it keeps
    roughly ≥3.4 bits/weight; otherwise around Q4_K_M with RAM offload, which is the usual
    quality/speed sweet spot. Full-precision files (F16/BF16/F32) are only picked as a last resort.
    """
    if not groups or not spec.ram:
        return None
    cands = [(i, g) for i, g in enumerate(groups) if not _FULL_RE.match(g.quant)] or list(enumerate(groups))
    fits = {i: judge(g.size + extra, spec).level for i, g in cands}

    gpu = [(i, g) for i, g in cands if fits[i] == "gpu" and _bpw(g.size, g.quant, params) >= GOOD_BPW]
    if gpu:
        # Bigger than Q8 rarely helps; Q8_0 is already near-lossless.
        return max(gpu, key=lambda c: c[1].size)[0]

    runnable = [(i, g) for i, g in cands if fits[i] in ("gpu", "split", "cpu")]
    if not runnable:
        tight = [(i, g) for i, g in cands if fits[i] == "tight"]
        return min(tight, key=lambda c: c[1].size)[0] if tight else None
    q4 = next((g.size for _, g in cands if g.quant.endswith("Q4_K_M")), None)
    if q4 is None:
        q4 = next((g.size for _, g in cands if 4.4 <= _bpw(g.size, g.quant, params) <= 5.2), None)
    if q4 is not None:
        upto = [(i, g) for i, g in runnable if g.size <= q4 * 1.02]
        if upto:
            return max(upto, key=lambda c: c[1].size)[0]
    bpw = {i: _bpw(g.size, g.quant, params) for i, g in runnable}
    mid = [(i, g) for i, g in runnable if GOOD_BPW <= bpw[i] <= 5.2]
    if mid:
        return max(mid, key=lambda c: c[1].size)[0]
    good = [(i, g) for i, g in runnable if bpw[i] >= GOOD_BPW]
    if good:
        return min(good, key=lambda c: c[1].size)[0]
    return max(runnable, key=lambda c: c[1].size)[0]


# ───────── detection (cached, non-blocking) ─────────
_spec: Spec | None = None
_lock = threading.Lock()


def _nvml_vram() -> tuple[int, str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        total, names = 0, []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            total += int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
            name = pynvml.nvmlDeviceGetName(h)
            names.append(name.decode() if isinstance(name, bytes) else name)
        return total, " + ".join(names)
    except Exception:
        return 0, ""


def spec(binary: str | None) -> Spec:
    """Current best guess. Starts with RAM (+ NVML if available) and fills in GPUs from
    ``llama-server --list-devices`` in the background, which covers CUDA/ROCm/Vulkan/Metal."""
    global _spec
    with _lock:
        if _spec is not None:
            return _spec
        s = Spec(ram=psutil.virtual_memory().total, unified=sys.platform == "darwin")
        s.vram, s.gpus = _nvml_vram()
        if s.vram:
            s.source = "nvml"
        _spec = s

    def work() -> None:
        hw = tuning.detect(binary) if binary else None
        if hw is not None and hw.devices:
            s.vram = sum(d.total for d in hw.devices)
            s.gpus = " + ".join(d.name for d in hw.devices)
            s.source = "llama-server"
        elif not s.source:
            s.source = "none"

    threading.Thread(target=work, daemon=True).start()
    return s
