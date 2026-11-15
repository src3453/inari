"""Host hardware sampling (CPU / RAM / disk / server process / optional NVIDIA GPU)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil


@dataclass
class GpuStat:
    name: str
    util: float | None = None
    mem_used: float | None = None  # bytes
    mem_total: float | None = None
    temp: float | None = None
    power: float | None = None  # W
    power_limit: float | None = None
    pcie_rx: float | None = None  # bytes/s (host -> GPU)
    pcie_tx: float | None = None


@dataclass
class HwStat:
    cpu: float = 0.0
    per_cpu: list[float] = field(default_factory=list)
    cores: int = 0
    threads: int = 0
    ram_used: float = 0.0
    ram_total: float = 0.0
    swap_used: float = 0.0
    swap_total: float = 0.0
    disk_read: float = 0.0  # bytes/s
    disk_write: float = 0.0
    proc_rss: float | None = None
    proc_cpu: float | None = None
    proc_threads: int | None = None
    gpus: list[GpuStat] = field(default_factory=list)


class _Nvml:
    def __init__(self) -> None:
        self.ok = False
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self.ok = bool(self.handles)
        except Exception:
            self.ok = False

    def sample(self) -> list[GpuStat]:
        if not self.ok:
            return []
        nv = self.nv
        out = []
        for h in self.handles:
            def safe(fn, *a):
                try:
                    return fn(*a)
                except Exception:
                    return None

            name = safe(nv.nvmlDeviceGetName, h) or "GPU"
            if isinstance(name, bytes):
                name = name.decode()
            g = GpuStat(name=name)
            u = safe(nv.nvmlDeviceGetUtilizationRates, h)
            g.util = float(u.gpu) if u else None
            m = safe(nv.nvmlDeviceGetMemoryInfo, h)
            if m:
                g.mem_used, g.mem_total = float(m.used), float(m.total)
            t = safe(nv.nvmlDeviceGetTemperature, h, 0)
            g.temp = float(t) if t is not None else None
            p = safe(nv.nvmlDeviceGetPowerUsage, h)
            g.power = p / 1000.0 if p is not None else None
            pl = safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            g.power_limit = pl / 1000.0 if pl is not None else None
            rx = safe(nv.nvmlDeviceGetPcieThroughput, h, getattr(nv, "NVML_PCIE_UTIL_RX_BYTES", 1))
            tx = safe(nv.nvmlDeviceGetPcieThroughput, h, getattr(nv, "NVML_PCIE_UTIL_TX_BYTES", 0))
            g.pcie_rx = rx * 1024.0 if rx is not None else None
            g.pcie_tx = tx * 1024.0 if tx is not None else None
            out.append(g)
        return out


class HwSampler:
    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid
        self._proc: psutil.Process | None = None
        if pid:
            try:
                self._proc = psutil.Process(pid)
                self._proc.cpu_percent(None)
            except psutil.Error:
                self._proc = None
        psutil.cpu_percent(None, percpu=True)
        self._disk = psutil.disk_io_counters()
        self._disk_t = time.monotonic()
        self._nvml = _Nvml()
        self.cores = psutil.cpu_count(logical=False) or 0
        self.threads = psutil.cpu_count(logical=True) or 0

    def sample(self) -> HwStat:
        s = HwStat(cores=self.cores, threads=self.threads)
        s.per_cpu = psutil.cpu_percent(None, percpu=True)
        s.cpu = sum(s.per_cpu) / len(s.per_cpu) if s.per_cpu else 0.0
        vm = psutil.virtual_memory()
        s.ram_used, s.ram_total = float(vm.total - vm.available), float(vm.total)
        sw = psutil.swap_memory()
        s.swap_used, s.swap_total = float(sw.used), float(sw.total)

        now = time.monotonic()
        d = psutil.disk_io_counters()
        if d and self._disk:
            dt = max(now - self._disk_t, 1e-3)
            s.disk_read = (d.read_bytes - self._disk.read_bytes) / dt
            s.disk_write = (d.write_bytes - self._disk.write_bytes) / dt
        self._disk, self._disk_t = d, now

        if self._proc is not None:
            try:
                with self._proc.oneshot():
                    s.proc_rss = float(self._proc.memory_info().rss)
                    s.proc_cpu = self._proc.cpu_percent(None) / max(self.threads, 1)
                    s.proc_threads = self._proc.num_threads()
            except psutil.Error:
                self._proc = None
        s.gpus = self._nvml.sample()
        return s
