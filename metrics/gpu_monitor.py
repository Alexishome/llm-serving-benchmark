from __future__ import annotations

import asyncio
import csv
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GPUSample:
    timestamp: float
    gpu_index: int
    utilization_gpu: float
    utilization_memory: float
    memory_used_mb: float
    memory_total_mb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class GPUMonitor:
    """Collect point-in-time GPU metrics using NVML when available."""

    def __init__(self, poll_interval: float = 1.0) -> None:
        self.poll_interval = poll_interval
        self._samples: list[GPUSample] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._nvml = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    @property
    def samples(self) -> list[GPUSample]:
        return list(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    async def _run(self) -> None:
        while self._running:
            try:
                self._samples.extend(await asyncio.to_thread(self._collect_once))
            except Exception:
                pass
            await asyncio.sleep(self.poll_interval)

    def _collect_once(self) -> list[GPUSample]:
        try:
            return self._collect_with_nvml()
        except Exception:
            return self._collect_with_nvidia_smi()

    def _collect_with_nvml(self) -> list[GPUSample]:
        if self._nvml is None:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml

        pynvml = self._nvml
        samples: list[GPUSample] = []
        timestamp = time.time()
        device_count = pynvml.nvmlDeviceGetCount()

        for gpu_index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            samples.append(
                GPUSample(
                    timestamp=timestamp,
                    gpu_index=gpu_index,
                    utilization_gpu=float(util.gpu),
                    utilization_memory=float(util.memory),
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_total_mb=memory.total / (1024 * 1024),
                )
            )
        return samples

    def _collect_with_nvidia_smi(self) -> list[GPUSample]:
        query = (
            "index,utilization.gpu,utilization.memory,memory.used,memory.total"
        )
        command = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, text=True)
        timestamp = time.time()
        samples: list[GPUSample] = []

        for line in output.strip().splitlines():
            index, util_gpu, util_mem, mem_used, mem_total = [part.strip() for part in line.split(",")]
            samples.append(
                GPUSample(
                    timestamp=timestamp,
                    gpu_index=int(index),
                    utilization_gpu=float(util_gpu),
                    utilization_memory=float(util_mem),
                    memory_used_mb=float(mem_used),
                    memory_total_mb=float(mem_total),
                )
            )
        return samples

    def save_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(GPUSample.__dataclass_fields__.keys()))
            writer.writeheader()
            for sample in self._samples:
                writer.writerow(sample.to_dict())
