"""Leaving the computer usable while a song is processed.

Left alone, the models take every CPU core and the whole GPU for minutes,
and the desktop (which draws with the same GPU) stutters or freezes. So by
default TabGenerator

- runs at low priority (nice 10): anything else wins when it wants a core;
- uses at most half the CPU cores for number crunching;
- pauses the GPU work for a few milliseconds many times a second, so the
  desktop and other programs get their turn (costs about a fifth in speed).

`--full-speed` (or TABGEN_FULL_SPEED=1) turns all of it off.
"""

import os
import time

GPU_SHARE = 0.8  # share of the time the GPU may be kept busy
SLICE_S = 0.04  # ... in slices of about this long

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def enabled() -> bool:
    return os.environ.get("TABGEN_FULL_SPEED", "") in ("", "0")


def cpu_threads() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def limit_cpu():
    """Call before numpy and torch are imported (their thread pools read the
    environment once); safe to call again later."""
    if not enabled():
        return
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(cpu_threads()))


_applied = False


def apply():
    """Lower this process's priority and cap torch's CPU threads (once)."""
    global _applied
    if _applied or not enabled():
        return
    _applied = True
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except (AttributeError, OSError):
        pass
    try:
        import torch

        torch.set_num_threads(min(torch.get_num_threads(), cpu_threads()))
    except Exception:
        pass


class Breather:
    """Call between small pieces of GPU work: once about SLICE_S of work has
    piled up it waits for the GPU to finish and sleeps long enough that the
    GPU is busy for only GPU_SHARE of the time."""

    def __init__(self, device: str = "cuda"):
        self.on = enabled() and str(device).startswith("cuda")
        self.since = time.perf_counter()

    def __call__(self):
        if not self.on:
            return
        busy = time.perf_counter() - self.since
        if busy < SLICE_S:
            return
        import torch

        torch.cuda.synchronize()
        busy = time.perf_counter() - self.since
        # Long pieces (a whole separation chunk) get their pause in one go,
        # but never a noticeable stall.
        time.sleep(min(busy * (1 - GPU_SHARE) / GPU_SHARE, 0.25))
        self.since = time.perf_counter()
