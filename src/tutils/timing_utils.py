# utilitybox/timing.py
from __future__ import annotations
import time
from typing import Optional, List, Dict, Any, Tuple

def _fmt(sec: float) -> str:
    if sec < 1e-6: return f"{sec*1e9:.1f} ns"
    if sec < 1e-3: return f"{sec*1e6:.1f} µs"
    if sec < 1:   return f"{sec*1e3:.2f} ms"
    return f"{sec:.3f} s"

class Timer:
    """
    Manual, segment-based timer:
      - You call start(name) / end(name) explicitly.
      - name is just a tag; reusing the same name creates a new record.
      - No auto-chaining or aggregation.

    Best for cases where you already have clear start/end hooks and only need
    to measure each segment.
    """

    def __init__(
        self,
        *,
        cuda: bool = False,         # whether to do torch.cuda.synchronize() at the timing point
        logger: Optional[Any] = None, # optional logging.Logger; only used when you manually log
    ) -> None:
        self.cuda = cuda
        self.logger = logger
        self._running: bool = False
        self._curr_name: Optional[str] = None
        self._t0_ns: Optional[int] = None
        self._records: List[Dict[str, Any]] = []  # [{'name': str, 'seconds': float}]
        self._elapsed_ns_total: int = 0

    # ---- internals ----
    def _sync_cuda(self) -> None:
        if not self.cuda:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass  # do not fail due to CUDA issues

    # ---- API ----
    def start(self, name: str) -> None:
        """
        Start timing a segment named `name`. Requires no segment running.
        Reusing the same name later still creates an independent record.
        """
        if self._running:
            raise RuntimeError("SegmentTimer.start(): a segment is already running. Call end() first.")
        if not name:
            raise ValueError("SegmentTimer.start(): name must be a non-empty string.")
        self._sync_cuda()
        self._curr_name = name
        self._t0_ns = time.perf_counter_ns()
        self._running = True

    def end(self, name: Optional[str] = None) -> float:
        """
        Stop the current segment and return its duration in seconds.
        If `name` is provided, validate it matches the name used at start.
        """
        if not self._running:
            raise RuntimeError("SegmentTimer.end(): no segment is running. Did you call start()?")
        if name is not None and name != self._curr_name:
            raise ValueError(f"SegmentTimer.end(): name mismatch. started='{self._curr_name}', ended='{name}'")
        self._sync_cuda()
        t1 = time.perf_counter_ns()
        span_ns = t1 - (self._t0_ns or t1)
        span_s = span_ns / 1e9

        # accumulate total time; same-name segments are just appended
        self._elapsed_ns_total += span_ns
        self._records.append({"name": self._curr_name, "seconds": span_s})

        # clear running state
        self._running = False
        self._curr_name = None
        self._t0_ns = None
        return span_s

    def reset(self, *, clear_records: bool = True) -> None:
        """Reset timer state; optionally clear history."""
        self._running = False
        self._curr_name = None
        self._t0_ns = None
        self._elapsed_ns_total = 0
        if clear_records:
            self._records.clear()

    # ---- properties & export ----
    @property
    def running(self) -> bool:
        return self._running

    @property
    def current_name(self) -> Optional[str]:
        return self._curr_name

    @property
    def total_seconds(self) -> float:
        """
        Total duration of all completed segments (excludes a running one).
        """
        return self._elapsed_ns_total / 1e9

    @property
    def records(self) -> List[Dict[str, Any]]:
        """
        Return all segment records in time order.
        Example: [{"name":"load","seconds":0.123}, {"name":"train","seconds":2.5}, {"name":"load","seconds":0.110}]
        """
        return list(self._records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "records": self.records,
        }

    def report(self) -> str:
        lines = [f"Total: {_fmt(self.total_seconds)}"]
        if self._records:
            lines.append("Segments:")
            for i, r in enumerate(self._records, 1):
                lines.append(f"  #{i:02d} {r['name']}: {_fmt(r['seconds'])}")
        if self._running and self._curr_name:
            lines.append(f"* still running: {self._curr_name}")
        return "\n".join(lines)
