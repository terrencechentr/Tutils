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
    纯手动分段计时：
      - 你负责在需要的位置调用 start(name) / end(name)
      - name 仅作为标记；即便重复使用同一个 name，也被视为新的独立记录
      - 不做“自动进入下一段”、不做“同名聚合/计数”

    适合：你已有清晰的开始/结束钩子（hook），只需要记录每段耗时。
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
            pass  # 不因 CUDA 问题报错

    # ---- API ----
    def start(self, name: str) -> None:
        """
        开始一段名为 `name` 的计时。要求当前没有在计时。
        注意：此 name 不与任何历史记录绑定，后续即使再用同名，也是新的独立段。
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
        结束当前段并返回该段秒数。
        若提供 name，则会校验是否与 start 时指定的 name 相同（帮助你捕获钩子写错的情况）。
        """
        if not self._running:
            raise RuntimeError("SegmentTimer.end(): no segment is running. Did you call start()?")
        if name is not None and name != self._curr_name:
            raise ValueError(f"SegmentTimer.end(): name mismatch. started='{self._curr_name}', ended='{name}'")
        self._sync_cuda()
        t1 = time.perf_counter_ns()
        span_ns = t1 - (self._t0_ns or t1)
        span_s = span_ns / 1e9

        # 累计总时长、保存记录（同名也只是追加一条新记录，不做聚合）
        self._elapsed_ns_total += span_ns
        self._records.append({"name": self._curr_name, "seconds": span_s})

        # 清理运行状态
        self._running = False
        self._curr_name = None
        self._t0_ns = None
        return span_s

    def reset(self, *, clear_records: bool = True) -> None:
        """重置计时器状态；可选是否清空历史记录。"""
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
        所有已完成段的总时长（不包含当前未结束段）。
        设计上你是显式 end，所以这里通常就是完整总和。
        """
        return self._elapsed_ns_total / 1e9

    @property
    def records(self) -> List[Dict[str, Any]]:
        """
        返回所有段的列表（按时间顺序）。同名也会出现多条。
        示例： [{"name":"load","seconds":0.123}, {"name":"train","seconds":2.5}, {"name":"load","seconds":0.110}]
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
