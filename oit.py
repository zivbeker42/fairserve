"""Overload & Interaction-driven throttling."""
from __future__ import annotations
from collections import deque, defaultdict
from typing import Deque, Dict

from models import Request, InteractionStage


class OIT:
    """Tracks per-user/app request rates and applies interaction-aware throttling."""

    def __init__(self, window: int = 60, kv_threshold: int = 5000, max_batch: int = 32):
        self.window = window
        self.kv_threshold = kv_threshold
        self.max_batch = max_batch
        self.user_windows: Dict[str, Deque[int]] = defaultdict(deque)
        self.app_windows: Dict[str, Deque[int]] = defaultdict(deque)
        self.throttled = 0

    def _evict(self, dq: Deque[int], now: int) -> None:
        while dq and dq[0] <= now - self.window:
            dq.popleft()

    def record_arrival(self, req: Request) -> None:
        dq_u = self.user_windows[req.user.user_id]
        dq_a = self.app_windows[req.application.app_id]
        dq_u.append(req.arrival_time)
        dq_a.append(req.arrival_time)

    def is_overloaded(self, kv_usage: int, running: int) -> bool:
        return kv_usage > self.kv_threshold or running >= self.max_batch

    def should_throttle(self, req: Request, kv_usage: int, running: int) -> bool:
        dq_u = self.user_windows[req.user.user_id]
        dq_a = self.app_windows[req.application.app_id]
        self._evict(dq_u, req.arrival_time)
        self._evict(dq_a, req.arrival_time)
        if not self.is_overloaded(kv_usage, running):
            return False
        if req.stage != InteractionStage.USER_PROMPT:
            return False
        # New interaction: apply RPM limits
        if len(dq_u) >= req.application.user_rpm_limit:
            return True
        if len(dq_a) >= req.application.app_rpm_limit:
            return True
        return False

    def throttle(self, req: Request) -> None:
        req.throttled = True
        self.throttled += 1
