"""Continuous batching simulator for FAIRSERVE and baselines."""
from __future__ import annotations
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple

from models import Interaction, Request, RunningBatch
from scheduler import BaseScheduler
from oit import OIT


class Simulator:
    def __init__(
        self,
        scheduler: BaseScheduler,
        oit: Optional[OIT],
        kv_capacity: int = 20000,
        max_batch: int = 16,
        max_time: int = 2000,
    ):
        self.scheduler = scheduler
        self.oit = oit
        self.kv_capacity = kv_capacity
        self.max_batch = max_batch
        self.max_time = max_time
        self.time = 0
        self.waiting: Deque[Request] = deque()
        self.running = RunningBatch()
        self.interactions: Dict[int, Interaction] = {}
        self.completed_requests: List[Request] = []
        self.throttled_requests: List[Request] = []
        self.wasted_tokens = 0
        self.metrics: Dict[str, float] = {}

    def submit_interaction(self, interaction: Interaction) -> None:
        self.interactions[interaction.interaction_id] = interaction
        req = interaction.next_request()
        if req:
            self._accept_request(req)

    def _accept_request(self, req: Request) -> None:
        self.scheduler.on_request_arrival(req)
        self.waiting.append(req)

    def _prefill(self, req: Request) -> None:
        req.start_time = self.time
        self.running.add(req)
        self.scheduler.on_prefill_added(req)

    def _decode_iteration(self) -> None:
        served: List[Request] = []
        for req in list(self.running.requests):
            if req.remaining_decode > 0:
                req.remaining_decode -= 1
                served.append(req)
                if req.done:
                    req.completion_time = self.time
                    self.running.remove(req)
                    self.completed_requests.append(req)
                    inter = self.interactions.get(req.interaction_id)
                    if inter:
                        inter.mark_stage_complete()
                        nxt = inter.next_request()
                        if nxt:
                            self._accept_request(nxt)
        self.scheduler.on_decode_iteration(served)

    def _try_schedule(self) -> None:
        new_reqs = self.scheduler.select_next_requests(self.waiting, self.interactions, self.running.kv_tokens, self.kv_capacity, self.max_batch - len(self.running.requests))
        for req in new_reqs:
            self._prefill(req)

    def step(self) -> None:
        self._try_schedule()
        self._decode_iteration()

    def run(self) -> None:
        while self.time < self.max_time and (self.waiting or self.running.requests or any(not i.complete for i in self.interactions.values())):
            self.step()
            self.time += 1
        # record wasted tokens
        for req in self.waiting:
            self.wasted_tokens += req.input_tokens + req.system_tokens + req.remaining_decode
        self.metrics = self._gather_metrics()

    def inject_requests(self, new_requests: List[Request]) -> None:
        for req in new_requests:
            if self.oit:
                if self.oit.should_throttle(req, self.running.kv_tokens, len(self.running.requests)):
                    self.oit.throttle(req)
                    self.throttled_requests.append(req)
                    continue
                self.oit.record_arrival(req)
            self._accept_request(req)

    def _gather_metrics(self) -> Dict[str, float]:
        per_user_tokens: Dict[str, float] = defaultdict(float)
        for req in self.completed_requests:
            per_user_tokens[req.user.user_id] += req.input_tokens + req.system_tokens + req.output_tokens_target
        avg_latency = sum(r.latency for r in self.completed_requests if r.latency is not None) / max(1, len(self.completed_requests))
        return {
            "completed": len(self.completed_requests),
            "avg_latency": avg_latency,
            "wasted_tokens": self.wasted_tokens,
            "throttled": len(self.throttled_requests),
            "per_user_tokens": per_user_tokens,
        }
