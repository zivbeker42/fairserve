"""Top-level simulator wiring fairness schedulers to the vLLM engine."""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional

from models import Interaction, Request
from oit import OIT
from scheduler import BaseScheduler
from vllm_engine import EngineEventType, VLLMEngineSimulator


class Simulator:
    """Orchestrates arrivals, fairness schedulers, and the vLLM engine."""

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
        self.current_tick = 0
        self.waiting: Deque[Request] = deque()
        self.interactions: Dict[int, Interaction] = {}
        self.completed_requests: List[Request] = []
        self.throttled_requests: List[Request] = []
        self.wasted_tokens = 0
        self.metrics: Dict[str, float] = {}
        self.engine = VLLMEngineSimulator(max_kv_tokens=kv_capacity, max_num_batched_tokens=max_batch)
        self._accounted_prefill: set[int] = set()
        self._id_to_request: Dict[int, Request] = {}

    @property
    def time(self) -> int:
        return self.current_tick

    def submit_interaction(self, interaction: Interaction) -> None:
        self.interactions[interaction.interaction_id] = interaction
        req = interaction.next_request()
        if req:
            self._accept_request(req)

    def _accept_request(self, req: Request) -> None:
        self.scheduler.on_request_arrival(req)
        self.waiting.append(req)
        self._id_to_request[req.request_id] = req

    def inject_requests(self, new_requests: List[Request]) -> None:
        for req in new_requests:
            snapshot = self.engine.get_state_snapshot()
            if self.oit:
                if self.oit.should_throttle(req, snapshot.kv_tokens_used, snapshot.num_active_decodes):
                    self.oit.throttle(req)
                    self.throttled_requests.append(req)
                    continue
                self.oit.record_arrival(req)
            self._accept_request(req)

    def _admit_to_engine(self) -> None:
        snapshot = self.engine.get_state_snapshot()
        selected = self.scheduler.select_next_requests(
            self.waiting,
            self.interactions,
            snapshot,
            max_to_release=self.max_batch,
        )
        for req in selected:
            self.engine.submit_request(req)

    def _process_events(self, events) -> None:
        decode_served: List[Request] = []
        for ev in events:
            req = self._id_to_request.get(ev.request_id)
            if ev.event_type == EngineEventType.PREFILL_CHUNK_STARTED and ev.chunk_id == 0 and req and req.request_id not in self._accounted_prefill:
                self.scheduler.on_prefill_added(req)
                self._accounted_prefill.add(req.request_id)
            if ev.event_type == EngineEventType.DECODE_STEP and req:
                decode_served.append(req)
            if ev.event_type == EngineEventType.REQUEST_COMPLETED and req:
                req.completion_time = ev.time
                self.completed_requests.append(req)
                inter = self.interactions.get(req.interaction_id)
                if inter:
                    inter.mark_stage_complete()
                    nxt = inter.next_request()
                    if nxt:
                        nxt.arrival_time = int(ev.time)
                        self._accept_request(nxt)
        if decode_served:
            self.scheduler.on_decode_iteration(decode_served)

    def step(self) -> None:
        self._admit_to_engine()
        events = self.engine.step()
        if events:
            self._process_events(events)
        self.current_tick += 1

    def run(self) -> None:
        while self.current_tick < self.max_time and (self.waiting or self.engine.has_pending_work() or any(not i.complete for i in self.interactions.values())):
            self.step()
        for req in self.waiting:
            self.wasted_tokens += req.input_tokens + req.system_tokens + req.remaining_decode
        self.metrics = self._gather_metrics()

    def _gather_metrics(self) -> Dict[str, float]:
        per_user_tokens: Dict[str, float] = defaultdict(float)
        for req in self.completed_requests:
            per_user_tokens[req.user.user_id] += req.input_tokens + req.system_tokens + req.output_tokens_target
        avg_latency = sum(r.completion_time - r.arrival_time for r in self.completed_requests if r.completion_time is not None) / max(1, len(self.completed_requests))
        return {
            "completed": len(self.completed_requests),
            "avg_latency": avg_latency,
            "wasted_tokens": self.wasted_tokens,
            "throttled": len(self.throttled_requests),
            "per_user_tokens": per_user_tokens,
        }
