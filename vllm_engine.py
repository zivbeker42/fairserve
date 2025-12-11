"""vLLM-style engine simulator with continuous batching and chunked prefill.

This module mirrors the two-layer architecture used by vLLM-like systems,
separating the outer fairness schedulers from the inner GPU execution loop.
Comments document inspiration from public sources (e.g., vLLM docs/blogs and
open-source simulators such as the FlexGen/vLLM scheduler discussions) without
copying code.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Optional

from models import Request


class EngineEventType(Enum):
    PREFILL_CHUNK_STARTED = auto()
    PREFILL_CHUNK_FINISHED = auto()
    DECODE_STEP = auto()
    REQUEST_COMPLETED = auto()


@dataclass
class EngineEvent:
    event_type: EngineEventType
    time: float
    request_id: int
    chunk_id: Optional[int] = None
    chunk_len: Optional[int] = None
    token_index: Optional[int] = None
    tokens_generated_this_step: Optional[int] = None


@dataclass
class EngineStateSnapshot:
    time: float
    num_active_decodes: int
    has_active_prefill: bool
    kv_tokens_used: int
    kv_tokens_capacity: int
    num_pending_prefills: int
    num_completed_requests: int


class VLLMEngineSimulator:
    """Simulates vLLM's continuous batching and chunked prefill engine.

    Inspired by public vLLM docs/blog posts that describe decode-maximal
    scheduling with chunked prefill interleaving and paged KV management. The
    engine exposes a submit/step/event API similar to lightweight simulators
    used in academic projects without exposing internal batch state to the
    fairness schedulers.
    """

    def __init__(
        self,
        max_kv_tokens: int = 20000,
        max_num_batched_tokens: int = 16,
        chunk_size: int = 256,
        a_prefill: float = 0.0001,
        b_prefill: float = 0.01,
        c_prefill: float = 0.1,
        a_decode: float = 0.00005,
        b_decode: float = 0.05,
    ):
        self.max_kv_tokens = max_kv_tokens
        self.max_num_batched_tokens = max_num_batched_tokens
        self.chunk_size = chunk_size
        self.a_prefill = a_prefill
        self.b_prefill = b_prefill
        self.c_prefill = c_prefill
        self.a_decode = a_decode
        self.b_decode = b_decode
        self.time: float = 0.0
        self.pending_prefill: Deque[Request] = deque()
        self.active_prefill: Optional[dict] = None
        self.active_decodes: List[Request] = []
        self.completed: List[Request] = []
        self.kv_tokens: int = 0
        self.prefill_started: set[int] = set()

    def submit_request(self, request: Request) -> None:
        """Queue a request for prefill; outer schedulers cannot peek inside."""
        self.pending_prefill.append(request)

    def has_pending_work(self) -> bool:
        return bool(self.pending_prefill or self.active_prefill or self.active_decodes)

    def _prefill_time(self, chunk_len: int) -> float:
        return self.a_prefill * (chunk_len ** 2) + self.b_prefill * chunk_len + self.c_prefill

    def _decode_time(self, batch_tokens: int) -> float:
        return self.a_decode * self.kv_tokens * batch_tokens + self.b_decode

    def _maybe_start_prefill(self, token_budget: int) -> Optional[tuple[int, List[EngineEvent], float]]:
        events: List[EngineEvent] = []
        if self.active_prefill is None and self.pending_prefill:
            candidate = self.pending_prefill[0]
            remaining = candidate.input_tokens + candidate.system_tokens
            chunk_len = min(self.chunk_size, remaining, token_budget)
            if chunk_len <= 0:
                return None
            if self.kv_tokens + remaining > self.max_kv_tokens:
                return None
            self.pending_prefill.popleft()
            self.active_prefill = {
                "request": candidate,
                "remaining": remaining,
                "chunk_id": 0,
            }
        if self.active_prefill is None:
            return None
        chunk_len = min(self.chunk_size, self.active_prefill["remaining"], token_budget)
        if chunk_len <= 0:
            return None
        req = self.active_prefill["request"]
        chunk_id = self.active_prefill["chunk_id"]
        start_event = EngineEvent(
            event_type=EngineEventType.PREFILL_CHUNK_STARTED,
            time=self.time,
            request_id=req.request_id,
            chunk_id=chunk_id,
            chunk_len=chunk_len,
        )
        events.append(start_event)
        self.active_prefill["remaining"] -= chunk_len
        self.active_prefill["chunk_id"] += 1
        finish_time = self._prefill_time(chunk_len)
        finish_event = EngineEvent(
            event_type=EngineEventType.PREFILL_CHUNK_FINISHED,
            time=self.time + finish_time,
            request_id=req.request_id,
            chunk_id=chunk_id,
            chunk_len=chunk_len,
        )
        events.append(finish_event)
        self.kv_tokens += chunk_len
        if req.start_time is None:
            req.start_time = self.time
        if self.active_prefill["remaining"] <= 0:
            self.active_prefill = None
            self.active_decodes.append(req)
        return chunk_len, events, finish_time

    def step(self) -> List[EngineEvent]:
        events: List[EngineEvent] = []
        token_budget = self.max_num_batched_tokens
        decode_candidates = [r for r in self.active_decodes if r.remaining_decode > 0]
        decode_served: List[Request] = []
        batch_tokens = 0
        if decode_candidates and token_budget > 0:
            take = min(len(decode_candidates), token_budget)
            selected = decode_candidates[:take]
            batch_tokens = len(selected)
            for req in selected:
                req.remaining_decode -= 1
                decode_served.append(req)
                events.append(
                    EngineEvent(
                        event_type=EngineEventType.DECODE_STEP,
                        time=self.time,
                        request_id=req.request_id,
                        token_index=req.output_tokens_target - req.remaining_decode,
                        tokens_generated_this_step=1,
                    )
                )
                self.kv_tokens += 1
            token_budget -= batch_tokens
        decode_time_cost = self._decode_time(batch_tokens) if batch_tokens else 0.0

        prefill_cost = 0.0
        prefill_result = None
        if token_budget > 0:
            prefill_result = self._maybe_start_prefill(token_budget)
        if prefill_result:
            chunk_len, prefill_events, prefill_cost = prefill_result
            token_budget -= chunk_len
            events.extend(prefill_events)

        completion_events: List[EngineEvent] = []
        for req in list(self.active_decodes):
            if req.remaining_decode <= 0:
                self.active_decodes.remove(req)
                completion_events.append(
                    EngineEvent(
                        event_type=EngineEventType.REQUEST_COMPLETED,
                        time=self.time + decode_time_cost + prefill_cost,
                        request_id=req.request_id,
                    )
                )
                self.completed.append(req)
        events.extend(completion_events)

        time_advance = decode_time_cost + prefill_cost
        if time_advance == 0 and not events:
            return []
        self.time += max(time_advance, 0.0001)
        return events

    def get_state_snapshot(self) -> EngineStateSnapshot:
        return EngineStateSnapshot(
            time=self.time,
            num_active_decodes=len([r for r in self.active_decodes if r.remaining_decode > 0]),
            has_active_prefill=self.active_prefill is not None,
            kv_tokens_used=self.kv_tokens,
            kv_tokens_capacity=self.max_kv_tokens,
            num_pending_prefills=len(self.pending_prefill),
            num_completed_requests=len(self.completed),
        )
