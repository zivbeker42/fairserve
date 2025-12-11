"""Schedulers for simulators: FCFS, VTC, and FAIRSERVE WSC."""
from __future__ import annotations
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional

from models import Interaction, Request, InteractionStage


class BaseScheduler:
    def on_request_arrival(self, request: Request) -> None:
        pass

    def on_prefill_added(self, request: Request) -> None:
        pass

    def on_decode_iteration(self, served: List[Request]) -> None:
        pass

    def select_next_requests(
        self,
        waiting: Deque[Request],
        interactions: Dict[int, Interaction],
        kv_tokens: int,
        kv_capacity: int,
        max_batch: int,
    ) -> List[Request]:
        raise NotImplementedError


class FCFSScheduler(BaseScheduler):
    """Simple first-come-first-serve scheduler."""

    def select_next_requests(self, waiting: Deque[Request], interactions: Dict[int, Interaction], kv_tokens: int, kv_capacity: int, max_batch: int) -> List[Request]:
        selected: List[Request] = []
        while waiting and len(selected) < max_batch and kv_tokens < kv_capacity:
            nxt = waiting[0]
            if kv_tokens + nxt.input_tokens + nxt.system_tokens <= kv_capacity:
                waiting.popleft()
                selected.append(nxt)
                kv_tokens += nxt.input_tokens + nxt.system_tokens
            else:
                break
        return selected


class VTCScheduler(BaseScheduler):
    """Virtual Token Counter fairness scheduler."""

    def __init__(self, w_p: float = 1.0, w_q: float = 1.0, counter_lift: bool = True):
        self.w_p = w_p
        self.w_q = w_q
        self.counter_lift = counter_lift
        self.counters: Dict[str, float] = defaultdict(float)

    def on_request_arrival(self, request: Request) -> None:
        if self.counter_lift:
            active = list(self.counters.values())
            if active:
                minimum = min(active)
                self.counters[request.user.user_id] = max(self.counters[request.user.user_id], minimum)

    def on_prefill_added(self, request: Request) -> None:
        self.counters[request.user.user_id] += self.w_p * (request.input_tokens + request.system_tokens)

    def on_decode_iteration(self, served: List[Request]) -> None:
        for req in served:
            self.counters[req.user.user_id] += self.w_q

    def select_next_requests(self, waiting: Deque[Request], interactions: Dict[int, Interaction], kv_tokens: int, kv_capacity: int, max_batch: int) -> List[Request]:
        selected: List[Request] = []
        while waiting and len(selected) < max_batch:
            # pick user with smallest counter
            waiting_by_user: Dict[str, Request] = {}
            for req in waiting:
                if req.user.user_id not in waiting_by_user:
                    waiting_by_user[req.user.user_id] = req
            if not waiting_by_user:
                break
            user_choice = min(waiting_by_user.keys(), key=lambda u: self.counters[u])
            candidate = waiting_by_user[user_choice]
            # ensure capacity
            if kv_tokens + candidate.input_tokens + candidate.system_tokens > kv_capacity:
                break
            waiting.remove(candidate)
            selected.append(candidate)
            kv_tokens += candidate.input_tokens + candidate.system_tokens
        return selected


class FairServeScheduler(BaseScheduler):
    """Weighted Service Counter scheduler implementing WSC."""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.0, counter_lift: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.counter_lift = counter_lift
        self.service: Dict[str, float] = defaultdict(float)

    def _weight(self, req: Request) -> float:
        return req.application.stage_weight(req.stage, self.alpha, self.beta, self.gamma)

    def on_request_arrival(self, request: Request) -> None:
        if self.counter_lift:
            if self.service:
                minimum = min(self.service.values())
                self.service[request.user.user_id] = max(self.service[request.user.user_id], minimum)

    def on_prefill_added(self, request: Request) -> None:
        tokens = self.alpha * request.input_tokens + self.beta * request.system_tokens
        inc = request.user.priority * tokens / self._weight(request)
        self.service[request.user.user_id] += inc

    def on_decode_iteration(self, served: List[Request]) -> None:
        for req in served:
            inc = req.user.priority * self.gamma / self._weight(req)
            self.service[req.user.user_id] += inc

    def select_next_requests(self, waiting: Deque[Request], interactions: Dict[int, Interaction], kv_tokens: int, kv_capacity: int, max_batch: int) -> List[Request]:
        selected: List[Request] = []
        # Prioritize incomplete interactions
        ready_interactions = [inter for inter in interactions.values() if not inter.complete and inter.next_index < len(inter.requests)]
        ready_users = {inter.requests[inter.next_index].user.user_id: inter for inter in ready_interactions if inter.next_index < len(inter.requests)}
        while len(selected) < max_batch and kv_tokens < kv_capacity:
            candidate: Optional[Request] = None
            # Interaction continuation first
            if ready_users:
                uid = min(ready_users.keys(), key=lambda u: self.service[u])
                inter = ready_users.pop(uid)
                req = inter.requests[inter.next_index]
                if kv_tokens + req.input_tokens + req.system_tokens <= kv_capacity:
                    candidate = req
                    inter.next_index += 1
                else:
                    break
            else:
                if not waiting:
                    break
                waiting_by_user: Dict[str, Request] = {}
                for req in waiting:
                    if req.user.user_id not in waiting_by_user:
                        waiting_by_user[req.user.user_id] = req
                uid = min(waiting_by_user.keys(), key=lambda u: self.service[u])
                candidate = waiting_by_user[uid]
                if kv_tokens + candidate.input_tokens + candidate.system_tokens > kv_capacity:
                    break
                waiting.remove(candidate)
            if candidate is None:
                break
            selected.append(candidate)
            kv_tokens += candidate.input_tokens + candidate.system_tokens
        return selected
