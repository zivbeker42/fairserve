"""Core data models for FAIRSERVE simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class InteractionStage(Enum):
    USER_PROMPT = 0
    AGENT_1 = 1
    AGENT_2 = 2
    FINAL = 3


@dataclass
class User:
    user_id: str
    priority: float = 1.0


@dataclass
class Application:
    app_id: str
    expected_input_tokens: Dict[int, int]
    expected_system_tokens: Dict[int, int]
    expected_output_tokens: Dict[int, int]
    user_rpm_limit: int = 120
    app_rpm_limit: int = 2000

    def stage_weight(self, stage: InteractionStage, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.0) -> float:
        idx = stage.value
        ni = self.expected_input_tokens.get(idx, 1)
        ns = self.expected_system_tokens.get(idx, 0)
        no = self.expected_output_tokens.get(idx, 1)
        return alpha * ni + beta * ns + gamma * no


@dataclass
class Request:
    request_id: int
    user: User
    application: Application
    interaction_id: int
    stage: InteractionStage
    input_tokens: int
    system_tokens: int
    output_tokens_target: int
    arrival_time: int
    remaining_decode: int = field(init=False)
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    throttled: bool = False
    stalled: bool = False

    def __post_init__(self) -> None:
        self.remaining_decode = self.output_tokens_target

    @property
    def done(self) -> bool:
        return self.remaining_decode <= 0

    @property
    def latency(self) -> Optional[int]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


@dataclass
class Interaction:
    interaction_id: int
    requests: List[Request]
    next_index: int = 0
    complete: bool = False

    def next_request(self) -> Optional[Request]:
        if self.complete or self.next_index >= len(self.requests):
            self.complete = True
            return None
        req = self.requests[self.next_index]
        self.next_index += 1
        return req

    def mark_stage_complete(self) -> None:
        if self.next_index >= len(self.requests):
            self.complete = True


@dataclass
class RunningBatch:
    requests: List[Request] = field(default_factory=list)
    kv_tokens: int = 0

    def add(self, request: Request) -> None:
        self.requests.append(request)
        self.kv_tokens += request.input_tokens + request.system_tokens

    def remove(self, request: Request) -> None:
        if request in self.requests:
            self.requests.remove(request)
            self.kv_tokens -= (request.input_tokens + request.system_tokens + (request.output_tokens_target - request.remaining_decode))
            if self.kv_tokens < 0:
                self.kv_tokens = 0
