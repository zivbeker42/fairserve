"""Synthetic workload generator for FAIRSERVE simulations."""
from __future__ import annotations
import random
from typing import Dict, List

from models import Application, Interaction, InteractionStage, Request, User


class WorkloadGenerator:
    def __init__(self, users: List[User], apps: List[Application], seed: int = 0):
        self.users = users
        self.apps = apps
        self.rng = random.Random(seed)
        self.request_id = 0
        self.interaction_id = 0

    def _sample_tokens(self, expected: int) -> int:
        low = max(1, int(0.7 * expected))
        high = int(1.3 * expected)
        return self.rng.randint(low, max(low + 1, high))

    def generate_interaction(self, user: User, app: Application, stages: List[InteractionStage]) -> Interaction:
        requests: List[Request] = []
        for stage in stages:
            idx = stage.value
            inp = self._sample_tokens(app.expected_input_tokens.get(idx, 1))
            sys = self._sample_tokens(app.expected_system_tokens.get(idx, 0))
            out = self._sample_tokens(app.expected_output_tokens.get(idx, 1))
            req = Request(
                request_id=self.request_id,
                user=user,
                application=app,
                interaction_id=self.interaction_id,
                stage=stage,
                input_tokens=inp,
                system_tokens=sys,
                output_tokens_target=out,
                arrival_time=0,
            )
            self.request_id += 1
            requests.append(req)
        interaction = Interaction(interaction_id=self.interaction_id, requests=requests)
        self.interaction_id += 1
        return interaction

    def poisson_arrivals(self, rate: float, duration: int) -> List[int]:
        times = []
        t = 0
        while t < duration:
            gap = self.rng.expovariate(rate)
            t += int(max(1, gap))
            if t < duration:
                times.append(t)
        return times

    def build_trace(self, duration: int, abusive_users: List[str]) -> Dict[int, List[Interaction]]:
        trace: Dict[int, List[Interaction]] = {}
        for idx, user in enumerate(self.users):
            app = self.apps[idx % len(self.apps)]
            rate = 0.05
            if user.user_id in abusive_users:
                rate = 0.3
            arrivals = self.poisson_arrivals(rate, duration)
            for ts in arrivals:
                stages = [InteractionStage.USER_PROMPT, InteractionStage.FINAL]
                if app.app_id == "multiagent":
                    stages = [InteractionStage.USER_PROMPT, InteractionStage.AGENT_1, InteractionStage.AGENT_2, InteractionStage.FINAL]
                inter = self.generate_interaction(user, app, stages)
                for req in inter.requests:
                    req.arrival_time = ts
                trace.setdefault(ts, []).append(inter)
        return trace
