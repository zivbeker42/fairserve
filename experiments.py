"""Experiment driver comparing FCFS, VTC, and FAIRSERVE."""
from __future__ import annotations
from typing import Dict, List

from models import Application, User
from scheduler import FCFSScheduler, VTCScheduler, FairServeScheduler
from simulator import Simulator
from oit import OIT
from workload import WorkloadGenerator


def default_apps() -> List[Application]:
    return [
        Application(
            app_id="summarization",
            expected_input_tokens={0: 500, 1: 100, 2: 50, 3: 20},
            expected_system_tokens={0: 10, 1: 10, 2: 10, 3: 5},
            expected_output_tokens={0: 80, 1: 40, 2: 40, 3: 20},
        ),
        Application(
            app_id="chat",
            expected_input_tokens={0: 200, 1: 150, 2: 100, 3: 50},
            expected_system_tokens={0: 20, 1: 10, 2: 10, 3: 5},
            expected_output_tokens={0: 150, 1: 100, 2: 80, 3: 50},
        ),
        Application(
            app_id="coding",
            expected_input_tokens={0: 120, 1: 80, 2: 80, 3: 20},
            expected_system_tokens={0: 30, 1: 10, 2: 10, 3: 5},
            expected_output_tokens={0: 300, 1: 120, 2: 120, 3: 60},
        ),
        Application(
            app_id="multiagent",
            expected_input_tokens={0: 80, 1: 60, 2: 60, 3: 40},
            expected_system_tokens={0: 10, 1: 20, 2: 20, 3: 10},
            expected_output_tokens={0: 100, 1: 80, 2: 80, 3: 60},
        ),
    ]


def run_experiment(duration: int = 400) -> Dict[str, dict]:
    users: List[User] = [User(user_id=f"user{i}") for i in range(10)]
    abusive = ["user0"]
    apps = default_apps()
    generator = WorkloadGenerator(users, apps, seed=1234)
    trace = generator.build_trace(duration=duration, abusive_users=abusive)

    results: Dict[str, dict] = {}
    schedulers = {
        "fcfs": FCFSScheduler(),
        "vtc": VTCScheduler(),
        "fairserve": FairServeScheduler(),
    }
    for name, sched in schedulers.items():
        oit = OIT(kv_threshold=15000, max_batch=16) if name == "fairserve" else None
        sim = Simulator(scheduler=sched, oit=oit, kv_capacity=18000, max_batch=16, max_time=duration)
        for ts in sorted(trace.keys()):
            for inter in trace[ts]:
                # shift time to arrival
                for req in inter.requests:
                    req.arrival_time = ts
                sim.submit_interaction(inter)
            while sim.time < ts:
                sim.step()
                sim.time += 1
        sim.run()
        results[name] = sim.metrics
    return results


if __name__ == "__main__":
    metrics = run_experiment(duration=300)
    for name, data in metrics.items():
        print(f"=== {name.upper()} ===")
        print(data)
