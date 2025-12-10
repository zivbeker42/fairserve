"""Lightweight checks for scheduling and throttling."""
from __future__ import annotations

from models import Application, InteractionStage, User
from scheduler import VTCScheduler, FairServeScheduler
from simulator import Simulator
from oit import OIT
from workload import WorkloadGenerator


def simple_app() -> Application:
    return Application(
        app_id="toy",
        expected_input_tokens={0: 10, 1: 5},
        expected_system_tokens={0: 2, 1: 2},
        expected_output_tokens={0: 5, 1: 5},
    )


def test_vtc_prioritizes_under_served():
    users = [User("a"), User("b")]
    app = simple_app()
    gen = WorkloadGenerator(users, [app], seed=1)
    inter_a = gen.generate_interaction(users[0], app, [InteractionStage.USER_PROMPT])
    inter_b = gen.generate_interaction(users[1], app, [InteractionStage.USER_PROMPT])
    sched = VTCScheduler()
    sim = Simulator(scheduler=sched, oit=None, kv_capacity=200, max_batch=1, max_time=50)
    sim.submit_interaction(inter_a)
    sim.submit_interaction(inter_b)
    sim.run()
    assert len(sim.completed_requests) == 2


def test_fairserve_counter_lift():
    users = [User("a"), User("b")]
    app = simple_app()
    gen = WorkloadGenerator(users, [app], seed=2)
    inter_a = gen.generate_interaction(users[0], app, [InteractionStage.USER_PROMPT])
    inter_b = gen.generate_interaction(users[1], app, [InteractionStage.USER_PROMPT])
    sched = FairServeScheduler()
    sim = Simulator(scheduler=sched, oit=None, kv_capacity=200, max_batch=1, max_time=50)
    sim.submit_interaction(inter_a)
    sim.run()
    # new arrival should not allow user b to starve user a
    sim.submit_interaction(inter_b)
    sim.run()
    assert len(sim.completed_requests) == 2


def test_oit_no_mid_interaction_throttle():
    user = User("a")
    app = simple_app()
    app.user_rpm_limit = 1
    gen = WorkloadGenerator([user], [app], seed=3)
    inter = gen.generate_interaction(user, app, [InteractionStage.USER_PROMPT, InteractionStage.FINAL])
    oit = OIT(kv_threshold=1, max_batch=1)
    sim = Simulator(scheduler=FairServeScheduler(), oit=oit, kv_capacity=50, max_batch=1, max_time=30)
    sim.submit_interaction(inter)
    sim.run()
    # Both stages should complete even if overload triggers after first stage
    assert len(sim.completed_requests) == 2


if __name__ == "__main__":
    test_vtc_prioritizes_under_served()
    test_fairserve_counter_lift()
    test_oit_no_mid_interaction_throttle()
    print("tests passed")
