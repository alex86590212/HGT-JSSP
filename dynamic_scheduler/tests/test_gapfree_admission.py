import copy

import pytest

from dynamic_scheduler.environment import gapfree_admission as admission
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle


def vehicle(vid, arrival, duration, lane="", zone=1):
    return DynamicVehicle(vid, arrival, [zone], [duration], 1.0, lane)


def blocked_snapshot():
    vehicles = [vehicle(0, 8, 10, "N_S_R"), vehicle(1, 9, 1, "N_S_R"), vehicle(2, 8, 1, "S_N_R")]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 10)
    env.admit(vehicles[0])
    env.admit(vehicles[1])
    return env


def test_pending_exchange_advances_without_locks_and_counts_displaced_follower():
    env = blocked_snapshot()
    assert env.admission_check(env._vehicles_by_id[2]) == (False, 18)
    before = copy.deepcopy(env.__dict__)
    proposals = admission._commitment_exchange_proposals(env, {0, 1, 2})
    proposal = next(p for p in proposals if p.target_id == 2)
    assert proposal.blocker_ids == frozenset({0, 1})
    assert proposal.deadline == 5.5
    assert proposal.gain == 9
    assert proposal.departures == {0: 9, 1: 19, 2: 8}
    assert proposal.current.committed_s1 == {2: 8}
    assert proposal.current.current_time == 0
    assert env.__dict__ == before


def test_exchange_round_applies_only_current_admissions_and_recomputes_until_converged():
    env = blocked_snapshot()
    stats = {}
    admission._run_commitment_exchange_round(env, {0, 1, 2}, stats=stats)
    assert env.committed_s1 == {2: 8}
    assert env.current_time == 0
    assert env.completed_log == []
    assert stats["accepted"] == 1
    assert stats["evaluated"] == 6
    assert stats["planned_gain"] == 9
    assert admission._project_detected_departures(env, {0, 1, 2}) == {0: 9, 1: 19, 2: 8}
    assert admission._commitment_exchange_proposals(env, {0, 1, 2}) == []


def test_episode_exchanges_before_immediately_eligible_lock_gate():
    vehicles = [vehicle(0, 1, 10), vehicle(1, 1, 1)]
    stats = {}
    assert admission.run_admission_episode_commitment_exchange(
        vehicles, 20, 10, stats=stats,
    ) == (0.5, 1.0, 2)
    assert stats["accepted"] == 1
    assert stats["planned_gain"] == 9


def test_exchange_search_exceeds_former_budget_and_leaves_no_profitable_proposal():
    env = blocked_snapshot()
    for vid in range(3, 260):
        independent = vehicle(vid, 8, 1, zone=vid)
        env._vehicles_by_id[vid] = independent
        env.admit(independent)
    detected_ids = set(env._vehicles_by_id)
    stats = {}
    admission._run_commitment_exchange_round(env, detected_ids, stats=stats)
    assert env.committed_s1[2] == 8
    assert stats["accepted"] == 1
    assert stats["evaluated"] == 520
    assert stats["planned_gain"] == 9
    assert admission._commitment_exchange_proposals(env, detected_ids) == []


def test_projection_retries_at_commitment_events_using_actual_admission_rules():
    vehicles = [vehicle(0, 0, 5), vehicle(1, 4, 1, zone=3),
                DynamicVehicle(2, 0, [2, 1], [4, 1], 1.0)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 10)
    env.admit(vehicles[0])
    env.lock_near()
    env.admit(vehicles[1])
    assert env.admission_check(vehicles[2]) == (False, 5)
    assert admission._project_detected_departures(env, {0, 1, 2}) == {0: 0, 1: 4, 2: 1.5}


def test_projection_retains_episode_boundary_admission_opportunity():
    vehicles = [vehicle(0, 0, 5), DynamicVehicle(1, 0, [2, 1], [4, 1], 1.0)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 1.5)
    env.admit(vehicles[0])
    env.lock_near()
    assert admission._project_detected_departures(env, {0, 1}) == {0: 0, 1: 1.5}


def test_advancing_target_is_rejected_when_displaced_vehicle_loses_more():
    vehicles = [vehicle(0, 8, 1), vehicle(1, 8, 10)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 30)
    env.admit(vehicles[0])
    before = copy.deepcopy(env.__dict__)
    assert admission._commitment_exchange_proposals(env, {0, 1}) == []
    admission._run_commitment_exchange_round(env, {0, 1})
    assert env.__dict__ == before


def test_exchange_preserves_existing_locked_route_and_log():
    env = blocked_snapshot()
    locked = vehicle(3, 0, 1, zone=2)
    env._vehicles_by_id[3] = locked
    env.admit(locked)
    env.lock_near()
    before_log = copy.deepcopy(env.completed_log)
    admission._run_commitment_exchange_round(env, {0, 1, 2, 3})
    assert env.locked_ids == {3}
    assert env.committed_s1 == {2: 8, 3: 0}
    assert env.zones[2].intervals == [(0, 1, 3)]
    assert env.completed_log == before_log
    assert env.current_time == 0


def test_locked_blocker_cannot_be_revised_after_eligibility():
    vehicles = [vehicle(0, 8, 10), vehicle(1, 8, 1)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 30)
    env.admit(vehicles[0])
    env.current_time = 5.5
    env.lock_near()
    before = copy.deepcopy(env.__dict__)
    assert admission._commitment_exchange_proposals(env, {0, 1}) == []
    assert env.__dict__ == before


def test_candidate_preserves_target_same_lane_predecessor():
    vehicles = [vehicle(0, 8, 10, "N_S_R"), vehicle(1, 7, 1, "S_N_R", zone=2),
                vehicle(2, 8, 1, "S_N_R")]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 30)
    env.admit(vehicles[0])
    env.admit(vehicles[1])
    proposal = next(p for p in admission._commitment_exchange_proposals(env, {0, 1, 2})
                    if p.target_id == 2)
    assert proposal.current.committed_s1 == {1: 7, 2: 8}
    assert proposal.departures == {0: 9, 1: 7, 2: 8}
    assert proposal.blocker_ids == frozenset({0})
    assert proposal.gain == 9


def test_projection_uses_returned_retry_instead_of_synthetic_continuous_gap():
    vehicles = [vehicle(0, 0, 5), DynamicVehicle(1, 0, [2, 1], [4, 1], 1.0)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 20)
    env.admit(vehicles[0])
    env.lock_near()
    assert admission._project_detected_departures(env, {0, 1}) == {0: 0, 1: 5}


@pytest.mark.parametrize("order, expected", [
    ("deadline", [2, 1, 3, 0]), ("arrival", [1, 3, 2, 0]), ("gain", [1, 3, 0, 2]),
])
def test_exchange_ordering_uses_all_lexicographic_tie_breaks(order, expected):
    env = admission.AdmissionEnv()
    env.reset([vehicle(0, 2, 1), vehicle(1, 1, 1), vehicle(2, 1, 1), vehicle(3, 1, 1)], 20)
    proposals = [admission._CommitmentExchange(vid, frozenset({9}), deadline, gain, {}, env)
                 for vid, deadline, gain in [(0, 5, 3), (3, 5, 4), (2, 4, 1), (1, 5, 4)]]
    assert [p.target_id for p in sorted(proposals, key=lambda p: admission._exchange_order_key(p, order))] == expected


def two_conflict_snapshot():
    vehicles = [vehicle(0, 8, 10), vehicle(1, 8, 1),
                vehicle(2, 6, 5, zone=2), vehicle(3, 6, 1, zone=2)]
    env = admission.AdmissionEnv()
    env.reset(vehicles, 30)
    env.admit(vehicles[0])
    env.admit(vehicles[2])
    return env


@pytest.mark.parametrize("order, current, planned_gain", [
    ("deadline", {0: 8, 3: 6}, 4),
    ("arrival", {0: 8, 3: 6}, 4),
    ("gain", {1: 8, 2: 6}, 9),
])
def test_same_profitable_candidates_select_different_real_exchanges(order, current, planned_gain):
    env = two_conflict_snapshot()
    proposals = admission._commitment_exchange_proposals(env, {0, 1, 2, 3})
    assert [(p.target_id, p.deadline, p.gain) for p in proposals] == [(1, 5.5, 9), (3, 3.5, 4)]
    chosen = min(proposals, key=lambda p: admission._exchange_order_key(p, order))
    assert chosen.current.committed_s1 == current
    assert chosen.gain == planned_gain


@pytest.mark.parametrize("order, accepted", [("deadline", 2), ("arrival", 2), ("gain", 1)])
def test_multiple_acceptances_recompute_gain_including_earlier_displacement(order, accepted):
    env = two_conflict_snapshot()
    stats = {}
    admission._run_commitment_exchange_round(env, {0, 1, 2, 3}, order=order, stats=stats)
    assert env.committed_s1 == {1: 8, 2: 6}
    assert stats["accepted"] == accepted
    assert stats["evaluated"] == 4 * (accepted + 1)
    assert stats["profitable"] == accepted + 1
    assert stats["planned_gain"] == 9
    assert admission._project_detected_departures(env, {0, 1, 2, 3}) == {0: 9, 1: 8, 2: 6, 3: 11}
    assert admission._commitment_exchange_proposals(env, {0, 1, 2, 3}) == []
