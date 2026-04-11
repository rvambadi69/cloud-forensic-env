#!/usr/bin/env python3
"""
Validation script for OpenEnv Phase 2 deep validation requirements.
Tests that all grader functions return scores strictly between 0 and 1.
"""
import sys
import os
import yaml

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.cloud_forensic_env_environment import (
    CloudForensicEnv,
    grade_easy,
    grade_medium,
    grade_hard,
    make_env,
)

PASS = "\u2705"
FAIL = "\u274c"
errors = []


def check_strict_range(label: str, value: float):
    """Assert value is strictly between 0 and 1 (exclusive)."""
    ok = (0.0 < value < 1.0)
    status = PASS if ok else FAIL
    print(f"  {status} {label}: {value}")
    if not ok:
        errors.append(f"{label} = {value} is NOT in (0, 1)")


def test_safe_score():
    print("\n=== Test _safe_score() boundary clamping ===")
    for val, label in [
        (0.0, "exact zero"),
        (1.0, "exact one"),
        (-0.5, "negative"),
        (1.5, "over one"),
        (0.5, "normal 0.5"),
        (0.001, "lower boundary"),
        (0.999, "upper boundary"),
        (0.0001, "very small"),
        (0.9999, "very close to 1"),
    ]:
        result = CloudForensicEnv._safe_score(val)
        check_strict_range(f"_safe_score({val}) [{label}]", result)


def test_compute_score_edge_cases():
    print("\n=== Test compute_score() edge cases ===")
    
    for scenario_id in ["easy", "medium", "hard"]:
        env = make_env(scenario_id)
        
        # Case 1: No flags at all
        env._reset_state()
        score = env.compute_score()
        check_strict_range(f"{scenario_id}/no_flags", score)
        
        # Case 2: All correct flags
        env._reset_state()
        env.flags_made = list(env.ground_truth_path)
        score = env.compute_score()
        check_strict_range(f"{scenario_id}/all_correct", score)
        
        # Case 3: All wrong flags
        env._reset_state()
        env.flags_made = [999, 998, 997]
        score = env.compute_score()
        check_strict_range(f"{scenario_id}/all_wrong", score)
        
        # Case 4: Partial flags
        env._reset_state()
        if env.ground_truth_path:
            env.flags_made = [env.ground_truth_path[0]]
        score = env.compute_score()
        check_strict_range(f"{scenario_id}/partial", score)


def test_graders():
    print("\n=== Test grade_easy/medium/hard for all scenarios ===")
    
    for scenario_id, grader_fn, grader_name in [
        ("easy", grade_easy, "grade_easy"),
        ("medium", grade_medium, "grade_medium"),
        ("hard", grade_hard, "grade_hard"),
    ]:
        env = make_env(scenario_id)
        
        # Empty state
        env._reset_state()
        score = grader_fn(env)
        check_strict_range(f"{grader_name}({scenario_id})/empty", score)
        
        # All correct
        env._reset_state()
        env.flags_made = list(env.ground_truth_path)
        score = grader_fn(env)
        check_strict_range(f"{grader_name}({scenario_id})/all_correct", score)
        
        # All wrong
        env._reset_state()
        env.flags_made = [999, 998, 997]
        score = grader_fn(env)
        check_strict_range(f"{grader_name}({scenario_id})/all_wrong", score)
        
        # No flags
        env._reset_state()
        score = grader_fn(env)
        check_strict_range(f"{grader_name}({scenario_id})/no_flags", score)


def test_openenv_yaml():
    print("\n=== Test openenv.yaml schema ===")
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tasks = config.get("tasks", [])
    task_count = len(tasks)
    ok = task_count >= 3
    status = PASS if ok else FAIL
    print(f"  {status} Task count: {task_count} (need >= 3)")
    if not ok:
        errors.append(f"Only {task_count} tasks found, need >= 3")
    
    for task in tasks:
        tid = task.get("id", "???")
        has_grader = bool(task.get("grader"))
        has_desc = bool(task.get("description"))
        has_name = bool(task.get("name"))
        has_id = bool(task.get("id"))
        
        status_g = PASS if has_grader else FAIL
        status_d = PASS if has_desc else FAIL
        print(f"  {status_g} Task '{tid}' has grader: {task.get('grader', 'MISSING')}")
        print(f"  {status_d} Task '{tid}' has description: {'yes' if has_desc else 'MISSING'}")
        
        if not has_grader:
            errors.append(f"Task {tid} missing grader")
        if not has_desc:
            errors.append(f"Task {tid} missing description")


if __name__ == "__main__":
    print("=" * 60)
    print(" OpenEnv Phase 2 Grader Validation")
    print("=" * 60)
    
    test_safe_score()
    test_compute_score_edge_cases()
    test_graders()
    test_openenv_yaml()
    
    print("\n" + "=" * 60)
    if errors:
        print(f" {FAIL} FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print(f" {PASS} ALL CHECKS PASSED — ready for submission!")
        sys.exit(0)
