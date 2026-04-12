#!/usr/bin/env python3
"""
Local Phase-2-style checks for OpenEnv task graders (manifest + score bounds).

Mirrors common hackathon rules:
  - At least MIN_TASKS tasks, each with a non-empty ``grader`` import path.
  - Grader must be importable and *callable as* ``grader(env)`` (not ``GraderClass(env)``).
  - Every sampled score must satisfy: finite and strictly 0 < score < 1.

Run from the environment root (directory containing openenv.yaml):

  python tools/validate_phase2_graders.py

Exit code 0 = all checks passed; non-zero = failure with details on stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Callable

MIN_TASKS = 3
# Exclusive bounds (validator wording: not 0.0, not 1.0)
LOW, HIGH = 0.0, 1.0


def _load_manifest(env_root: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from e
    path = env_root / "openenv.yaml"
    if not path.is_file():
        raise SystemExit(f"Missing manifest: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("openenv.yaml must parse to a mapping")
    return data


def _resolve_grader(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"Invalid grader spec (expected 'module:callable'): {spec!r}")
    mod_name, attr = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    obj = getattr(module, attr)
    if isinstance(obj, type):
        raise ValueError(
            f"{spec!r} is a class. Validators call grader(env); use a function or "
            f"functools.partial, not a class that cannot be invoked as Class(env)."
        )
    if not callable(obj):
        raise ValueError(f"{spec!r} is not callable")
    return obj


def _assert_open_unit_interval(name: str, score: Any) -> None:
    if isinstance(score, bool):
        raise AssertionError(f"{name}: score is bool {score!r} (forbidden)")
    if not isinstance(score, (int, float)):
        raise AssertionError(f"{name}: score must be int or float, got {type(score).__name__}")
    x = float(score)
    if not math.isfinite(x):
        raise AssertionError(f"{name}: score must be finite, got {score!r}")
    if not (LOW < x < HIGH):
        raise AssertionError(
            f"{name}: score must satisfy {LOW} < score < {HIGH} (strict), got {x!r}"
        )


TASK_TO_SCENARIO = {
    "easy_iam_escalation": "easy_iam_escalation",
    "medium_lateral_movement": "medium_lateral_movement",
    "hard_advanced_persistence": "hard_advanced_persistence",
}


async def _iter_sampled_scores(env: Any, grader_fn: Callable[..., Any]) -> list[tuple[str, float]]:
    """Build several deterministic env configurations and return (label, score) pairs."""

    async def fresh() -> None:
        await env.reset()

    out: list[tuple[str, float]] = []

    await fresh()
    out.append(("after_reset", float(grader_fn(env))))

    await fresh()
    env.flags_made = [-999, -1]
    out.append(("wrong_flags_only", float(grader_fn(env))))

    await fresh()
    for eid in list(getattr(env, "ground_truth_path", []) or []):
        if eid not in env.flags_made:
            env.flags_made.append(eid)
    out.append(("all_ground_truth_flagged", float(grader_fn(env))))

    await fresh()
    gt = list(getattr(env, "ground_truth_path", []) or [])
    if gt:
        env.flags_made = [gt[0], -123]
    else:
        env.flags_made = [-1]
    out.append(("mixed_or_empty_gt", float(grader_fn(env))))

    return out


async def run_checks(env_root: Path, verbose: bool) -> None:
    manifest = _load_manifest(env_root)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit("manifest['tasks'] must be a list")

    with_grader = [t for t in tasks if isinstance(t, dict) and t.get("grader")]
    if len(with_grader) < MIN_TASKS:
        raise SystemExit(
            f"Need at least {MIN_TASKS} tasks with a 'grader' field; found {len(with_grader)}"
        )

    from cloud_forensic_env.server.cloud_forensic_env_environment import make_env

    for task in with_grader:
        tid = task.get("id", "<missing id>")
        spec = task.get("grader")
        if not spec or not isinstance(spec, str):
            raise SystemExit(f"Task {tid!r}: grader must be a non-empty string")

        try:
            grader_fn = _resolve_grader(spec.strip())
        except Exception as e:
            raise SystemExit(f"Task {tid!r} grader {spec!r}: {e}") from e

        scenario = TASK_TO_SCENARIO.get(str(tid))
        if not scenario:
            raise SystemExit(
                f"Task {tid!r}: add a mapping in TASK_TO_SCENARIO for this task id"
            )

        env = make_env(scenario)
        try:
            labeled_scores = await _iter_sampled_scores(env, grader_fn)
        except Exception as e:
            raise SystemExit(f"Task {tid!r}: failed to sample grader scores: {e}") from e

        for label, score in labeled_scores:
            try:
                _assert_open_unit_interval(f"task={tid} state={label}", score)
            except AssertionError as e:
                raise SystemExit(str(e)) from e
            if verbose:
                print(f"OK  {tid:30} {label:28} score={score:.6f}")

    print(
        f"PASS  {len(with_grader)} task(s) with graders; "
        f"all sampled scores in ({LOW}, {HIGH}) (strict, finite)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing openenv.yaml (default: parent of tools/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_checks(args.env_root, args.verbose))


if __name__ == "__main__":
    main()
