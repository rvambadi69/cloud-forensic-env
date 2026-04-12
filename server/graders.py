"""Task graders for OpenEnv manifests.

Hackathon loaders may call either ``EasyGrader(env)`` or ``EasyGrader()(env)``.
The metaclass supports both; scores always pass through ``grade_*`` and
``_safe_score`` so values stay strictly in (0, 1).
"""

from __future__ import annotations

from typing import Any

from cloud_forensic_env.server.cloud_forensic_env_environment import (
    grade_easy,
    grade_medium,
    grade_hard,
)


class _GraderCallMeta(type):
    """If first argument looks like an env, return a float; else build an instance."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if args and hasattr(args[0], "compute_score"):
            return cls._grade(args[0])
        return super().__call__(*args, **kwargs)


class EasyGrader(metaclass=_GraderCallMeta):
    @staticmethod
    def _grade(env: Any) -> float:
        return grade_easy(env)

    def __call__(self, env: Any) -> float:
        return grade_easy(env)


class MediumGrader(metaclass=_GraderCallMeta):
    @staticmethod
    def _grade(env: Any) -> float:
        return grade_medium(env)

    def __call__(self, env: Any) -> float:
        return grade_medium(env)


class HardGrader(metaclass=_GraderCallMeta):
    @staticmethod
    def _grade(env: Any) -> float:
        return grade_hard(env)

    def __call__(self, env: Any) -> float:
        return grade_hard(env)


__all__ = ["EasyGrader", "MediumGrader", "HardGrader", "grade_easy", "grade_medium", "grade_hard"]
