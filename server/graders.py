"""Programmatic graders for OpenEnv tasks.

Manifest loaders invoke ``grader(env)`` with the environment instance. Class-based
graders fail that contract because ``GraderClass(env)`` is invalid; use these
module-level functions instead.
"""

from cloud_forensic_env.server.cloud_forensic_env_environment import (
    grade_easy,
    grade_medium,
    grade_hard,
)

__all__ = ["grade_easy", "grade_medium", "grade_hard"]
