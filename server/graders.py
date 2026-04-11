from typing import Any


def _safe_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


class EasyGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        base = float(env.compute_score())
        return _safe_score(min(0.85, base))


class MediumGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        base = float(env.compute_score())
        penalty = 0.1 if len(getattr(env, "flags_made", [])) < len(getattr(env, "ground_truth_path", [])) else 0.0
        return _safe_score(min(0.9, base - penalty))


class HardGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        base = float(env.compute_score())
        path_match = len(set(getattr(env, "flags_made", [])) & set(getattr(env, "ground_truth_path", [])))
        completeness = path_match / max(1, len(getattr(env, "ground_truth_path", [])))
        adjusted = base * completeness * 0.9
        return _safe_score(adjusted)
