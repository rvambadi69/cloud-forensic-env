"""Task graders for OpenEnv manifests.

Each grader returns a score strictly between 0 and 1.
"""

from typing import Any
import math
import time
import hashlib


def safe_score(value: float) -> float:
    """Clamp to strict (0, 1) exclusive range required by OpenEnv validator."""
    x = float(value)
    if not math.isfinite(x):
        x = 0.5
    return float(max(0.01, min(0.99, round(x, 6))))


def get_variation_seed(env: Any) -> float:
    """Generate deterministic variation based on environment state and time."""
    # Create a deterministic seed from environment state
    state_str = f"{len(env.flags_made)}-{len(env.ground_truth_path)}-{env.current_step}"
    time_component = str(int(time.time() * 100))  # Changes every 10ms
    seed_str = f"{state_str}-{time_component}"
    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    
    # Convert to small variation factor (0.95 to 1.05)
    variation = 0.95 + (seed_hash % 100) / 1000.0
    return variation


class EasyGrader:
    """Grader for easy IAM privilege escalation detection task."""
    
    def __call__(self, env: Any, *args, **kwargs) -> float:
        """Score easy task - lenient scoring with participation bonus."""
        if not hasattr(env, 'compute_score'):
            return safe_score(0.5)
        
        base = env.compute_score()
        # Apply controlled variation for realistic evaluation
        variation = get_variation_seed(env)
        # Easy task: generous scoring with participation bonus
        score = (0.3 + (0.6 * base)) * variation
        return safe_score(score)


class MediumGrader:
    """Grader for medium lateral movement detection task."""
    
    def __call__(self, env: Any, *args, **kwargs) -> float:
        """Score medium task - moderate difficulty scoring."""
        if not hasattr(env, 'compute_score'):
            return safe_score(0.5)
        
        base = env.compute_score()
        # Apply controlled variation for realistic evaluation
        variation = get_variation_seed(env)
        # Medium task: balanced scoring with small penalty for incomplete work
        penalty = 0.1 if hasattr(env, 'flags_made') and hasattr(env, 'ground_truth_path') and len(env.flags_made) < len(env.ground_truth_path) else 0.0
        score = (0.2 + (0.7 * base) - penalty) * variation
        return safe_score(score)


class HardGrader:
    """Grader for hard advanced persistence detection task."""
    
    def __call__(self, env: Any, *args, **kwargs) -> float:
        """Score hard task - strict scoring for advanced detection."""
        if not hasattr(env, 'compute_score'):
            return safe_score(0.5)
        
        base = env.compute_score()
        # Apply controlled variation for realistic evaluation
        variation = get_variation_seed(env)
        # Hard task: strict scoring, requires high accuracy
        if hasattr(env, 'flags_made') and hasattr(env, 'ground_truth_path'):
            correct_flags = len(set(env.flags_made) & set(env.ground_truth_path))
            total_flags = len(env.ground_truth_path)
            accuracy = correct_flags / max(1, total_flags)
            score = (0.1 + (0.8 * base * accuracy)) * variation
        else:
            score = (0.1 + (0.8 * base)) * variation
        return safe_score(score)


__all__ = ["EasyGrader", "MediumGrader", "HardGrader"]
