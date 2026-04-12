import json
import math
import os
from typing import Dict, Any
from pathlib import Path
try:
    from cloud_forensic_env.models import Observation, Action, LogEntry
except ImportError:
    from models import Observation, Action, LogEntry


class CloudForensicEnv:
    def __init__(self, scenario_path: str | None = None):
        # Initialize fields up-front so reset/step never hit missing attributes.
        self.scenario_data = {}
        self.logs = []
        self.ground_truth_path = []
        self.services = []
        self.alerts = []
        self._reset_state()

        if scenario_path is None:
            scenario_id = os.getenv("OPENENV_SCENARIO", "easy")
            scenario_path = self._scenario_path_from_id(scenario_id)

        self._load_scenario(scenario_path)

    def _load_scenario(self, scenario_path: str) -> None:
        with open(scenario_path, 'r') as f:
            self.scenario_data = json.load(f)
        self.logs = [LogEntry(**log) for log in self.scenario_data['logs']]
        self.ground_truth_path = self.scenario_data['attack_path']
        self.services = self.scenario_data.get("services", [])
        self.alerts = [self.scenario_data.get("description", "")]

    @staticmethod
    def _scenario_path_from_id(scenario_id: str) -> str:
        scenario_map = {
            "easy": "easy_iam_escalation.json",
            "medium": "medium_lateral_movement.json",
            "hard": "hard_advanced_persistence.json",
            "easy_iam_escalation": "easy_iam_escalation.json",
            "medium_lateral_movement": "medium_lateral_movement.json",
            "hard_advanced_persistence": "hard_advanced_persistence.json",
        }
        file_name = scenario_map.get(scenario_id)
        if file_name is None:
            raise ValueError(
                f"Unknown scenario_id '{scenario_id}'. Expected one of: "
                "easy, medium, hard, easy_iam_escalation, medium_lateral_movement, hard_advanced_persistence"
            )

        candidates = [
            Path(__file__).resolve().parent / "attack_scenarios" / file_name,
            Path("/app/env/server/attack_scenarios") / file_name,
            Path.cwd() / "server" / "attack_scenarios" / file_name,
        ]

        for scenario_file in candidates:
            if scenario_file.exists():
                return str(scenario_file)

        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Scenario file not found. Checked:\n{tried}")

    def _reset_state(self):
        self.current_step = 0
        self.logs_analyzed = []
        self.flags_made = []
        self.reward_total = 0.0
        self.done = False
        self.investigation_notes = ""

    # --- WINNING LOGIC: THE GRADER ---
    @staticmethod
    def _safe_score(value: float) -> float:
        """Clamp to strict (0, 1) exclusive range required by OpenEnv validator."""
        x = float(value)
        if not math.isfinite(x):
            x = 0.5
        # Use more conservative bounds to ensure strict (0, 1) compliance
        return float(max(0.05, min(0.95, round(x, 6))))

    def compute_score(self) -> float:
        """Calculates progress-based reward. Always returns strictly (0, 1)."""
        if not self.ground_truth_path:
            return self._safe_score(0.5)

        total = len(self.ground_truth_path)
        if total == 0:
            return self._safe_score(0.5)

        correct_flags = len(set(self.flags_made) & set(self.ground_truth_path))
        wrong_flags = len(set(self.flags_made) - set(self.ground_truth_path))

        progress = correct_flags / total
        penalty = wrong_flags / total

        # Base score of 0.2 for participation, up to 0.8 for accuracy
        score = 0.2 + (0.6 * progress) - (0.2 * penalty)

        return self._safe_score(score)

    async def reset(self) -> Observation:
        self._reset_state()

        if not self.logs:
            scenario_id = os.getenv("OPENENV_SCENARIO", "easy")
            self._load_scenario(self._scenario_path_from_id(scenario_id))

        if not self.logs:
            raise RuntimeError("No logs loaded for scenario; cannot reset environment")

        return Observation(
            current_log_index=0,
            total_logs=len(self.logs),
            log_entry=self.logs[0],
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=0.1,
            done=False,
        )

    async def step(self, action: Action) -> Observation:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        if not self.logs:
            raise RuntimeError("No logs loaded for scenario; cannot step environment")

        reward = 0.1

        if action.action_type == "analyze":
            if action.notes:
                self.investigation_notes += f"\nStep {self.current_step}: {action.notes}"
            reward = 0.1

        elif action.action_type == "flag_suspicious":
            if action.flagged_event_ids:
                for event_id in action.flagged_event_ids:
                    if event_id in self.ground_truth_path:
                        if event_id not in self.flags_made:
                            self.flags_made.append(event_id)
                            reward += (0.8 / max(1, len(self.ground_truth_path)))
                    else:
                        reward = 0.02

        elif action.action_type == "reconstruct_path":
            reward = self.compute_score()
            # Ensure final reward is safely within bounds and mark as success
            reward = min(0.7, reward)  # Cap at 0.7 to stay safely away from 1.0
            self.done = True

        elif action.action_type == "next":
            if self.current_step < len(self.logs) - 1:
                self.current_step += 1
                reward = 0.02
            else:
                self.done = True
                reward = self.compute_score()
                # Ensure final reward is safely within bounds
                reward = min(0.7, reward)  # Cap at 0.7 to stay safely away from 1.0

        final_step_reward = self._safe_score(reward)
        self.reward_total += final_step_reward

        obs = Observation(
            current_log_index=self.current_step,
            total_logs=len(self.logs),
            log_entry=self.logs[self.current_step] if self.current_step < len(self.logs) else None,
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=final_step_reward,
            done=self.done,
        )
        return obs

    @property
    def state(self):
        """Return a plain dict — avoids OpenEnvState schema validation crashes."""
        return {
            "step_count": self.current_step,
            "done": self.done,
            "reward_accumulated": self.reward_total,
        }
    
    def get_metadata(self):
        return {
            "name": "cloud_forensic_env",
            "description": "Cloud forensic investigation environment"    
       }



    def close(self) -> None:
        pass

    async def reset_async(self) -> Observation:
        return await self.reset()

    async def step_async(self, action: Action) -> Observation:
        return await self.step(action)

    async def close_async(self) -> None:
        self.close()

def make_env(scenario_id: str = "easy"):
    return CloudForensicEnv(CloudForensicEnv._scenario_path_from_id(scenario_id))


def grade_easy(env) -> float:
    """Grader for easy task — lenient scoring, clamped to strict (0, 1)."""
    base = env.compute_score()
    # Easy task: cap at 0.8 to leave more headroom
    return CloudForensicEnv._safe_score(min(0.8, base))


def grade_medium(env) -> float:
    """Grader for medium task — moderate penalty, clamped to strict (0, 1)."""
    base = env.compute_score()
    # Penalize incomplete flag coverage
    penalty = 0.1 if len(env.flags_made) < len(env.ground_truth_path) else 0.0
    return CloudForensicEnv._safe_score(min(0.85, base - penalty))


def grade_hard(env) -> float:
    """Grader for hard task — strict scoring, clamped to strict (0, 1)."""
    base = env.compute_score()
    # Penalize incomplete reconstruction
    path_match = len(set(env.flags_made) & set(env.ground_truth_path))
    completeness = path_match / max(1, len(env.ground_truth_path))
    adjusted = base * completeness * 0.8  # More conservative multiplier
    return CloudForensicEnv._safe_score(adjusted)


CloudForensicEnvironment = CloudForensicEnv