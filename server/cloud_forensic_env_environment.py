import json
import os
from typing import Dict, Any
from pathlib import Path
from cloud_forensic_env.models import Observation, Action, EnvironmentState, LogEntry

class CloudForensicEnv:
    def __init__(self, scenario_path: str | None = None):
        if scenario_path is None:
            scenario_id = os.getenv("OPENENV_SCENARIO", "easy")
            scenario_path = self._scenario_path_from_id(scenario_id)

        with open(scenario_path, 'r') as f:
            self.scenario_data = json.load(f)
        self.logs = [LogEntry(**log) for log in self.scenario_data['logs']]
        self.ground_truth_path = self.scenario_data['attack_path']
        self.services = self.scenario_data.get("services", [])
        self.alerts = [self.scenario_data.get("description", "")]
        self._reset_state()

    @staticmethod
    def _scenario_path_from_id(scenario_id: str) -> str:
        scenario_map = {
            "easy": "easy_iam_escalation.json",
            "medium": "medium_lateral_movement.json",
            "hard": "hard_advanced_persistence.json",
        }
        file_name = scenario_map.get(scenario_id)
        if file_name is None:
            raise ValueError(f"Unknown scenario_id '{scenario_id}'. Expected one of: easy, medium, hard")

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
    def _clamp(self, value: float) -> float:
        """Ensures all rewards stay in the production-safe hackathon range."""
        return max(0.01, min(value, 0.99))

    def compute_score(self) -> float:
        """Calculates progress-based reward. Production grade."""
        if not self.ground_truth_path:
            return 0.5
        
        # Calculate ratio of correctly identified suspicious events
        total =len(self.ground_truth_path)
        correct_flags = len(set(self.flags_made) & set(self.ground_truth_path))
        
        if total ==0:
            return 0.5
        
        progress = correct_flags / total
        
        # Base score of 0.1 for finishing, up to 0.8 for accuracy
        score = 0.1 + (0.8 * progress)
        return self._clamp(score)

    async def reset(self) -> Observation:
        self._reset_state()
        return Observation(
            current_log_index=0,
            total_logs=len(self.logs),
            log_entry=self.logs[0],
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=0.01,
            done=False,
        )

    async def step(self, action: Action) -> Any:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        reward = 0.01 # Minimal floor reward for taking an action

        if action.action_type == "analyze":
            if action.notes:
                self.investigation_notes += f"\nStep {self.current_step}: {action.notes}"
            reward = 0.05 # Small incentive for documentation

        elif action.action_type == "flag_suspicious":
            if action.flagged_event_ids:
                for event_id in action.flagged_event_ids:
                    if event_id in self.ground_truth_path:
                        if event_id not in self.flags_made:
                            self.flags_made.append(event_id)
                            # Incrementally reward discovery
                            reward += (0.5 / max(1,len(self.ground_truth_path)))
                    else:
                        # Soft penalty: no negative numbers, just low reward
                        reward = 0.02 

        elif action.action_type == "reconstruct_path":
            # The final exam: Use the weighted grader
            reward = self.compute_score()
            self.done = True

        elif action.action_type == "next":
            if self.current_step < len(self.logs) - 1:
                self.current_step += 1
                reward = 0.02
            else:
                self.done = True
                reward = self.compute_score()

        # Final production clamp
        final_step_reward = self._clamp(reward)
        self.reward_total += final_step_reward

        return Observation(
            current_log_index=self.current_step,
            total_logs=len(self.logs),
            log_entry=self.logs[self.current_step] if self.current_step < len(self.logs) else None,
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=final_step_reward,
            done=self.done,
        )

    async def state(self) -> EnvironmentState:
        return EnvironmentState(
            scenario_id=self.scenario_data['scenario_id'],
            current_step=self.current_step,
            logs_analyzed=self.logs_analyzed,
            flags_made=self.flags_made,
            attack_path_ground_truth=self.ground_truth_path,
            reward_accumulated=self.reward_total,
            done=self.done,
        )

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
    return env.compute_score()

def grade_medium(env) -> float:
    return env.compute_score()

def grade_hard(env) -> float:
    return env.compute_score()

CloudForensicEnvironment = CloudForensicEnv