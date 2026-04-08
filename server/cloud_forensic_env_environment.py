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

        # Prefer colocated files next to this module (works in editable installs).
        candidates = [
            Path(__file__).resolve().parent / "attack_scenarios" / file_name,
            # Fallback for packaged installs inside containers where source is under /app/env.
            Path("/app/env/server/attack_scenarios") / file_name,
            # Fallback for local runs from repository root.
            Path.cwd() / "server" / "attack_scenarios" / file_name,
        ]

        for scenario_file in candidates:
            if scenario_file.exists():
                return str(scenario_file)

        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Scenario file not found. Checked:\n"
            f"{tried}"
        )

    def _reset_state(self):
        self.current_step = 0
        self.logs_analyzed = []
        self.flags_made = []
        self.reward_total = 0.0
        self.done = False
        self.investigation_notes = ""

    async def reset(self) -> Observation:
        self._reset_state()
        return Observation(
            current_log_index=0,
            total_logs=len(self.logs),
            log_entry=self.logs[0],
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=0.0,
            done=False,
        )

    async def step(self, action: Action) -> Observation:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        reward = 0.0

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
                            reward += 1.0
                    else:
                        reward -= 0.5

        elif action.action_type == "reconstruct_path":
            if action.reconstructed_path:
                correct = 0
                for i, step_id in enumerate(action.reconstructed_path):
                    if i < len(self.ground_truth_path) and step_id == self.ground_truth_path[i]:
                        correct += 1
                reward = correct / len(self.ground_truth_path) if self.ground_truth_path else 0.0
            else:
                reward = -1.0
            self.done = True

        elif action.action_type == "next":
            if self.current_step < len(self.logs) - 1:
                self.current_step += 1
            else:
                self.done = True

        self.reward_total += reward

        next_obs = Observation(
            current_log_index=self.current_step,
            total_logs=len(self.logs),
            log_entry=self.logs[self.current_step] if self.current_step < len(self.logs) else None,
            investigation_so_far=self.investigation_notes,
            services=self.services,
            alerts=self.alerts,
            reward=reward,
            done=self.done,
        )

        return next_obs

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
        # No external resources to release for this in-memory environment.
        pass

    # Aliases required by OpenEnv HTTP server
    async def reset_async(self) -> Observation:
        return await self.reset()

    async def step_async(self, action: Action) -> Observation:
        return await self.step(action)

    async def close_async(self) -> None:
        self.close()

# Factory function (required by openenv.yaml)
def make_env(scenario_id: str = "easy"):
    return CloudForensicEnv(CloudForensicEnv._scenario_path_from_id(scenario_id))

# For app.py compatibility
CloudForensicEnvironment = CloudForensicEnv