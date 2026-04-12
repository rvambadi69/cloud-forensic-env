from pydantic import BaseModel, model_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class LogEntry(BaseModel):
    timestamp: datetime
    event_source: str
    event_name: str
    user_identity: Dict[str, Any]
    source_ip: str
    request_parameters: Dict[str, Any]
    response_elements: Optional[Dict[str, Any]] = None
    is_malicious: bool = False

class Observation(BaseModel):
    current_log_index: int
    total_logs: int
    log_entry: Optional[LogEntry]
    investigation_so_far: str
    services: List[str] = []
    alerts: List[str] = []
    reward: float = 0.0   # required by OpenEnv server
    done: bool = False

class Action(BaseModel):
    action_type: Literal["analyze", "flag_suspicious", "reconstruct_path", "next"]
    notes: Optional[str] = None
    flagged_event_ids: Optional[List[int]] = None
    reconstructed_path: Optional[List[int]] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_value_payload(cls, data: Any) -> Any:
        """Support simple discrete payloads like {"value": 1} from generic UIs."""
        if not isinstance(data, dict):
            return data

        if "action_type" in data:
            return data

        value = data.get("value")
        if value is None:
            return data

        # Map common discrete actions to this environment's action schema.
        mapping = {
            0: "analyze",
            1: "next",
            2: "flag_suspicious",
            3: "reconstruct_path",
        }
        if isinstance(value, int) and value in mapping:
            coerced = dict(data)
            coerced["action_type"] = mapping[value]
            return coerced

        return data

class EnvironmentState(BaseModel):
    scenario_id: str
    current_step: int
    logs_analyzed: List[int]
    flags_made: List[int]
    attack_path_ground_truth: List[int]
    reward_accumulated: float
    done: bool

# Aliases for OpenEnv
CloudForensicAction = Action
CloudForensicObservation = Observation
