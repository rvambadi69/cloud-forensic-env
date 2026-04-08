import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from server.cloud_forensic_env_environment import make_env
from cloud_forensic_env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TASK_NAME = os.getenv("TASK_NAME", "easy")
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = textwrap.dedent("""
You are a cloud security forensic investigator. You receive cloud audit logs one at a time.
- ANALYZE logs and take notes.
- FLAG suspicious events (index numbers).
- When confident, RECONSTRUCT the full attack path (list of indices in order).
- Use "next" to move to the next log.

Respond ONLY in JSON:
{
  "action_type": "analyze|flag_suspicious|reconstruct_path|next",
  "notes": "...",
  "flagged_event_ids": [0,1,...],
  "reconstructed_path": [0,1,2,...]
}
""").strip()

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def build_prompt(obs, history):
    log_entry = obs.log_entry
    return f"""
Current log ({obs.current_log_index+1}/{obs.total_logs}):
Event: {log_entry.event_name if log_entry else 'None'} from {log_entry.event_source if log_entry else 'None'}
User: {log_entry.user_identity if log_entry else 'None'}
IP: {log_entry.source_ip if log_entry else 'None'}
Parameters: {log_entry.request_parameters if log_entry else 'None'}

Investigation notes:
{obs.investigation_so_far}

Recent history: {history[-3:] if history else 'None'}

What is your next action? Respond with JSON.
"""

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = make_env(scenario_id=TASK_NAME)

    history = []
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env="cloud_forensic_env", model=MODEL_NAME)

    try:
        obs = await env.reset()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_prompt(obs, history)
            error_msg = None
            action = None

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content.strip()
                try:
                    action_dict = json.loads(response_text)
                    action = Action(**action_dict)
                except Exception:
                    action = Action(action_type="next", notes="JSON parse error")
            except Exception as e:
                action = Action(action_type="next")
                error_msg = str(e)

            step_result = await env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 4:
                obs, reward, done, _info = step_result
            elif isinstance(step_result, dict):
                obs = step_result["observation"]
                reward = step_result["reward"]
                done = step_result["done"]
            else:
                # OpenEnv HTTP-server-compatible direct envs may return Observation directly.
                obs = step_result
                reward = float(getattr(step_result, "reward", 0.0))
                done = bool(getattr(step_result, "done", False))

            rewards.append(reward)
            steps_taken = step
            log_step(step, action.action_type, reward, done, error_msg)
            history.append(f"Step {step}: {action.action_type} -> reward {reward:.2f}")

        # Calculate final score (simple average for compliance)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(success, steps_taken, rewards)

if __name__ == "__main__":
    asyncio.run(main())