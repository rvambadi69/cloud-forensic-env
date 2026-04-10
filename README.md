---
title: Cloud Forensic Env Environment Server
emoji: 🎴
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Cloud Forensic Env Environment

Cloud Forensic Env is an OpenEnv-compatible environment for realistic cloud incident response tasks.
The agent investigates AWS-like audit logs, flags suspicious activity, and reconstructs an attack path.

This project is designed for the OpenEnv Hackathon workflow:
- real-world task (cloud forensics)
- 3 tasks with increasing difficulty
- typed Pydantic action/observation/state models
- deterministic programmatic reward signals during trajectory
- inference script at root (`inference.py`) using OpenAI client with env vars

## Environment Overview

The environment simulates security operations work:
- triaging suspicious cloud audit events
- collecting investigation notes
- identifying malicious events
- reconstructing the full attack chain

Task difficulty levels:
- easy: IAM privilege escalation
- medium: lateral movement across services
- hard: advanced persistence with multiple stages

## Action and Observation Spaces

### Action model

`Action` fields:
- `action_type`: one of `analyze`, `flag_suspicious`, `reconstruct_path`, `next`
- `notes` (optional): free-text analysis notes
- `flagged_event_ids` (optional): list of suspicious event indices
- `reconstructed_path` (optional): list of event indices in suspected attack order

Compatibility note:
- The API also accepts shorthand payloads like `{"action": {"value": 1}}` and maps them to a valid action type.

### Observation model

`Observation` fields:
- `current_log_index`: current position in timeline
- `total_logs`: total logs in scenario
- `log_entry`: current log object
- `investigation_so_far`: accumulated notes
- `services`: services involved in scenario
- `alerts`: scenario description/alert context
- `reward`: immediate reward from latest action
- `done`: episode termination flag

### State model

`EnvironmentState` includes:
- scenario id
- current step
- analyzed/flagged events
- ground-truth path
- accumulated reward
- done

## Reward Function

Reward is incremental and deterministic:
- `analyze` with notes: `+0.1`
- `flag_suspicious`:
    - correct unseen event flag: `+1.0`
    - incorrect flag: `-0.5`
- `reconstruct_path`:
    - score is fraction of correctly ordered steps in reconstructed path
    - missing reconstruction payload: `-1.0`
- `next`: no immediate reward

Termination:
- episode ends when reconstruction is attempted, or when logs are exhausted

## Tasks

Defined in `openenv.yaml` and backed by JSON scenarios in `server/attack_scenarios/`:
- `easy_iam_escalation` (easy)
- `medium_lateral_movement` (medium)
- `hard_advanced_persistence` (hard)

## Quick Start (Local)

From project directory:

```powershell
Set-Location 'c:\Users\Rahul V\OneDrive\Desktop\MSRIT_6th_Sem_Notes-main\openenv_hackathon\cloud_forensic_env'
..\scaler\Scripts\uv.exe run --active server
```

If port 8000 is busy, the server auto-selects the next free port and prints it.

Verify server:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
```

If auto-port switched, replace `8000` with the printed port.

Useful endpoints:
- `/` service metadata
- `/health` health probe
- `/docs` Swagger UI
- `/openapi.json` OpenAPI schema

## API Usage Example

Reset episode:

```json
POST /reset
{}
```

Step with explicit action schema:

```json
POST /step
{
    "action": {
        "action_type": "next"
    },
    "timeout_s": 30
}
```

Step with shorthand compatibility payload:

```json
POST /step
{
    "action": {
        "value": 1
    },
    "timeout_s": 30
}
```

## OpenEnv Validation

Run validation from environment root:

```bash
openenv validate
```

## Inference Baseline Script

The required script is at project root: `inference.py`.
It uses `openai.OpenAI` and reads:
- `API_BASE_URL` (default provided)
- `MODEL_NAME` (default provided)
- `HF_TOKEN` (required)

Run example:

```powershell
Set-Location 'C:\Users\Rahul V\OneDrive\Desktop\MSRIT_6th_Sem_Notes-main\openenv_hackathon\cloud_forensic_env'
$env:HF_TOKEN="<your_token>"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:TASK_NAME="easy"
& "C:\Users\Rahul V\openenv_env\Scripts\activate.ps1"
python .\inference.py
```

Use the existing virtual environment at `C:\Users\Rahul V\openenv_env`. Do not create a new one for this project.

Inference output format:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

### Baseline Performance

Baseline score is computed in `inference.py` as mean reward across steps and converted to success using threshold `0.6`.

To record benchmark numbers, run inference for all tasks and append the measured scores:
- easy: pending local run with valid `HF_TOKEN`
- medium: pending local run with valid `HF_TOKEN`
- hard: pending local run with valid `HF_TOKEN`

## Docker (Containerized Execution)

Build:

```bash
docker build -t cloud-forensic-env -f server/Dockerfile .
```

The root-level `Dockerfile` also supports the exact command used by the hackathon checks:

```bash
docker build -t cloud-forensic-env .
```

Run:

```bash
docker run --rm -p 8000:8000 cloud-forensic-env
```

Or, for the exact validator-style command:

```bash
docker run -p 8000:8000 cloud-forensic-env
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

## Hugging Face Spaces Deployment

From this directory:

```bash
openenv push
```

Recommended before submission:
- ensure only required Spaces are running
- ensure target Space status is `Running`
- confirm `/health` and `/docs` respond after deployment

## Project Structure

```text
cloud_forensic_env/
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── cloud_forensic_env_environment.py
│   ├── Dockerfile
│   └── attack_scenarios/
│       ├── easy_iam_escalation.json
│       ├── medium_lateral_movement.json
│       └── hard_advanced_persistence.json
└── uv.lock
```

The simplest way to use the Cloud Forensic Env environment is through the `CloudForensicEnv` class:

```python
from cloud_forensic_env import CloudForensicAction, CloudForensicEnv

try:
    # Create environment from Docker image
    cloud_forensic_envenv = CloudForensicEnv.from_docker_image("cloud_forensic_env-env:latest")

    # Reset
    result = cloud_forensic_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = cloud_forensic_envenv.step(CloudForensicAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    cloud_forensic_envenv.close()
```

That's it! The `CloudForensicEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t cloud_forensic_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**CloudForensicAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**CloudForensicObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Cloud Forensic Env environment server running, you can connect directly:

```python
from cloud_forensic_env import CloudForensicEnv

# Connect to existing server
cloud_forensic_envenv = CloudForensicEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = cloud_forensic_envenv.reset()
result = cloud_forensic_envenv.step(CloudForensicAction(message="Hello!"))
```

Note: When connecting to an existing server, `cloud_forensic_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from cloud_forensic_env import CloudForensicAction, CloudForensicEnv

# Connect with context manager (auto-connects and closes)
with CloudForensicEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(CloudForensicAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    CloudForensicEnvironment,  # Pass class, not instance
    CloudForensicAction,
    CloudForensicObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from cloud_forensic_env import CloudForensicAction, CloudForensicEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with CloudForensicEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(CloudForensicAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/cloud_forensic_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
cloud_forensic_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # CloudForensicEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── cloud_forensic_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
