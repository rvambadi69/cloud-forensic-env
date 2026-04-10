# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI application for Cloud Forensic Environment
"""
import os
import socket
from fastapi import Response
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    raise ImportError("openenv-core required. Run: pip install openenv-core")

from cloud_forensic_env.models import CloudForensicAction, CloudForensicObservation
from cloud_forensic_env.server.cloud_forensic_env_environment import CloudForensicEnvironment

app = create_app(
    CloudForensicEnvironment,
    CloudForensicAction,
    CloudForensicObservation,
    env_name="cloud_forensic_env",
    max_concurrent_envs=1,
)


@app.get("/")
async def root() -> dict:
    return {
        "name": "cloud_forensic_env",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
        "web": "/web",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/web", include_in_schema=False)
async def web() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _resolve_bind_port(host: str, preferred_port: int) -> int:
    auto_port = os.getenv("OPENENV_AUTO_PORT", "1").lower() not in {"0", "false", "no"}
    if _is_port_available(host, preferred_port):
        return preferred_port

    if not auto_port:
        raise RuntimeError(
            f"Port {preferred_port} is already in use. "
            "Set OPENENV_PORT to a free port or enable OPENENV_AUTO_PORT=1."
        )

    for port in range(preferred_port + 1, preferred_port + 20):
        if _is_port_available(host, port):
            print(f"[cloud_forensic_env] Port {preferred_port} is busy, switching to {port}.")
            return port

    raise RuntimeError(
        f"No free port found in range {preferred_port}-{preferred_port + 19}. "
        "Set OPENENV_PORT to a free port."
    )


def main() -> None:
    import uvicorn

    host = os.getenv("OPENENV_HOST", "0.0.0.0")
    preferred_port = int(os.getenv("OPENENV_PORT", "8000"))
    bind_port = _resolve_bind_port(host, preferred_port)
    uvicorn.run(app, host=host, port=bind_port)


if __name__ == "__main__":
    main()