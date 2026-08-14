"""Serve a model's weights via vLLM and wait for it to become ready.

Every evaluation funnels through the same OpenAI-compatible path: either the user
gives an inference-service IP (``base_url``) or we deploy the weights with vLLM
and point the harness at the resulting ``base_url`` (ARCHITECTURE.md section 2).
"""

from __future__ import annotations

import time

import httpx

from benchmark_diagnosis.config import ServingConfig


def serve_command(model_id: str, config: ServingConfig) -> list[str]:
    """Build the ``vllm serve`` argv for one model deployment.

    Args:
        model_id: Model identifier, served under the same name via
            ``--served-model-name``.
        config: Serving configuration (host/port/tensor parallel/...).

    Returns:
        The argv list ready for :func:`subprocess.run`/:func:`subprocess.Popen`.
    """
    cmd: list[str] = [
        "vllm",
        "serve",
        model_id,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--served-model-name",
        model_id,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    if config.max_model_len is not None:
        cmd += ["--max-model-len", str(config.max_model_len)]
    cmd += list(config.extra_args)
    return cmd


def wait_until_ready(
    base_url: str,
    *,
    timeout_seconds: float = 600.0,
    interval: float = 2.0,
) -> bool:
    """Poll the OpenAI-compatible ``/models`` endpoint until the server answers.

    Args:
        base_url: Endpoint root (e.g. ``http://localhost:8000/v1``); the health
            probe hits ``{base_url}/models`` (i.e. ``/v1/models``).
        timeout_seconds: Stop polling after this many seconds.
        interval: Seconds between probes.

    Returns:
        True if the endpoint returned HTTP 200 within the timeout, else False.
    """
    probe_url = f"{base_url.rstrip('/')}/models"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=interval) as client:
                resp = client.get(probe_url)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            # Server not up yet (connection refused / timeout); keep polling.
            pass
        time.sleep(interval)
    return False
