"""
Shared fixtures for integration tests.
Assumes docker compose is running with Prefect, VictoriaMetrics, and worker.
"""
import pytest
import httpx
import time
from datetime import datetime, timedelta

PREFECT_API_URL = "http://localhost:4200/api"
VM_URL = "http://localhost:8428"
WORK_POOL_NAME = "anomaly-pool"

# Timeout for waiting on flow runs (seconds)
FLOW_RUN_TIMEOUT = 300
POLL_INTERVAL = 5


@pytest.fixture(scope="module")
def prefect_api():
    """httpx client for the Prefect API."""
    with httpx.Client(base_url=PREFECT_API_URL, timeout=30) as client:
        # Verify Prefect is reachable
        resp = client.get("/health")
        assert resp.status_code == 200, f"Prefect API not reachable: {resp.text}"
        yield client


@pytest.fixture(scope="module")
def vm_client():
    """httpx client for VictoriaMetrics."""
    with httpx.Client(base_url=VM_URL, timeout=30) as client:
        yield client


def vm_query(client: httpx.Client, query: str, time_ts: float = None) -> list:
    """Query VictoriaMetrics and return results."""
    params = {"query": query}
    if time_ts:
        params["time"] = time_ts
    resp = client.get("/api/v1/query", params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("result", [])


def get_or_create_flow(client: httpx.Client, flow_name: str) -> str:
    """Get flow ID by name, or create it."""
    resp = client.post("/flows/filter", json={"flows": {"name": {"any_": [flow_name]}}})
    resp.raise_for_status()
    flows = resp.json()
    if flows:
        return flows[0]["id"]

    resp = client.post("/flows/", json={"name": flow_name})
    resp.raise_for_status()
    return resp.json()["id"]


def create_deployment(
    client: httpx.Client,
    flow_name: str,
    deployment_name: str,
    entrypoint: str,
    parameters: dict,
) -> str:
    """Create a deployment and return its ID. If it already exists, delete and recreate."""
    flow_id = get_or_create_flow(client, flow_name)

    # Check if deployment exists
    resp = client.post(
        "/deployments/filter",
        json={"deployments": {"name": {"any_": [deployment_name]}}},
    )
    resp.raise_for_status()
    existing = resp.json()
    for dep in existing:
        if dep["name"] == deployment_name:
            client.delete(f"/deployments/{dep['id']}")

    # Create
    resp = client.post(
        "/deployments/",
        json={
            "name": deployment_name,
            "flow_id": flow_id,
            "entrypoint": entrypoint,
            "parameters": parameters,
            "work_pool_name": WORK_POOL_NAME,
            "enforce_parameter_schema": False,
            "path": "/app/src",
            "pull_steps": [
                {
                    "prefect.deployments.steps.set_working_directory": {
                        "directory": "/app/src"
                    }
                }
            ],
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


def trigger_and_wait(client: httpx.Client, deployment_id: str) -> dict:
    """
    Trigger a flow run from a deployment and wait for completion.
    Returns the final flow run object.
    Raises AssertionError on timeout or failure.
    """
    # Trigger
    resp = client.post(f"/deployments/{deployment_id}/create_flow_run", json={})
    resp.raise_for_status()
    flow_run = resp.json()
    flow_run_id = flow_run["id"]

    # Poll
    deadline = time.time() + FLOW_RUN_TIMEOUT
    while time.time() < deadline:
        resp = client.get(f"/flow_runs/{flow_run_id}")
        resp.raise_for_status()
        run = resp.json()
        state_type = run.get("state", {}).get("type", "")

        if state_type == "COMPLETED":
            return run
        elif state_type in ("FAILED", "CRASHED", "CANCELLED"):
            state_msg = run.get("state", {}).get("message", "")
            raise AssertionError(
                f"Flow run {flow_run_id} ended with state {state_type}: {state_msg}"
            )

        time.sleep(POLL_INTERVAL)

    raise AssertionError(
        f"Flow run {flow_run_id} timed out after {FLOW_RUN_TIMEOUT}s. "
        f"Last state: {state_type}"
    )
