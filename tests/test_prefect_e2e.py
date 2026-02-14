"""
E2E integration tests for Prefect flows.

Requires a running stack:
    docker compose up -d

Tests create deployments via the Prefect API, trigger flow runs,
and verify output metrics appear in VictoriaMetrics.
"""
import pytest
import time
import httpx
from datetime import datetime, timedelta, timezone

from conftest import (
    PREFECT_API_URL,
    VM_URL,
    WORK_POOL_NAME,
    vm_query,
    get_or_create_flow,
    create_deployment,
    trigger_and_wait,
)


# -- Flow / entrypoint metadata --
FORECAST_FLOW = "forecast_flow"
FORECAST_ENTRYPOINT = "anomaly_times.flows.forecasting:forecast_flow"

DETECTION_FLOW = "detect_anomalies_flow"
DETECTION_ENTRYPOINT = "anomaly_times.flows.detection:detect_anomalies_flow"

# Metric used for testing — must be seeded in VM first (via load_boom.py)
TEST_METRIC = "boom_metric_ds_0_T"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seed_boom_data(vm_client):
    """
    Ensure BOOM data is present in VictoriaMetrics.
    If not present, seed it using load_boom.py logic.
    """
    # Force delete existing to ensure clean timestamps
    vm_client.get("/api/v1/admin/tsdb/delete_series", params={"match[]": TEST_METRIC})
    time.sleep(2)
    
    results = vm_query(vm_client, TEST_METRIC)
    if results:
         # Should be empty now
         pass

    # Seed data
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/load_boom.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Failed to seed BOOM data: {result.stderr}"

    # Wait for VM to flush
    time.sleep(2)

    # Verify
    results = vm_query(vm_client, TEST_METRIC)
    assert len(results) > 0, "BOOM data still not found after seeding"


# ---------------------------------------------------------------------------
# Flow registration tests
# ---------------------------------------------------------------------------

class TestFlowRegistration:
    """Verify flows can be registered via API."""

    def test_register_forecast_flow(self, prefect_api):
        flow_id = get_or_create_flow(prefect_api, FORECAST_FLOW)
        assert flow_id, "Failed to register forecast_flow"

    def test_register_detection_flow(self, prefect_api):
        flow_id = get_or_create_flow(prefect_api, DETECTION_FLOW)
        assert flow_id, "Failed to register detect_anomalies_flow"


# ---------------------------------------------------------------------------
# Forecasting E2E tests (parametrized by model type)
# ---------------------------------------------------------------------------

class TestForecastDeployment:
    """
    For each model type, create a forecast deployment via the API,
    trigger a flow run, wait for completion, and verify forecast
    metrics are written to VictoriaMetrics.
    """

    @pytest.mark.parametrize(
        "model_type",
        [
            "arima",
            pytest.param("timesnet", marks=pytest.mark.skip(reason="Requires Ray with heavy ML deps")),
            pytest.param("chronos", marks=pytest.mark.skip(reason="Requires Ray with heavy ML deps")),
        ],
    )
    def test_forecast_e2e(self, prefect_api, vm_client, seed_boom_data, model_type):
        """
        Create deployment → trigger run → wait → verify output metrics.
        """
        deployment_name = f"test-forecast-{model_type}"

        # 1. Create deployment
        deployment_id = create_deployment(
            client=prefect_api,
            flow_name=FORECAST_FLOW,
            deployment_name=deployment_name,
            entrypoint=FORECAST_ENTRYPOINT,
            parameters={
                "promql": TEST_METRIC,
                "model_type": model_type,
                "lookback_minutes": 60,
                "forecast_horizon_minutes": 30,
                "tsdb_url": "http://victoria-metrics:8428",
            },
        )
        assert deployment_id, f"Failed to create {deployment_name}"

        # 2. Trigger and wait for completion
        run = trigger_and_wait(prefect_api, deployment_id)
        assert run["state"]["type"] == "COMPLETED"

        # 3. Verify forecast metrics in VictoriaMetrics
        # The flow writes anomaly_pred, anomaly_lower, anomaly_upper
        time.sleep(2)  # Give VM a moment to flush

        for metric_name in ["anomaly_pred", "anomaly_lower", "anomaly_upper"]:
            # Query from "tomorrow" looking back 48h to catch everything from yesterday to tomorrow
            results = vm_query(vm_client, f"{metric_name}[48h]", time_ts=time.time() + 86400)
            assert len(results) > 0, (
                f"Expected '{metric_name}' in VictoriaMetrics after "
                f"{model_type} forecast run, but found nothing"
            )

        # 4. Cleanup — delete deployment
        prefect_api.delete(f"/deployments/{deployment_id}")


# ---------------------------------------------------------------------------
# Anomaly detection E2E test
# ---------------------------------------------------------------------------

class TestDetectionDeployment:
    """
    Create a detection deployment, run it, and verify
    anomaly_score metrics are written to VictoriaMetrics.

    NOTE: Detection requires existing forecast data (anomaly_pred, etc.).
    This test depends on a successful forecast run having already written data.
    """

    def test_detection_e2e_run(self, prefect_api, vm_client, seed_boom_data):
        """
        Create detection deployment → trigger → wait → verify anomaly_score.
        """
        # First, ensure forecast data exists (run arima forecast if needed)
        # Check using robust future query
        pred_results = vm_query(vm_client, "anomaly_pred[48h]", time_ts=time.time() + 86400)
        
        if not pred_results:
            # Run a quick forecast first
            dep_id = create_deployment(
                client=prefect_api,
                flow_name=FORECAST_FLOW,
                deployment_name="test-forecast-prereq",
                entrypoint=FORECAST_ENTRYPOINT,
                parameters={
                    "promql": TEST_METRIC,
                    "model_type": "arima",
                    "lookback_minutes": 60,
                    "forecast_horizon_minutes": 30,
                    "tsdb_url": "http://victoria-metrics:8428",
                },
            )
            run = trigger_and_wait(prefect_api, dep_id)
            assert run["state"]["type"] == "COMPLETED", "Prerequisite forecast failed"
            prefect_api.delete(f"/deployments/{dep_id}")
            time.sleep(2)

        # Create detection deployment
        deployment_name = "test-detection"
        deployment_id = create_deployment(
            client=prefect_api,
            flow_name=DETECTION_FLOW,
            deployment_name=deployment_name,
            entrypoint=DETECTION_ENTRYPOINT,
            parameters={
                "promql": TEST_METRIC,
                "tsdb_url": "http://victoria-metrics:8428",
            },
        )
        assert deployment_id

        # Trigger and wait
        run = trigger_and_wait(prefect_api, deployment_id)
        assert run["state"]["type"] == "COMPLETED"

        # Verify anomaly_score in VM with robust query
        time.sleep(2)
        results = vm_query(vm_client, "anomaly_score[48h]", time_ts=time.time() + 86400)
        assert len(results) > 0, (
            "Expected 'anomaly_score' in VictoriaMetrics after detection run, "
            "but found nothing. Check if forecast overlaps with actuals."
        )

        # Cleanup
        prefect_api.delete(f"/deployments/{deployment_id}")
