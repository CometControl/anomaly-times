"""
Script to dynamically create Prefect deployments via the API.

This allows you to create hundreds of forecasting deployments with different parameters.
Flows are automatically registered if they don't exist yet.

Usage:
    python scripts/create_deployments.py
"""
import httpx
import json

# Prefect API URL
PREFECT_API_URL = "http://localhost:4200/api"
WORK_POOL_NAME = "anomaly-pool"

# Flow definitions - maps flow name to its entrypoint
FLOWS = {
    "forecast_flow": {
        "name": "forecast_flow",
        "entrypoint": "src/anomaly_times/flows/forecasting.py:forecast_flow",
    },
    "detect_anomalies_flow": {
        "name": "detect_anomalies_flow", 
        "entrypoint": "src/anomaly_times/flows/detection.py:detect_anomalies_flow",
    },
}


def get_or_create_flow(flow_name: str) -> str:
    """
    Get the flow ID by name, or create/register it if it doesn't exist.
    Returns the flow ID.
    """
    # Try to find existing flow
    response = httpx.post(
        f"{PREFECT_API_URL}/flows/filter",
        json={"flows": {"name": {"any_": [flow_name]}}}
    )
    response.raise_for_status()
    flows = response.json()
    
    if flows:
        return flows[0]["id"]
    
    # Flow doesn't exist - create it
    print(f"  Flow '{flow_name}' not registered. Registering...")
    
    if flow_name not in FLOWS:
        raise ValueError(f"Unknown flow '{flow_name}'. Add it to FLOWS dict.")
    
    flow_def = FLOWS[flow_name]
    response = httpx.post(
        f"{PREFECT_API_URL}/flows/",
        json={"name": flow_name}
    )
    response.raise_for_status()
    flow_data = response.json()
    print(f"  ✅ Registered flow '{flow_name}' with ID: {flow_data['id']}")
    return flow_data["id"]


def create_deployment(
    flow_name: str,
    deployment_name: str,
    parameters: dict,
    schedule_interval_minutes: int = None,
    description: str = None
) -> dict:
    """
    Create a new deployment via the Prefect API.
    
    Args:
        flow_name: The name of the flow (e.g., "forecast_flow")
        deployment_name: Deployment name (e.g., "forecast-metric-a")
        parameters: Default parameters for this deployment
        schedule_interval_minutes: Optional schedule interval in minutes
        description: Optional description
    
    Returns:
        The created deployment data
    """
    flow_id = get_or_create_flow(flow_name)
    
    if flow_name not in FLOWS:
        raise ValueError(f"Unknown flow '{flow_name}'. Add it to FLOWS dict.")
    
    deployment_data = {
        "name": deployment_name,
        "flow_id": flow_id,
        "entrypoint": FLOWS[flow_name]["entrypoint"],
        "parameters": parameters,
        "work_pool_name": WORK_POOL_NAME,
        "description": description or f"Deployment for {deployment_name}",
        "enforce_parameter_schema": False,
    }
    
    # Add schedule if specified
    if schedule_interval_minutes:
        deployment_data["schedules"] = [
            {
                "schedule": {
                    "interval": schedule_interval_minutes * 60,  # seconds
                    "timezone": "UTC"
                },
                "active": True
            }
        ]
    
    response = httpx.post(
        f"{PREFECT_API_URL}/deployments/",
        json=deployment_data
    )
    response.raise_for_status()
    return response.json()


def bulk_create_deployments(configs: list[dict]) -> list[dict]:
    """
    Create multiple deployments from a configuration list.
    
    Args:
        configs: List of dicts with keys:
            - flow: Name of the flow (e.g., "forecast_flow")
            - metric: The PromQL query or metric name
            - schedule_minutes: Optional schedule interval
            - model_type: Optional model type override
            - ... any other flow parameters
    
    Returns:
        List of created deployment data
    """
    created = []
    
    for config in configs:
        flow_name = config.get("flow", "forecast_flow")
        metric = config["metric"]
        deployment_name = f"{flow_name}-{metric.replace('_', '-')}"
        
        # Build parameters based on flow type
        if flow_name == "forecast_flow":
            parameters = {
                "promql": metric,
                "model_type": config.get("model_type", "arima"),
                "lookback_minutes": config.get("lookback_minutes", 60),
                "forecast_horizon_minutes": config.get("forecast_horizon_minutes", 30),
            }
        elif flow_name == "detect_anomalies_flow":
            parameters = {
                "promql": metric,
            }
        else:
            parameters = config.get("parameters", {})
        
        try:
            deployment = create_deployment(
                flow_name=flow_name,
                deployment_name=deployment_name,
                parameters=parameters,
                schedule_interval_minutes=config.get("schedule_minutes"),
                description=config.get("description", f"{flow_name} for {metric}")
            )
            print(f"✅ Created: {deployment_name}")
            created.append(deployment)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                print(f"⚠️  '{deployment_name}' already exists, skipping")
            else:
                print(f"❌ Failed '{deployment_name}': {e.response.text}")
    
    return created


def delete_all_deployments():
    """Delete all existing deployments (useful for cleanup/reset)."""
    response = httpx.post(f"{PREFECT_API_URL}/deployments/filter", json={})
    response.raise_for_status()
    
    for deployment in response.json():
        httpx.delete(f"{PREFECT_API_URL}/deployments/{deployment['id']}")
        print(f"🗑️  Deleted: {deployment['name']}")


# Example usage
if __name__ == "__main__":
    # Optional: Clean up existing deployments first
    # print("Cleaning up existing deployments...")
    # delete_all_deployments()
    
    # Define your metrics and their configurations
    configs = [
        # Forecasting deployments
        {
            "flow": "forecast_flow",
            "metric": "boom_metric_ds_0_T",
            "schedule_minutes": 5,
            "model_type": "arima",
        },
        {
            "flow": "forecast_flow", 
            "metric": "boom_metric_ds_1_T",
            "schedule_minutes": 10,
            "model_type": "arima",
        },
        # Detection deployments
        {
            "flow": "detect_anomalies_flow",
            "metric": "boom_metric_ds_0_T",
            "schedule_minutes": 5,
        },
    ]
    
    print(f"Creating {len(configs)} deployments...\n")
    created = bulk_create_deployments(configs)
    print(f"\n✨ Created {len(created)} deployments!")
    print("View them at http://localhost:4200/deployments")
