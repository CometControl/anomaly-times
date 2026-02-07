"""
Register flows with Prefect server (without creating deployments).
This runs on startup to make flows visible in the UI.
"""
import httpx
import sys

PREFECT_API_URL = "http://nginx:4200/api"

# All flows to register
FLOWS = [
    "forecast_flow",
    "detect_anomalies_flow",
]


def register_flows():
    """Register all flows with the Prefect server."""
    for flow_name in FLOWS:
        try:
            # Check if flow already exists
            response = httpx.post(
                f"{PREFECT_API_URL}/flows/filter",
                json={"flows": {"name": {"any_": [flow_name]}}},
                timeout=30
            )
            response.raise_for_status()
            
            if response.json():
                print(f"✓ Flow '{flow_name}' already registered")
                continue
            
            # Register the flow
            response = httpx.post(
                f"{PREFECT_API_URL}/flows/",
                json={"name": flow_name},
                timeout=30
            )
            response.raise_for_status()
            print(f"✓ Registered flow '{flow_name}'")
            
        except Exception as e:
            print(f"✗ Failed to register '{flow_name}': {e}")
            return 1
    
    print("\n✨ All flows registered!")
    return 0


if __name__ == "__main__":
    sys.exit(register_flows())
