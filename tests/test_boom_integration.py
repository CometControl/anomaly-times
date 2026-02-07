import requests
import time
from datetime import datetime

VM_QUERY_URL = "http://localhost:8428/api/v1/query"

def test_boom_data_loaded():
    """
    Verifies that BOOM dataset metrics are present in VictoriaMetrics.
    """
    # Define expected metrics
    # base name + series_id (sanitized)
    expected_series = [
        "boom_metric_ds_0_T",
        "boom_metric_ds_1_T"
    ]
    
    for series_name in expected_series:
        response = requests.get(VM_QUERY_URL, params={"query": series_name})
        assert response.status_code == 200, f"Failed to query {series_name}: {response.text}"
        
        data = response.json()
        assert data["status"] == "success"
        results = data["data"]["result"]
        
        assert len(results) > 0, f"No data found for metric {series_name}"
        
        print(f"Verified {series_name}: Found {len(results)} series.")
        
        # Check metadata/labels
        first_result = results[0]
        assert "item_id" in first_result["metric"]
        assert "source" in first_result["metric"]
        assert first_result["metric"]["source"] == "boom_dataset"

def test_boom_data_range():
    """
    Verifies that the data spans both past and future relative to 'now'.
    """
    series_name = "boom_metric_ds_0_T"
    
    now = time.time()
    
    # Check for future data
    # By default /query evaluates at 'now'. We must specify a future 'time' or use a range lookback from future.
    # We'll try to query at now + 2 days.
    future_check_time = now + (2 * 86400)
    
    response = requests.get(VM_QUERY_URL, params={"query": series_name, "time": future_check_time})
    results = response.json()["data"]["result"]
    
    assert len(results) > 0, "No data found when querying 2 days in the future!"
    
    last_point = results[0]["value"]
    future_timestamp = float(last_point[0])
    
    print(f"Current time: {datetime.fromtimestamp(now)}")
    print(f"Future query time: {datetime.fromtimestamp(future_check_time)}")
    print(f"Result timestamp: {datetime.fromtimestamp(future_timestamp)}")
    
    # Assert that the result is indeed in the future
    assert future_timestamp > now + 86400, f"Expected data in the future! Found: {future_timestamp}"

    # To check past data, we might need a range query or just assume if it exists and ends in future, it started in past.
    # But let's verify we have data from the past too.
    # Query for data from 1 hour ago
    past_timestamp = now - 3600
    response_past = requests.get(VM_QUERY_URL, params={"query": series_name, "time": past_timestamp})
    data_past = response_past.json()["data"]["result"]
    
    assert len(data_past) > 0, "No data found in the past (1 hour ago)"
    print(f"Verified past data exists at {datetime.fromtimestamp(past_timestamp)}")

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_boom_data_loaded()
        test_boom_data_range()
        print("All integration tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
