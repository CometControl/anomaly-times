# 𝕬𝖓𝖔𝖒𝖆𝖑𝖞 𝕿𝖎𝖒𝖊𝖘

**Scalable, Real-Time Time Series Forecasting & Anomaly Detection Platform**

Anomaly Times is an infrastructure-agnostic library of Prefect Flows and Tasks designed to power a production-grade Anomaly Detection Platform. It integrates state-of-the-art forecasting models (Chronos, Nixtla) with a robust orchestration engine (Prefect) to detect anomalies in real-time Prometheus-compatible metrics.

## Features

*   **Platform-First Design**: Designed to be used as a backend service. Use the Prefect API/UI to spawn thousands of unique monitoring missions.
*   **State-of-the-Art Models**: We integrate the best open-source forecasting libraries (Nixtla, TSFM) and will continue to add more in the future.
*   **PromQL Native**: Inputs are raw PromQL queries. Supports Panel Data (multiple series per query) natively.
*   **Infrastructure Agnostic**:
    *   **Local**: Single command `docker-compose` stack.
    *   **Production**: High Availability Helm Charts for Kubernetes.
*   **Scalable**: Built on Prefect 3.0 and Ray for distributed task execution.

## Architecture

The system is composed of **Flow Templates** registered to a **Prefect Server**.
Admins deploy the templates; Users/Apps schedule them with parameters.

1.  **Forecasting Flow**: Periodically runs (e.g., hourly), fetches history, trains/predicts, and writes forecasts (`anomaly_pred`, `anomaly_lower`, `anomaly_upper`) to VictoriaMetrics (or via VMAgent to any Remote Write API).
2.  **Detection Flow**: Runs frequently (e.g., minutely), fetches real-time data, compares with forecasts (matching labels), and writes `anomaly_score`.

## Getting Started (Local)

Run the full stack (VictoriaMetrics, Grafana, PostgreSQL, Nginx, Prefect Server HA, Worker) locally:

```bash
docker-compose up --build
```

- **Prefect UI**: [http://localhost:4200](http://localhost:4200)
- **Grafana**: [http://localhost:3000](http://localhost:3000)
- **VictoriaMetrics**: [http://localhost:8428](http://localhost:8428)

### Deploying the Flows
Once the stack is up, register the base templates:

```bash
# Build image and register deployments
prefect deploy --all --param WORK_POOL_NAME=anomaly-pool --param IMAGE_NAME=anomaly-times:latest
```

## Production Deployment (Kubernetes)

Use the official Prefect Helm charts with our custom values.

```bash
# 1. Add Repo
helm repo add prefect https://prefecthq.github.io/prefect-helm

# 2. Install Server (HA)
helm install prefect-server prefect/prefect-server -f helm/prefect-server-values.yaml

# 3. Install Worker (Connects to Server)
helm install prefect-worker prefect/prefect-worker -f helm/prefect-worker-values.yaml
```

## Usage

To start monitoring a metric, successful deployment implies you just need to **Add a Schedule** to the base deployment.

### Via Prefect UI
1.  Navigate to **Deployments** -> **forecasting-deployment**.
2.  Click **+ Add Schedule**.
3.  Set **Cron**: `0 * * * *` (Hourly) OR use **Interval** (e.g., every 3600s).
4.  Set **Parameters** (JSON):
    ```json
    {
      "promql": "rate(http_requests_total[5m])",
      "model_type": "lgbm"
    }
    ```
5.  Repeat for **detection-deployment** (e.g., Cron `* * * * *`).

### Via API
POST to `/deployments/{id}/schedules` to programmatically create thousands of monitoring missions.

## Development

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```
