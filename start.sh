#!/usr/bin/env bash
set -euo pipefail

uvicorn app.api:app --host 0.0.0.0 --port 8000 &
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# Stop the other process when the container is asked to exit
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# Return as soon as either process exits, so Docker sees the failure and can restart the container
wait -n