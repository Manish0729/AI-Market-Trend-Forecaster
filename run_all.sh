#!/bin/bash
set -euo pipefail

# Launch Streamlit (port 8502) and static homepage (port 5500)

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start Streamlit if not already running
if ! lsof -i :8502 >/dev/null 2>&1; then
  (
    source "$ROOT_DIR/.venv/bin/activate" || true
    cd "$ROOT_DIR/stock_forecaster"
    streamlit run app.py --server.headless true --server.port 8502
  ) &
else
  echo "Streamlit already running on port 8502"
fi

# Start static server for homepage
if ! lsof -i :5500 >/dev/null 2>&1; then
  (
    cd "$ROOT_DIR"
    python3 -m http.server 5500
  ) &
else
  echo "Homepage already served on port 5500"
fi

HOMEPAGE_URL="http://localhost:5500"
APP_URL="http://localhost:8502"

echo "Homepage: $HOMEPAGE_URL"
echo "App:      $APP_URL"

# Wait for ports then open browser tabs
wait_for_port() {
  local port="$1"; local tries=30;
  while ! lsof -i :"$port" >/dev/null 2>&1; do
    tries=$((tries-1)); [ "$tries" -le 0 ] && break; sleep 0.5;
  done
}

wait_for_port 5500
wait_for_port 8502

if command -v open >/dev/null 2>&1; then
  open "$HOMEPAGE_URL"
  open "$APP_URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$HOMEPAGE_URL" >/dev/null 2>&1 &
  xdg-open "$APP_URL" >/dev/null 2>&1 &
fi

