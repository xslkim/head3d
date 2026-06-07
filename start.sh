#!/usr/bin/env bash
# Start the head3d web service.
set -e
cd "$(dirname "$0")"

PORT="${PORT:-18001}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Always rebuild the extended template so it stays in sync with the current
# extension algorithm (27 points / 3 rings).
"$PY" build_template.py

ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "head3d -> http://localhost:${PORT}/  (LAN: http://${ip}:${PORT}/)"
exec "$PY" -m python.app --host 0.0.0.0 --port "$PORT"
