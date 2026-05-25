#!/usr/bin/env bash
# head3d web service launcher
#
# Usage:
#   ./start.sh              # start in background (default port 18001)
#   ./start.sh start        # same as above
#   ./start.sh stop         # stop background service
#   ./start.sh restart      # restart
#   ./start.sh status       # show PID / health
#   ./start.sh fg           # run in foreground (see logs in terminal)
#   ./start.sh logs         # tail background log
#
# Environment overrides (optional):
#   HEAD3D_PORT=18001
#   HEAD3D_HOST=0.0.0.0
#   HEAD3D_DEVICE=cuda
#   HEAD3D_LANDMARK_BACKEND=subprocess

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON="${VENV}/bin/python"
HOST="${HEAD3D_HOST:-0.0.0.0}"
PORT="${HEAD3D_PORT:-18001}"
LANDMARK_BACKEND="${HEAD3D_LANDMARK_BACKEND:-subprocess}"
PID_FILE="${HEAD3D_PID_FILE:-/tmp/head3d_web_${PORT}.pid}"
LOG_FILE="${HEAD3D_LOG_FILE:-/tmp/head3d_web_${PORT}.log}"

usage() {
  sed -n '3,14p' "$0" | sed 's/^# \?//'
}

require_venv() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "error: virtualenv not found at $VENV" >&2
    echo "hint:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

port_in_use() {
  if command -v fuser >/dev/null 2>&1; then
    fuser "${PORT}/tcp" >/dev/null 2>&1
    return $?
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${PORT}" | grep -q ":${PORT}"
    return $?
  fi
  return 1
}

free_port() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 1
  fi
}

local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || true
}

print_urls() {
  local ip
  ip="$(local_ip)"
  echo
  echo "head3d web service is up on port ${PORT}"
  echo "  local:   http://127.0.0.1:${PORT}/hairline"
  echo "  preview: http://127.0.0.1:${PORT}/preview"
  if [[ -n "$ip" ]]; then
    echo "  lan:     http://${ip}:${PORT}/hairline"
    echo "  preview: http://${ip}:${PORT}/preview"
  fi
  echo "  health:  http://127.0.0.1:${PORT}/health"
  echo "  log:     ${LOG_FILE}"
  echo
}

build_cmd() {
  local -a cmd=(
    env PYTHONPATH="$ROOT"
    "$PYTHON" -m python.web_service
    --host "$HOST"
    --port "$PORT"
    --landmark-backend "$LANDMARK_BACKEND"
  )
  if [[ -n "${HEAD3D_DEVICE:-}" ]]; then
    cmd+=(--device "$HEAD3D_DEVICE")
  fi
  if [[ "${HEAD3D_DEBUG:-}" == "1" ]]; then
    cmd+=(--debug)
  fi
  if [[ "${HEAD3D_THREADED:-}" == "1" ]]; then
    cmd+=(--threaded)
  fi
  printf '%q ' "${cmd[@]}"
}

wait_for_health() {
  local i
  for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

do_start() {
  require_venv

  if is_running; then
    echo "already running (pid $(cat "$PID_FILE"), port ${PORT})"
    print_urls
    exit 0
  fi

  if port_in_use; then
    echo "port ${PORT} is in use; trying to free it ..."
    free_port
  fi

  local cmd
  cmd="$(build_cmd)"
  echo "starting: ${cmd}"
  nohup bash -c "${cmd}" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  if wait_for_health; then
    echo "started pid $(cat "$PID_FILE")"
    print_urls
  else
    echo "error: process started but /health did not respond in time" >&2
    echo "last log lines:" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    exit 1
  fi
}

do_stop() {
  local stopped=0

  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "stopping pid ${pid} ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        stopped=1
        break
      fi
      sleep 0.25
    done
    if [[ "$stopped" -eq 0 ]]; then
      echo "force killing pid ${pid} ..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  rm -f "$PID_FILE"

  if port_in_use; then
    echo "freeing port ${PORT} ..."
    free_port
  fi

  echo "stopped"
}

do_status() {
  if is_running; then
    echo "running: pid $(cat "$PID_FILE"), port ${PORT}"
    curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null && echo || true
    print_urls
  else
    echo "not running (port ${PORT})"
    if [[ -f "$LOG_FILE" ]]; then
      echo "last log: ${LOG_FILE}"
    fi
    exit 1
  fi
}

do_fg() {
  require_venv
  if is_running; then
    echo "background service already running on port ${PORT}; stop it first or use another port." >&2
    exit 1
  fi
  local cmd
  cmd="$(build_cmd)"
  echo "foreground: ${cmd}"
  eval "$cmd"
}

do_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "no log file yet: ${LOG_FILE}" >&2
    exit 1
  fi
  tail -f "$LOG_FILE"
}

ACTION="${1:-start}"
case "$ACTION" in
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  fg) do_fg ;;
  logs) do_logs ;;
  -h|--help|help) usage ;;
  *)
    echo "unknown command: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
