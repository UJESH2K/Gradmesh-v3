#!/bin/sh

# GradMesh local process manager. Runtime state is kept inside .gradmesh/.
set -u

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME_DIR=${GRADMESH_RUNTIME_DIR:-"$ROOT_DIR/.gradmesh"}
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"
COORDINATOR_HOST=${GRADMESH_HOST:-0.0.0.0}
COORDINATOR_PORT=${GRADMESH_PORT:-8000}
COORDINATOR_URL=${GRADMESH_SERVER_URL:-${SERVER_URL:-"http://127.0.0.1:$COORDINATOR_PORT"}}
LOCAL_WORKER_COUNT=${GRADMESH_WORKER_COUNT:-${WORKER_COUNT:-1}}
LOCAL_WORKER_BACKEND=${GRADMESH_WORKER_BACKEND:-${WORKER_BACKEND:-auto}}
LOCAL_MAX_BATCH_SIZE=${GRADMESH_MAX_BATCH_SIZE:-${MAX_BATCH_SIZE:-2}}
LOCAL_HEARTBEAT_SECONDS=${GRADMESH_HEARTBEAT_SECONDS:-${HEARTBEAT_SECONDS:-5}}
LOCAL_POLL_SECONDS=${GRADMESH_POLL_SECONDS:-${POLL_SECONDS:-1.5}}

find_python() {
    if [ -n "${GRADMESH_PYTHON:-}" ]; then
        printf '%s\n' "$GRADMESH_PYTHON"
    elif [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
    elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
        printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    elif [ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
        printf '%s\n' "$ROOT_DIR/.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "GradMesh: Python was not found. Set PYTHON=/path/to/python." >&2
        exit 1
    fi
}

PYTHON_BIN=$(find_python)
mkdir -p "$PID_DIR" "$LOG_DIR"

safe_name() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-'
}

pid_is_running() {
    [ -f "$1" ] || return 1
    pid=$(sed -n '1p' "$1")
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

clean_stale_pid() {
    pid_file=$1
    if [ -f "$pid_file" ] && ! pid_is_running "$pid_file"; then
        rm -f "$pid_file"
    fi
}

start_server() {
    pid_file="$PID_DIR/server.pid"
    clean_stale_pid "$pid_file"
    if pid_is_running "$pid_file"; then
        echo "coordinator already running (pid $(sed -n '1p' "$pid_file"))"
        return 0
    fi
    (
        cd "$ROOT_DIR" || exit 1
        nohup "$PYTHON_BIN" -m uvicorn server:app --host "$COORDINATOR_HOST" --port "$COORDINATOR_PORT" \
            >>"$LOG_DIR/server.log" 2>&1 &
        echo $! >"$pid_file"
    )
    sleep 1
    if ! pid_is_running "$pid_file"; then
        echo "coordinator failed to start; inspect $LOG_DIR/server.log" >&2
        return 1
    fi
    echo "coordinator started (pid $(sed -n '1p' "$pid_file"), http://127.0.0.1:$COORDINATOR_PORT/dashboard)"
}

wait_for_server() {
    attempts=0
    while [ "$attempts" -lt 30 ]; do
        if "$PYTHON_BIN" -c "import urllib.request; urllib.request.urlopen('$COORDINATOR_URL/', timeout=1).read()" >/dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    echo "coordinator did not become reachable at $COORDINATOR_URL within 30 seconds" >&2
    return 1
}

start_worker() {
    worker_name=${1:-local-worker-1}
    key=$(safe_name "$worker_name")
    pid_file="$PID_DIR/worker-$key.pid"
    log_file="$LOG_DIR/worker-$key.log"
    clean_stale_pid "$pid_file"
    if pid_is_running "$pid_file"; then
        echo "worker '$worker_name' already running (pid $(sed -n '1p' "$pid_file"))"
        return 0
    fi
    wait_for_server || return 1
    (
        cd "$ROOT_DIR" || exit 1
        nohup "$PYTHON_BIN" worker.py \
            --server-url "$COORDINATOR_URL" \
            --name "$worker_name" \
            --node-id "local-$key" \
            --backend "$LOCAL_WORKER_BACKEND" \
            --max-batch-size "$LOCAL_MAX_BATCH_SIZE" \
            --heartbeat-seconds "$LOCAL_HEARTBEAT_SECONDS" \
            --poll-seconds "$LOCAL_POLL_SECONDS" \
            >>"$log_file" 2>&1 &
        echo $! >"$pid_file"
    )
    sleep 1
    if ! pid_is_running "$pid_file"; then
        echo "worker '$worker_name' failed to start; inspect $log_file" >&2
        return 1
    fi
    echo "worker '$worker_name' started (pid $(sed -n '1p' "$pid_file"), backend=$LOCAL_WORKER_BACKEND)"
}

start_workers() {
    count=${1:-$LOCAL_WORKER_COUNT}
    case "$count" in
        ''|*[!0-9]*|0) echo "worker count must be a positive integer" >&2; return 2 ;;
    esac
    index=1
    while [ "$index" -le "$count" ]; do
        start_worker "local-worker-$index" || return 1
        index=$((index + 1))
    done
}

stop_pid_file() {
    label=$1
    pid_file=$2
    if ! pid_is_running "$pid_file"; then
        clean_stale_pid "$pid_file"
        echo "$label is not running"
        return 0
    fi
    pid=$(sed -n '1p' "$pid_file")
    kill "$pid" 2>/dev/null || true
    attempts=0
    while kill -0 "$pid" 2>/dev/null && [ "$attempts" -lt 10 ]; do
        sleep 1
        attempts=$((attempts + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$label did not stop after 10 seconds; forcing pid $pid" >&2
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
    echo "$label stopped"
}

stop_worker() {
    worker_name=${1:?worker name is required}
    key=$(safe_name "$worker_name")
    stop_pid_file "worker '$worker_name'" "$PID_DIR/worker-$key.pid"
}

stop_workers() {
    found=0
    for pid_file in "$PID_DIR"/worker-*.pid; do
        [ -e "$pid_file" ] || continue
        found=1
        key=$(basename "$pid_file" .pid)
        stop_pid_file "$key" "$pid_file"
    done
    [ "$found" -eq 1 ] || echo "no managed workers are running"
}

show_status_line() {
    label=$1
    pid_file=$2
    clean_stale_pid "$pid_file"
    if pid_is_running "$pid_file"; then
        echo "RUNNING  $label  pid=$(sed -n '1p' "$pid_file")"
    else
        echo "STOPPED  $label"
    fi
}

status_all() {
    show_status_line coordinator "$PID_DIR/server.pid"
    found=0
    for pid_file in "$PID_DIR"/worker-*.pid; do
        [ -e "$pid_file" ] || continue
        found=1
        show_status_line "$(basename "$pid_file" .pid)" "$pid_file"
    done
    [ "$found" -eq 1 ] || echo "STOPPED  workers (none managed)"
}

show_logs() {
    target=${1:-server}
    if [ "$target" = "server" ]; then
        log_file="$LOG_DIR/server.log"
    else
        key=$(safe_name "$target")
        log_file="$LOG_DIR/worker-$key.log"
    fi
    [ -f "$log_file" ] || { echo "no log found at $log_file" >&2; return 1; }
    tail -n 100 "$log_file"
}

usage() {
    cat <<'EOF'
Usage:
  sh run.sh start                         Start coordinator + configured workers (default: 1)
  sh run.sh start server                  Start only the coordinator
  sh run.sh start workers [COUNT]         Start COUNT local workers
  sh run.sh start worker NAME             Start one named worker
  sh run.sh stop                          Stop all managed workers, then coordinator
  sh run.sh stop server                   Stop only the coordinator
  sh run.sh stop workers                  Stop all managed workers
  sh run.sh stop worker NAME              Stop one named worker
  sh run.sh restart                       Restart the local stack
  sh run.sh status                        Show managed processes
  sh run.sh logs [server|WORKER_NAME]      Print the last 100 log lines

Common environment variables:
  GRADMESH_WORKER_COUNT=3 GRADMESH_WORKER_BACKEND=cuda sh run.sh start
  GRADMESH_PORT=9000 GRADMESH_SERVER_URL=http://127.0.0.1:9000 sh run.sh start
  GRADMESH_PYTHON=/path/to/python GRADMESH_MAX_BATCH_SIZE=1 sh run.sh start workers 2
EOF
}

command=${1:-}
target=${2:-}
case "$command" in
    start)
        case "$target" in
            '') start_server && start_workers "$LOCAL_WORKER_COUNT" ;;
            server) start_server ;;
            workers) start_workers "${3:-$LOCAL_WORKER_COUNT}" ;;
            worker) [ -n "${3:-}" ] || { usage; exit 2; }; start_worker "$3" ;;
            *) usage; exit 2 ;;
        esac
        ;;
    stop)
        case "$target" in
            '') stop_workers; stop_pid_file coordinator "$PID_DIR/server.pid" ;;
            server) stop_pid_file coordinator "$PID_DIR/server.pid" ;;
            workers) stop_workers ;;
            worker) [ -n "${3:-}" ] || { usage; exit 2; }; stop_worker "$3" ;;
            *) usage; exit 2 ;;
        esac
        ;;
    restart)
        stop_workers
        stop_pid_file coordinator "$PID_DIR/server.pid"
        start_server && start_workers "$LOCAL_WORKER_COUNT"
        ;;
    status) status_all ;;
    logs) show_logs "${2:-server}" ;;
    help|-h|--help) usage ;;
    *) usage; exit 2 ;;
esac
