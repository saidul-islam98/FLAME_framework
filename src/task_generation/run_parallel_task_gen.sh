#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORKER_COUNT="${WORKER_COUNT:-8}"
TASKS_TAG="${1:-}"

if [[ -z "$TASKS_TAG" ]]; then
  TASKS_TAG="$(date +"_%Y%m%d_%H%M%S")"
fi

echo "Running standalone task generation in parallel via src.task_generation.runner"
echo "  tasks_tag: $TASKS_TAG"
echo "  worker_count: $WORKER_COUNT"

PIDS=()
for ((i=0; i<WORKER_COUNT; i++)); do
  CMD=(
    python3 -m src.task_generation.runner
    --tasks-tag "$TASKS_TAG"
    --worker-index "$i"
    --worker-count "$WORKER_COUNT"
  )

  echo "Starting worker $i/$WORKER_COUNT: ${CMD[*]}"
  "${CMD[@]}" &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more workers failed for tasks_tag=$TASKS_TAG" >&2
  exit 1
fi

echo "All workers finished successfully for tasks_tag=$TASKS_TAG"
echo "Reuse this same tasks_tag to resume:"
echo "  $TASKS_TAG"
