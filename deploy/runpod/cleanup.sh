#!/bin/sh
# Remove the QuorumAI Phi-4-mini endpoint and its template (idempotent: safe to re-run).
#   deploy/runpod/cleanup.sh        -> show what exists (read-only)
#   deploy/runpod/cleanup.sh --yes  -> scale to 0 workers, wait, delete endpoint, delete template
# Needs RUNPOD_MANAGEMENT_API_KEY in the environment (never printed).
set -eu
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "--yes" ]; then
  python3 deploy/runpod/runpod_deploy.py cleanup --yes
else
  python3 deploy/runpod/runpod_deploy.py status
  echo "dry run: pass --yes to delete" >&2
fi
