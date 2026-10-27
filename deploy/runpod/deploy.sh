#!/bin/sh
# Provision the QuorumAI Phi-4-mini endpoint on RunPod Serverless (idempotent).
#   deploy/runpod/deploy.sh                  -> plan + read-only preflight only
#   deploy/runpod/deploy.sh --authorize-deploy -> create or reuse template + endpoint
# Needs RUNPOD_MANAGEMENT_API_KEY in the environment (never printed). Sends no inference.
set -eu
cd "$(dirname "$0")/../.."
python3 deploy/runpod/runpod_deploy.py plan
python3 deploy/runpod/runpod_deploy.py preflight
if [ "${1:-}" = "--authorize-deploy" ]; then
  python3 deploy/runpod/runpod_deploy.py deploy --authorize-deploy
  python3 deploy/runpod/runpod_deploy.py status
fi
