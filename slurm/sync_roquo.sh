#!/usr/bin/env bash
# Mirror the git-tracked tree to the roquo cluster without touching the
# cluster's existing venvs (.venv*) or run outputs.
# The cluster login node has no git credentials for this private repo, so we
# rsync the tracked file list instead of `git pull`.
set -euo pipefail

REMOTE_HOST="${ROQUO_HOST:-<USER>@<CLUSTER_LOGIN_HOST>}"
REMOTE_DIR="${ROQUO_DIR:-~/sqd-gqe}"
SSH_KEY="${ROQUO_SSH_KEY:-$HOME/.ssh/roquo_ed25519}"

cd "$(git rev-parse --show-toplevel)"
git ls-files > /tmp/sqd_gqe_tracked_files.txt

rsync -av --files-from=/tmp/sqd_gqe_tracked_files.txt \
    -e "ssh -i ${SSH_KEY}" \
    ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Synced $(wc -l < /tmp/sqd_gqe_tracked_files.txt) tracked files to ${REMOTE_HOST}:${REMOTE_DIR}"
