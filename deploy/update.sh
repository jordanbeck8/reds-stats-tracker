#!/usr/bin/env bash
# Full update cycle: pull → fetch stats → build site → publish → push data back.
# Runs on supernova via reds-update.service (nightly + post-game one-shots).
set -euo pipefail

REPO="${REDS_REPO:-$HOME/apps/reds-stats-tracker}"
SERVE_DIR="${REDS_SERVE:-$HOME/apps/reds-serve}"

cd "$REPO"
git pull --rebase --autostash

# 1. Fetch stats + schedule, write snapshots + README (venv python).
uv run python -m pipeline.run

# 2. Build the static site (loaders need the venv python on PATH).
export PATH="$REPO/.venv/bin:$HOME/.bun/bin:$PATH"
(cd site && bun install --frozen-lockfile && bun run build)

# 3. Publish atomically-ish: serve dir is only replaced after a good build.
mkdir -p "$SERVE_DIR"
rsync -a --delete site/dist/ "$SERVE_DIR/"

# 4. Push snapshots + README back to GitHub so the repo stays source of truth.
git add data README.md
if git diff --cached --quiet; then
  echo "No data changes — skipping commit."
else
  git commit -m "data: snapshot $(date +'%F %H:%M %Z')"
  git push
fi

echo "✓ update complete $(date -Is)"
