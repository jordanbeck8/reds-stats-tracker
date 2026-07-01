#!/bin/sh
cd "$(dirname "$0")/site"
export PATH="$(cd .. && pwd)/.venv/bin:$PATH"
exec bun run dev --port 3000 --no-open
