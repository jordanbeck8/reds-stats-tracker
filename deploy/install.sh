#!/usr/bin/env bash
# One-time supernova setup: link systemd user units, enable timers, print
# the tailscale serve command (needs sudo/operator, so it's manual).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR" "$HOME/apps/reds-serve"
for unit in reds-update.service reds-update.timer reds-scheduler.service reds-scheduler.timer; do
  ln -sf "$REPO/deploy/systemd/$unit" "$UNIT_DIR/$unit"
done

systemctl --user daemon-reload
systemctl --user enable --now reds-update.timer reds-scheduler.timer

# Survive logout — timers keep firing without an active session.
loginctl enable-linger "$USER" || echo "warn: enable-linger failed (run: sudo loginctl enable-linger $USER)"

echo
echo "Timers enabled:"
systemctl --user list-timers 'reds-*' --no-pager
echo
echo "Serve the dashboard (one-time, persists across reboots):"
echo "  sudo tailscale serve --bg $HOME/apps/reds-serve"
echo
echo "Then browse: https://supernova.<tailnet>.ts.net"
