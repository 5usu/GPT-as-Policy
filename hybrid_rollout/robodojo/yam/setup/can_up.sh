#!/usr/bin/env bash
# Bring every CAN interface up at 1 Mbit/s (what the YAM motors use). Needs sudo.
# Moves nothing: it configures the network interface, not the motors.
set -euo pipefail
ifaces=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^can' || true)
[ -n "$ifaces" ] || { echo "no CAN interfaces found (is the USB-CAN adapter plugged in?)"; exit 1; }
for i in $ifaces; do
  sudo ip link set "$i" down || true
  sudo ip link set "$i" up type can bitrate 1000000
  echo "$i: $(ip -br link show "$i")"
done
echo
echo "Persistent names (can_follower_l / can_follower_r) are set up per i2rt"
echo "docs/guides/set-persistent-can-ids.md; otherwise channel order can change on replug."
