#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/runpod_ssh.sh <runpod_l40s|runpod_a5000> [ssh args...]

Examples:
  scripts/runpod_ssh.sh runpod_l40s
  scripts/runpod_ssh.sh runpod_a5000 tmux attach -t train-joystick-smoke
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

target="$1"
shift

case "$target" in
  runpod_l40s)
    host="103.196.86.48"
    port="37586"
    user="root"
    ;;
  runpod_a5000)
    host="69.30.85.239"
    port="22081"
    user="root"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

exec ssh -i ~/.ssh/id_ed25519 -p "$port" "$user@$host" "$@"
