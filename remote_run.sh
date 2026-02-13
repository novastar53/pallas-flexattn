#!/bin/bash
# remote_run.sh - Run Python scripts on remote GPU machines
#
# Usage: ./remote_run.sh [-d] <ssh-host> <script-path> [args...]
#        ./remote_run.sh <ssh-host> <command>
#
# Options:
#   -d           - Detached mode (run in background with tmux)
#
# Commands:
#   attach       - Attach to running tmux session
#   status       - Check if script is running
#   stop         - Stop script and download logs

set -e

REMOTE_DIR="~/pallas-flexattn"
REPO_URL="https://github.com/novastar53/pallas-flexattn.git"
DEFAULT_SESSION="pallas_remote"
LOG_DIR="remote_logs"

DETACH=false
while getopts "d" opt; do
    case $opt in
        d) DETACH=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

SSH_HOST=${1:-}
SECOND_ARG=${2:-}

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 [-d] <ssh-host> <script-path> [args...]"
    exit 1
fi

COMMANDS="attach|status|stop"
if [[ "$SECOND_ARG" =~ ^($COMMANDS)$ ]]; then
    COMMAND="$SECOND_ARG"
else
    COMMAND="run"
fi

echo "Remote run script - Command: $COMMAND"
echo "This is a template. Customize for your remote workflow."
