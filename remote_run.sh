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

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

DETACH=false
while getopts "d" opt; do
    case $opt in
        d) DETACH=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

SSH_HOST=${1:-}
SCRIPT_PATH=${2:-}
shift 2 || true
SCRIPT_ARGS="$@"

if [ -z "$SSH_HOST" ] || [ -z "$SCRIPT_PATH" ]; then
    echo "Usage: $0 [-d] <ssh-host> <script-path> [args...]"
    exit 1
fi

echo -e "${GREEN}Setting up remote environment on $SSH_HOST...${NC}"

ssh "$SSH_HOST" "bash -l" << REMOTE_SCRIPT
set -e

if [ -d "$REMOTE_DIR" ]; then
    echo "Updating existing repo..."
    cd "$REMOTE_DIR"
    git fetch origin
    git checkout main
    git pull origin main
else
    echo "Cloning repo..."
    git clone "$REPO_URL" "$REMOTE_DIR"
    cd "$REMOTE_DIR"
fi

# Install uv if needed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

source \$HOME/.local/bin/env 2>/dev/null || true

echo "Installing dependencies..."
uv sync --extra gpu

echo "Remote setup complete!"
REMOTE_SCRIPT

echo -e "${GREEN}Running benchmark on $SSH_HOST...${NC}"
echo -e "${BLUE}Script: $SCRIPT_PATH${NC}"
echo -e "${BLUE}Args: $SCRIPT_ARGS${NC}"

# Run the script
ssh "$SSH_HOST" "bash -l" << REMOTE_SCRIPT
cd "$REMOTE_DIR"
source \$HOME/.local/bin/env 2>/dev/null || true
echo "Starting: uv run python $SCRIPT_PATH $SCRIPT_ARGS"
uv run python $SCRIPT_PATH $SCRIPT_ARGS
REMOTE_SCRIPT

echo -e "${GREEN}Benchmark complete!${NC}"
