#!/usr/bin/env bash
# Convenience wrapper around run_cascade.py. Activates the venv setup.sh made and
# forwards all arguments. Examples:
#   ./run.sh --precision 4bit --show-gold --limit 20
#   ./run.sh --precision 8bit --only mcq
#   ./run.sh --backend stub --show-gold        # no-GPU wiring test
set -euo pipefail
cd "$(dirname "$0")"
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
exec python run_cascade.py "$@"
