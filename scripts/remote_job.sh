#!/bin/bash
# Forced-command target for the NUC trigger.
# The NUC SSH key is restricted to this script in authorized_keys.
set -euo pipefail

export PATH=/home/oli/.local/bin:/usr/local/bin:/usr/bin:/bin
cd "$(dirname "$0")/.."

exec uv run --frozen --group data python -m scripts.job --cadence weekly_after_daily
