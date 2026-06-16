#!/usr/bin/env bash
# =============================================================================
# setup.sh — thin alias for install.sh
#
# Useful when you are already inside the repo and want to update:
#   bash setup.sh
#
# Equivalent to:
#   bash ~/tracking/install.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${REPO_ROOT}/install.sh"
