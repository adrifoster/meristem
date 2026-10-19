#!/usr/bin/env bash
#
# Submit the meristem.run smoke test pipeline.
#
# Usage (on a PBS machine, with the meristem conda environment active):
#   cp tests/run/smoke/smoke_settings.example.sh tests/run/smoke/smoke_settings.sh
#   # edit smoke_settings.sh
#   bash tests/run/smoke/submit_smoke_test.sh
#
# Results:
#   Check with: python tests/run/smoke/check_smoke_test.py <root>
#   where <root> is the root printed by this script.

set -euo pipefail

SMOKE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SETTINGS="${SMOKE_DIR}/smoke_settings.sh"

if [ ! -f "${SETTINGS}" ]; then
  echo "ERROR: ${SETTINGS} not found"
  echo "  cp ${SMOKE_DIR}/smoke_settings.example.sh ${SETTINGS}"
  echo "  and edit it for your machine, then rerun."
  exit 1
fi
# shellcheck source=/dev/null
source "${SETTINGS}"

# variables used in smoke_test.yaml
export SMOKE_DIR PROJECT SMOKE_ROOT

CONFIG="${SMOKE_DIR}/smoke_test.yaml"

echo "=== meristem.run smoke test ==="
echo "Config: ${CONFIG}"
echo "Root:   ${SMOKE_ROOT}"
echo ""

meristem run --create --config "${CONFIG}"
meristem run --root "${SMOKE_ROOT}" --submit

echo ""
echo "Smoke test submitted. Check status with:"
echo "  meristem run --root ${SMOKE_ROOT} --print-status"
echo ""
echo "When complete, verify results with:"
echo "  python ${SMOKE_DIR}/check_smoke_test.py ${SMOKE_ROOT}"
