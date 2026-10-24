#!/bin/bash -l
set -Eeuo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
exec bash "$repo/run_rtt_iclr2027.sh" "$@"
