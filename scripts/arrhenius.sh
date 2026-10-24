#!/bin/bash -l
# Dataset-specific wrappers call this script from the Arrhenius login node.
# All numerical work is submitted to Slurm; shared GPU jobs deliberately omit -N.
set -Eeuo pipefail
umask 002

fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == _worker ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || fail 'Internal workers require a Slurm allocation'
  run_dir="${2:?Missing run directory}"
  stage="${3:?Missing stage}"
  mapfile -t frozen < <(python3 - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
plan=json.loads((Path(sys.argv[1])/'orchestration/plan.json').read_text())
for name in ('code_dir','venv','base','python_module','build_module'):
    print(plan[name])
PY
  )
  (( ${#frozen[@]} == 5 )) || fail 'Cannot read the frozen run environment'
  export RTT_CODE_DIR="${frozen[0]}" RTT_VENV="${frozen[1]}" RTT_BASE="${frozen[2]}"
  export RTT_PYTHON_MODULE="${frozen[3]}" RTT_BUILD_MODULE="${frozen[4]}"
  type module >/dev/null 2>&1 || fail 'Lmod module command is unavailable'
  module --force purge
  module load "$RTT_BUILD_MODULE"
  module load "$RTT_PYTHON_MODULE"
  python_bin="${RTT_VENV}/bin/python"
  [[ -x "$python_bin" ]] || fail "Environment not found: $RTT_VENV; run scripts/setup_arrhenius.sh"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  export PYTHONPATH="${RTT_CODE_DIR}/src"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  export TORCH_HOME="${RTT_BASE}/cache/torch" XDG_CACHE_HOME="${RTT_BASE}/cache/xdg"
  export PIP_CACHE_DIR="${RTT_BASE}/cache/pip"
  export MPLCONFIGDIR="${run_dir}/cache/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p -- "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$run_dir/environment"
  suffix="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
  module list > "$run_dir/environment/modules_${suffix}.txt" 2>&1
  "$python_bin" -m pip freeze > "$run_dir/environment/packages_${suffix}.txt"
  nvidia-smi > "$run_dir/environment/gpu_${suffix}.txt"
  cd -- "$RTT_CODE_DIR"
  exec srun --ntasks=1 "$python_bin" -u -m rtt.hpc worker --run-dir "$run_dir" --stage "$stage"
fi

usage() {
  cat <<'EOF'
Usage:
  bash experiments/<dataset>/run.sh prepare [RUN_DIR]
  bash experiments/<dataset>/run.sh smoke [RUN_DIR]
  bash experiments/<dataset>/run.sh pilot [RUN_DIR]
  bash experiments/<dataset>/run.sh full [RUN_DIR]
  bash experiments/<dataset>/run.sh status RUN_DIR
  bash experiments/<dataset>/run.sh resume RUN_DIR

Or call: bash scripts/arrhenius.sh <dataset> <command> [RUN_DIR]

Setup: bash scripts/setup_arrhenius.sh
prepare permits documented dataset downloads; pilot/full expect prepared data.
Set RTT_DOWNLOAD=1 to explicitly allow downloads during training preparation.
smoke uses synthetic fixtures and never downloads a benchmark dataset.

Preserved account: naiss2025-5-643-gpu
Preserved base: /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls
Overrides: RTT_BASE, RTT_VENV, RTT_ACCOUNT, RTT_DATA_ROOT, RTT_CONFIG, RTT_CPUS,
           RTT_AUDIT_SHARDS, RTT_MAX_CONCURRENT, RTT_MEM_<STAGE>, RTT_TIME_<STAGE>
EOF
}

case "${1:---help}" in -h|--help|help) usage; exit 0 ;; esac
dataset="${1:?Missing dataset}"
mode="${2:---help}"
case "$mode" in -h|--help|help) usage; exit 0 ;; esac
case "$dataset" in
  ogbn_arxiv|good_arxiv_degree|good_arxiv_time|twitch|facebook100|elliptic|ogbn_products|imagenet_c|acs_income|ogbg_molpcba) ;;
  *) fail "Unknown dataset: $dataset" ;;
esac
case "$mode" in prepare|smoke|pilot|full|status|resume) ;; *) usage; fail "Unknown command: $mode" ;; esac

export RTT_BASE="${RTT_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}"
export RTT_ACCOUNT="${RTT_ACCOUNT:-naiss2025-5-643-gpu}"
export RTT_VENV="${RTT_VENV:-${RTT_BASE}/code/venv_arrhenius_gh200}"
export RTT_PYTHON_MODULE="${RTT_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
export RTT_BUILD_MODULE="${RTT_BUILD_MODULE:-GPU/buildenv-nvhpc/25.9-cu13.0}"
export RTT_CODE_DIR="${RTT_CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export RTT_DATA_ROOT="${RTT_DATA_ROOT:-${RTT_BASE}/datasets/rtt/${dataset}}"
export RTT_CPUS="${RTT_CPUS:-16}" RTT_MAX_CONCURRENT="${RTT_MAX_CONCURRENT:-2}"
hpc="${RTT_CODE_DIR}/src/rtt/hpc.py"
[[ -f "$hpc" ]] || fail "Missing orchestration module: $hpc"

if [[ "$mode" == status || "$mode" == resume ]]; then
  [[ -n "${3:-}" ]] || fail "$mode requires RUN_DIR"
  [[ "$mode" == status ]] || command -v sbatch >/dev/null || fail 'sbatch not found; submit on Arrhenius'
  exec python3 "$hpc" "$mode" --run-dir "$3"
fi

command -v sbatch >/dev/null || fail 'sbatch not found; submit on the Arrhenius login node'
config="${RTT_CONFIG:-${RTT_CODE_DIR}/experiments/${dataset}/config.json}"
[[ -f "$config" ]] || fail "Dataset configuration not found: $config"
run_dir="${3:-${RTT_BASE}/outputs/RTT_${dataset}_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p -- "$run_dir/logs" "$RTT_BASE/logs"
run_dir="$(cd -- "$run_dir" && pwd -P)"
profile="$mode"
extras=()
if [[ "$mode" == prepare ]]; then
  profile=full
  extras+=(--prepare-only)
  [[ "${RTT_DOWNLOAD:-1}" != 1 ]] || extras+=(--download)
elif [[ "$mode" != smoke && "${RTT_DOWNLOAD:-0}" == 1 ]]; then
  extras+=(--download)
fi
exec python3 "$hpc" submit --dataset "$dataset" --profile "$profile" \
  --config "$config" --repo "$RTT_CODE_DIR" --run-dir "$run_dir" \
  --data-root "$RTT_DATA_ROOT" "${extras[@]}"
