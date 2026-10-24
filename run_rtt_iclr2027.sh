#!/bin/bash -l
#SBATCH --account=naiss2025-5-643-gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --job-name=rtt_prepare
#SBATCH --output=RTT_controller_%j.out
#SBATCH --error=RTT_controller_%j.err

# Use `bash run_rtt_iclr2027.sh pilot` from the login node. The submission
# branch creates all log directories before sbatch and preserves these exact
# allocation/storage settings. It never starts training on a login node.
# There is deliberately no -N: Arrhenius shared-node jobs must not reserve
# a whole node merely to request one GPU.
set -Eeuo pipefail
umask 002

export RTT_BASE="${RTT_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}"
export RTT_ACCOUNT="${RTT_ACCOUNT:-naiss2025-5-643-gpu}"
export RTT_VENV="${RTT_VENV:-${RTT_BASE}/code/venv_arrhenius_gh200}"
export RTT_DATA_ROOT="${RTT_DATA_ROOT:-${RTT_BASE}/datasets/ogb_cache}"
export RTT_GRAPH_NPZ="${RTT_GRAPH_NPZ:-${RTT_BASE}/datasets/graph/rtt_ogbn_arxiv_graph.npz}"
export RTT_PYTHON_MODULE="${RTT_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
export RTT_BUILD_MODULE="${RTT_BUILD_MODULE:-GPU/buildenv-nvhpc/25.9-cu13.0}"
export RTT_CPUS="${RTT_CPUS:-16}"
export RTT_MAX_CONCURRENT="${RTT_MAX_CONCURRENT:-2}"

usage() {
  cat <<'EOF'
Reference-Tail Trust: native ogbn-arxiv on Arrhenius

  bash run_rtt_iclr2027.sh prepare            # download/validate/cache ogbn-arxiv only
  bash run_rtt_iclr2027.sh smoke [RUN_DIR]    # tiny SYNTHETIC correctness pipeline
  bash run_rtt_iclr2027.sh pilot [RUN_DIR]    # reduced counts, real ogbn-arxiv
  bash run_rtt_iclr2027.sh full [RUN_DIR]     # paper-sized native core protocol
  bash run_rtt_iclr2027.sh status RUN_DIR    # read-only local progress summary
  bash run_rtt_iclr2027.sh resume RUN_DIR    # same code, data, seeds and unfinished tasks
  bash run_rtt_iclr2027.sh reduce RUN_DIR    # regenerate reduction after completed tasks

First run: bash setup_arrhenius.sh
Defaults preserve account naiss2025-5-643-gpu and storage naiss2025-6-428.
The core pipeline includes native RTT, matched output correctors, calibration,
selection, all registered audit components, reserved-node audit and checker
diagnostics. Breadth, development sweeps and other paper datasets are excluded.
Smoke/pilot results do not support the paper's statistical claims.

Useful overrides:
  RTT_BASE, RTT_CODE_DIR, RTT_VENV, RTT_DATA_ROOT, RTT_GRAPH_NPZ, RTT_ACCOUNT
  RTT_SEED=1 RTT_CPUS=16 RTT_MAX_CONCURRENT=2
  RTT_TIME_PHASE2=20:00:00 RTT_MEM_PHASE2=110G (any upper-case stage name)
  RTT_AUDIT_SHARD=128 RTT_COMPONENT_SHARD=512
  RTT_CONFIG=/absolute/path/to/overrides.json (copied into each new suite)
Set RTT_GRAPH_PT only to explicitly import a trusted legacy graph.pt cache.
EOF
}

fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
mode="${1:---help}"
case "$mode" in -h|--help|help) usage; exit 0 ;; esac

# For ordinary bash use this is the real source file. Under sbatch, the wrapper
# explicitly exports RTT_CODE_DIR; direct sbatch use falls back to submit-dir.
if [[ -z "${RTT_CODE_DIR:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RTT_CODE_DIR="${SLURM_SUBMIT_DIR:?Set RTT_CODE_DIR for direct sbatch use}"
  else
    RTT_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  fi
fi
export RTT_CODE_DIR
[[ -f "${RTT_CODE_DIR}/rtt_pipeline.py" ]] || fail "Missing rtt_pipeline.py in RTT_CODE_DIR=${RTT_CODE_DIR}"

if [[ "$mode" == status ]]; then
  [[ -n "${2:-}" ]] || fail "status requires RUN_DIR"
  exec python3 "${RTT_CODE_DIR}/rtt_pipeline.py" status --suite "$2"
fi

case "$mode" in
  prepare|smoke|pilot|full|resume|reduce) ;;
  _worker|_gate|_initialize|_prepare|_resume|_reduce) ;;
  *) usage >&2; fail "Unknown command: $mode" ;;
esac

if [[ "$mode" != _* ]]; then
  case "$mode" in
    resume|reduce)
      [[ -n "${2:-}" ]] || fail "$mode requires RUN_DIR"
      run_dir="$(readlink -f -- "$2")"
      [[ -f "${run_dir}/000_MANIFEST/pipeline.json" ]] || fail "Not an initialized RTT suite: $run_dir"
      ;;
    *)
      run_dir="${2:-${RTT_BASE}/outputs/RTT_ogbn_arxiv_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
      mkdir -p -- "$run_dir"
      run_dir="$(cd -- "$run_dir" && pwd -P)"
      ;;
  esac
  mkdir -p -- "${RTT_BASE}/logs" "${run_dir}/logs"
  export RTT_CODE_DIR
  if [[ "$mode" == prepare ]]; then
    action=_prepare
  elif [[ "$mode" == resume || "$mode" == reduce ]]; then
    action="_$mode"
  else
    action=_initialize
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    command -v sbatch >/dev/null || fail "sbatch not found; run this command on the Arrhenius login node"
    job=$(sbatch --parsable --account="$RTT_ACCOUNT" --partition=gpu \
      --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=48G --gpus=1 \
      --time="${RTT_TIME_PREPARE:-03:00:00}" --job-name="rtt_${mode}" \
      --chdir="$RTT_CODE_DIR" --export=ALL \
      --output="${run_dir}/logs/controller_%j.out" \
      --error="${run_dir}/logs/controller_%j.err" \
      "${RTT_CODE_DIR}/run_rtt_iclr2027.sh" "$action" "$run_dir" "$mode")
    printf 'Submitted controller job %s\nRun directory: %s\nLogs: %s/logs\n' "$job" "$run_dir" "$run_dir"
    exit 0
  fi
  # Direct sbatch invocation runs the same controller in the existing allocation.
  set -- "$action" "$run_dir" "$mode"
  mode="$action"
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Internal worker commands require a Slurm allocation"
run_dir="${2:?Internal command needs RUN_DIR}"

# Resume/workers use the environment frozen with the suite. The small bootstrap
# uses only the host's standard-library Python, then switches to the ARM venv.
if [[ -f "${run_dir}/000_MANIFEST/pipeline.json" ]]; then
  mapfile -t saved < <(python3 - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1])/'000_MANIFEST/pipeline.json').read_text())
for key in ('code_dir', 'venv', 'base', 'python_module', 'build_module'):
    print(p[key])
PY
  )
  (( ${#saved[@]} == 5 )) || fail "Could not read frozen suite environment"
  export RTT_CODE_DIR="${saved[0]}" RTT_VENV="${saved[1]}" RTT_BASE="${saved[2]}"
  export RTT_PYTHON_MODULE="${saved[3]}" RTT_BUILD_MODULE="${saved[4]}"
fi

type module >/dev/null 2>&1 || fail "Lmod module command is unavailable"
module --force purge
module load "$RTT_BUILD_MODULE"
module load "$RTT_PYTHON_MODULE"
python_bin="${RTT_VENV}/bin/python"
[[ -x "$python_bin" ]] || fail "Missing venv. Run bash setup_arrhenius.sh first: $RTT_VENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_HOME="${RTT_BASE}/cache/torch" XDG_CACHE_HOME="${RTT_BASE}/cache/xdg"
export PIP_CACHE_DIR="${RTT_BASE}/cache/pip" MPLCONFIGDIR="${run_dir}/cache/matplotlib"
mkdir -p -- "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "${run_dir}/environment"
module list > "${run_dir}/environment/modules_${SLURM_JOB_ID}.txt" 2>&1
"$python_bin" -m pip freeze > "${run_dir}/environment/packages_${SLURM_JOB_ID}.txt"
nvidia-smi > "${run_dir}/environment/gpu_${SLURM_JOB_ID}.txt"
cd -- "$RTT_CODE_DIR"

case "$mode" in
  _initialize)
    srun --ntasks=1 "$python_bin" -u rtt_pipeline.py initialize --suite "$run_dir" --code-dir "$RTT_CODE_DIR" --mode "${3:?}"
    ;;
  _prepare)
    dataset_args=(--dataset_kind cached --dataset_name ogbn-arxiv --dataset_root "$RTT_DATA_ROOT" --rtt_graph_npz "$RTT_GRAPH_NPZ")
    [[ -z "${RTT_GRAPH_PT:-}" ]] || dataset_args+=(--graph_pt "$RTT_GRAPH_PT")
    [[ -z "${RTT_CONFIG:-}" ]] || dataset_args+=(--config "$RTT_CONFIG")
    mkdir -p -- "$(dirname -- "$RTT_GRAPH_NPZ")"
    exec 9>"${RTT_GRAPH_NPZ}.lock"
    flock -x 9
    srun --ntasks=1 "$python_bin" -u run_rtt_iclr2027_arrhenius.py --prepare_dataset_only --out_dir "$run_dir" "${dataset_args[@]}"
    ;;
  _worker|_gate)
    srun --ntasks=1 "$python_bin" -u rtt_pipeline.py "${mode#_}" --suite "$run_dir" --stage "${3:?}"
    ;;
  _resume|_reduce)
    srun --ntasks=1 "$python_bin" -u rtt_pipeline.py "${mode#_}" --suite "$run_dir"
    ;;
esac
