#!/bin/bash -l
#SBATCH --account=naiss2025-5-643-gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=rtt_setup
#SBATCH --output=RTT_setup_%j.out
#SBATCH --error=RTT_setup_%j.err
set -Eeuo pipefail
umask 002

export RTT_BASE="${RTT_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}"
export RTT_VENV="${RTT_VENV:-${RTT_BASE}/code/venv_arrhenius_gh200}"
export RTT_ACCOUNT="${RTT_ACCOUNT:-naiss2025-5-643-gpu}"
export RTT_PYTHON_MODULE="${RTT_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
export RTT_BUILD_MODULE="${RTT_BUILD_MODULE:-GPU/buildenv-nvhpc/25.9-cu13.0}"
if [[ -z "${RTT_CODE_DIR:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RTT_CODE_DIR="${SLURM_SUBMIT_DIR:?Set RTT_CODE_DIR for direct sbatch use}"
  else
    RTT_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  fi
fi
export RTT_CODE_DIR

if [[ "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash setup_arrhenius.sh
Creates the ARM64 venv at its final path and installs the pinned requirements.
An existing environment is checked and retained. To intentionally synchronize
an existing environment to these pins, use RTT_SETUP_UPDATE=1.
Use RTT_VENV=/a/separate/path to avoid modifying a shared environment.
EOF
  exit 0
fi

mkdir -p -- "${RTT_BASE}/logs" "${RTT_BASE}/cache/pip"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  job=$(sbatch --parsable --account="$RTT_ACCOUNT" --partition=gpu --ntasks=1 \
    --ntasks-per-node=1 --cpus-per-task=4 --mem=24G --gpus=1 --time=01:00:00 \
    --chdir="$RTT_CODE_DIR" --export=ALL --job-name=rtt_setup \
    --output="${RTT_BASE}/logs/RTT_setup_%j.out" --error="${RTT_BASE}/logs/RTT_setup_%j.err" \
    "${RTT_CODE_DIR}/setup_arrhenius.sh")
  printf 'Submitted environment setup/check job %s\nLogs: %s/logs/RTT_setup_*.out\n' "$job" "$RTT_BASE"
  exit 0
fi

module --force purge
module load "$RTT_BUILD_MODULE"
module load "$RTT_PYTHON_MODULE"
[[ "$(uname -m)" == aarch64 ]] || { echo 'This setup must execute on an Arrhenius Arm GPU node.' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR="${RTT_BASE}/cache/pip"
unset PYTHONPATH
mkdir -p -- "$(dirname -- "$RTT_VENV")"
exec 9>"${RTT_VENV}.setup.lock"
flock -x 9
install=0
if [[ ! -e "$RTT_VENV" ]]; then
  python3 -m venv "$RTT_VENV"
  install=1
elif [[ ! -x "${RTT_VENV}/bin/python" ]]; then
  printf 'Incomplete venv exists: %s. Inspect it or choose RTT_VENV; it will not be deleted.\n' "$RTT_VENV" >&2
  exit 1
fi
python_bin="${RTT_VENV}/bin/python"
if [[ "$install" == 1 || "${RTT_SETUP_UPDATE:-0}" == 1 ]]; then
  "$python_bin" -m pip install 'torch==2.12.1+cu130' --index-url https://download.pytorch.org/whl/cu130
  "$python_bin" -m pip install -r "${RTT_CODE_DIR}/requirements-arrhenius.txt"
else
  printf 'Checking existing venv without changing packages: %s\n' "$RTT_VENV"
fi
"$python_bin" -m pip check
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --ntasks=1 "$python_bin" - <<'PY'
import importlib, json, platform, sys
for name in ('numpy', 'scipy', 'matplotlib', 'pandas', 'ogb', 'tqdm', 'mpmath'):
    importlib.import_module(name)
import torch
if platform.machine() != 'aarch64' or not torch.cuda.is_available():
    raise SystemExit('Expected aarch64 with usable CUDA; environment preflight failed.')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
for dtype in (torch.float32, torch.float64):
    x = torch.tensor([[1., 2.], [3., 4.]], device='cuda', dtype=dtype)
    indices = torch.tensor([[0, 1], [0, 1]], device='cuda')
    op = torch.sparse_coo_tensor(indices, torch.ones(2, device='cuda', dtype=dtype), (2, 2)).coalesce()
    assert torch.equal(torch.sparse.mm(op, x), x)
    assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
print(json.dumps({'status': 'PASS', 'python': sys.version, 'architecture': platform.machine(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0)}, indent=2))
PY
"$python_bin" -m pip freeze > "${RTT_BASE}/logs/RTT_environment_${SLURM_JOB_ID}.txt"
flock -u 9
printf 'Environment checked. Next: bash %s/run_rtt_iclr2027.sh smoke\n' "$RTT_CODE_DIR"
