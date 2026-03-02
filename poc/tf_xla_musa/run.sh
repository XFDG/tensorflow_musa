#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[1/4] Environment check"
bash "${ROOT_DIR}/check_env.sh" | tee "${LOG_DIR}/env_check.log"

echo "[2/4] Run baseline (XLA OFF)"
TF_CPP_MIN_LOG_LEVEL=0 \
USE_XLA=0 \
python3 "${ROOT_DIR}/tf_xla_smoke.py" 2>&1 | tee "${LOG_DIR}/run_no_xla.log"

echo "[3/4] Run XLA (XLA ON)"
TF_CPP_MIN_LOG_LEVEL=0 \
USE_XLA=1 \
python3 "${ROOT_DIR}/tf_xla_smoke.py" 2>&1 | tee "${LOG_DIR}/run_xla.log"

echo "[4/4] Extract XLA evidence lines"
grep -Eai "xla|jit|compiled cluster" "${LOG_DIR}/run_xla.log" | sed -n '1,80p' | tee "${LOG_DIR}/xla_evidence.log" || true

echo
echo "Done. Logs in: ${LOG_DIR}"
echo "  - env_check.log"
echo "  - run_no_xla.log"
echo "  - run_xla.log"
echo "  - xla_evidence.log"
