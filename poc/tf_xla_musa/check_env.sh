#!/usr/bin/env bash
set -euo pipefail

echo "== Basic =="
date
hostname
pwd

echo
echo "== Python =="
python3 -V || true
python3 -c "import tensorflow as tf; print('tensorflow', tf.__version__)" || true

echo
echo "== MUSA env =="
echo "MUSA_HOME=${MUSA_HOME:-}"
echo "MUSA_PATH=${MUSA_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

echo
echo "== MUSA binaries =="
which mthreads-smi || true
which musa-smi || true
which mcc || true
which llvm-link || true
which opt || true
which llc || true
which ld.lld || true

echo
echo "== MUSA toolkit files =="
ls -l /usr/local/musa 2>/dev/null | sed -n '1,20p' || true
ls -l /usr/local/musa/include/musa.h 2>/dev/null || true
ls -l /usr/local/musa/bin/mcc 2>/dev/null || true

echo
echo "== MUSA libs =="
ls -l /usr/local/musa/lib 2>/dev/null | egrep 'libmusa|libmusart|libmublas|libmufft|libmusparse|libmudnn' || true

echo
echo "== Device list from TF =="
python3 - <<'PY'
import tensorflow as tf
print(tf.config.list_physical_devices())
PY
