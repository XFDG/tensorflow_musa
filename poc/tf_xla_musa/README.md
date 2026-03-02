# TensorFlow + XLA + MUSA PoC

This folder provides a minimal reproducible PoC to verify:

1. TensorFlow can run in your MUSA server environment.
2. XLA path can be enabled with `jit_compile=True`.
3. You can collect logs for evidence (`XLA off` vs `XLA on`).

## Files

- `tf_xla_smoke.py`: Minimal TensorFlow training-step smoke test.
- `check_env.sh`: Environment and toolkit checks.
- `run.sh`: Runs `XLA off` and `XLA on`, saves logs.
- `logs/`: Generated at runtime.

## Prerequisites

- Python 3.10+ with TensorFlow installed in the runtime environment.
- MUSA toolkit and runtime visible in the server/container.

## Run

```bash
cd poc/tf_xla_musa
bash run.sh
```

## Output logs

- `logs/env_check.log`
- `logs/run_no_xla.log`
- `logs/run_xla.log`
- `logs/xla_evidence.log`

## Optional env vars

- `USE_XLA=0/1` (set by `run.sh`)
- `STEPS` (default `50`)
- `WARMUP` (default `5`)
