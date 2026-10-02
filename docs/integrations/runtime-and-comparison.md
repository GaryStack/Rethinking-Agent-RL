# Runtime preflight and checkpoint comparison

Both utilities work independently of the training recipe. Reports are local JSON;
outputs are immutable and no data is uploaded. Use the repository with the core
installed, or set `PYTHONPATH=src`.

## Check facilities before model loading

```bash
python scripts/check_runtime.py --python /absolute/inference-env/bin/python \
  --cuda-device 0 --timeout 15 --output runs/runtime-gpu0.json
# Opt in when bounded coding execution is needed:
python scripts/check_runtime.py --python /absolute/inference-env/bin/python \
  --namespace --timeout 20 --output runs/runtime-coding.json
```

The selected interpreter retains its venv path. Package metadata inspection does
not import frameworks. CUDA checks call the actual driver, then allocate a
16-element Torch tensor on the explicitly selected GPU. Driver failure skips the
tensor check. Namespace checks launch the existing bounded executor and a trusted
`print(6*7)` program. No model loading, inference or training occurs.

Each subprocess records status, PID, exit code, elapsed time and bounded output.
Timeouts kill only the probe's process group. A blocked kernel call that cannot
be reaped is recorded without waiting indefinitely. Missing dependencies and
startup timeouts remain infrastructure results. They are not model task scores.
Any requested probe failing returns exit code 1; choose a new output for a retry.

`requested_checks_passed` covers only the requested checks. CPU metadata success
does not establish GPU readiness, and a small GPU allocation does not establish
Ray/vLLM/FSDP compatibility. Namespace startup is not a replacement for the
path/network/resource isolation tests. Use the pinned import preflight and real
training acceptance for their respective claims.

## Compare independently executed checkpoint evaluations

Run the public checkpoint CLI with the same export/config/pool/seeds, then:

```bash
python scripts/compare_checkpoint_validation.py \
  --left runs/ray-smoke/diagnostics/validation-1 \
  --right runs/standalone-step1 --pool runs/ray-smoke/pool \
  --output runs/checkpoint-parity.json
# Add --verify-weights when the original HF exports remain available.
```

Both inputs first pass the shared artifact, state-replay and metric-recomputation
audit. Corrupt artifacts and incomplete points are rejected. The complete point
is compared, including checkpoint step/identity and config/pool identity. SHA256
of multi-turn trajectories and single-decision actions must also match. Comparing
a directory with itself is rejected; identical scores at different checkpoint
steps cannot pass.

Matching inputs return 0. Valid but different inputs produce the differences and
return 1. Invalid inputs raise an audit error without writing a pass report. There
is no tolerance setting that hides raw differences. Default comparison needs no
GPU, model weights or optional framework and works on copied raw artifacts.

The operator must execute the jobs independently; directory comparison cannot
prove that files were not copied. Evaluation output parity does not establish
full-trainer optimizer/state/RNG parity. Those remain separate P3 checks.
