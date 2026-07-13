# Benchmark runtime layout

PrismVLA keeps model inference, simulator runtimes, and generated artifacts separate.

From the repository root, the expected data-disk layout is:

```text
../benchmarks/
├── libero/
│   ├── assets/
│   └── datasets/
└── calvin/
    ├── runtime/
    ├── lerobot/
    ├── raw/
    └── annotations/
../envs/
├── libero/
└── calvin/
local_data/
├── checkpoints/
├── eval/
└── runs/
```

The model server runs in the model environment. Benchmark clients run in their own
Python environments and communicate through the transport-neutral policy client;
WebSocket is the default cross-process transport. Benchmark-specific code lives in
`experiments/libero` and `experiments/calvin`. Each benchmark keeps parameter parsing
in `config.py` and the complete simulator/request/rollout/result flow in `eval.py`; only
the shared wire protocol, client, and sparse-history contract live in `prism/serve`.

The wire protocol follows StarVLA's deployment boundary: MessagePack envelopes,
native NumPy arrays encoded as dtype/shape/raw bytes, a metadata handshake, and
structured request IDs and errors. Images must remain contiguous `uint8` arrays;
do not convert them to JSON lists. WebSocket compression and message-size limits
are disabled because image bytes are already compact and request validation occurs
after MessagePack decoding.

Every policy request carries the two ordered current views plus explicit sparse
history. History is keyed by the same ordered view names, with two images per view,
relative ages `[6, 3]`, and a two-element validity mask. The benchmark client captures
local action-chunk offsets 2 and 5 and sends them at the next 8-step replan boundary.
The server remains episode-stateless; initial requests use zero-valued history slots
with both validity entries false.

LIBERO episode limits are control-step budgets, not policy-request counts. The default
budgets are 220, 280, 300, and 520 environment steps for `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`, respectively. Open-loop replanning
does not reduce these benchmark limits.

The direct policy backend returns eight seven-dimensional actions in parallel. Its
seventh value is a continuous canonical gripper prediction where `0` means closed
and `1` means open. Both clients apply the strict predicate `prediction > 0.5`, so a
prediction equal to 0.5 is closed. LIBERO maps the decoded value with
`1 - 2 * open` (`+1=close, -1=open`); CALVIN maps it with `2 * open - 1`
(`-1=close, +1=open`). There is no `PRISM_CALVIN_GRIPPER_MODE` switch and no action
autoencoder in the accepted inference path. After statistical
de-normalization, each benchmark adapter explicitly clips the first six
relative motion dimensions to its verified `[-1, 1]` controller input bounds.
The q01/q99 normalization range is not treated as a physical safety bound.

The server accepts the model-agnostic `PolicyBackend` protocol and includes a
checkpoint-aware direct implementation, `CheckpointPolicyBackend`. It verifies the
checkpoint manifest plus architecture, DataSpec, statistics, dataset, robot, and
request contracts; normalizes raw canonical state with checkpoint-embedded
statistics; reconstructs the policy graph from the embedded architecture;
strictly restores the `Accelerator.save_state` model artifact from the same
verified checkpoint; and denormalizes the direct `[1, 8, 7]` result. The
advanced `from_loaded_policy` constructor remains available for tests and
externally managed deployments, but the normal server path does not accept
injected weights.

Start the checkpoint-backed server in the model environment:

```bash
python scripts/serve_policy.py \
  --checkpoint ../outputs/prismvla/<run>/checkpoints/step-00010000 \
  --device cuda \
  --host 127.0.0.1 \
  --port 9000 \
  --local-files-only
```

Run a benchmark client from the repository root. Each script uses its benchmark's
`configs/eval.yaml` profile by default:

```bash
experiments/libero/run_eval.sh
experiments/calvin/run_eval.sh
```

Pass a profile path as the first argument when using another configuration:

```bash
experiments/libero/run_eval.sh experiments/libero/configs/eval.yaml
experiments/calvin/run_eval.sh experiments/calvin/configs/eval.yaml
```

Use `PRISM_LIBERO_PYTHON` or `PRISM_CALVIN_PYTHON` to override the default
sibling environment. Use `PRISM_SERVER_URI` to connect to a server other than
`ws://127.0.0.1:9000`.

The smoke profiles only validate configuration:

```bash
experiments/libero/run_eval.sh experiments/libero/configs/smoke.yaml
experiments/calvin/run_eval.sh experiments/calvin/configs/smoke.yaml
```

A real runtime smoke test must additionally create the simulator, reset a task,
serialize both camera views and state, exchange one policy request, and execute
at least one environment step.

The repository includes opt-in runtime integration tests that go further: each
test runs nine control steps from two eight-action policy responses, verifies the
second request contains the offset-2 and offset-5 frames, round-trips the full
request through MessagePack, and asserts that the control-step budget is not
mistaken for a planning-step budget. Run them in their simulator environments:

```bash
../envs/libero/bin/python -m pip install pytest==8.3.5
../envs/calvin/bin/python -m pip install pytest==8.3.5
```

```bash
PRISM_RUN_LIBERO_INTEGRATION=1 \
  ../envs/libero/bin/python -m pytest -q \
  tests/eval/test_libero_runtime_integration.py

PRISM_RUN_CALVIN_INTEGRATION=1 \
  PYTHONPATH=.:../benchmarks/calvin/runtime/calvin_env:../benchmarks/calvin/runtime/calvin_models \
  ../envs/calvin/bin/python -m pytest -q \
  tests/eval/test_calvin_runtime_integration.py
```
