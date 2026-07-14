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
the shared wire protocol, client, and history-precompute state machine live in
`prism/serve`.

The wire protocol follows StarVLA's deployment boundary: MessagePack envelopes,
native NumPy arrays encoded as dtype/shape/raw bytes, a metadata handshake, and
structured request IDs and errors. Protocol v2 has three request types:
`reset_history`, `push_history_observation`, and `infer`. A background response
router matches generic result envelopes by request ID and request type, so history
work can be outstanding while the simulator executes later actions. Images must
use the `uint8` dtype; non-contiguous views are materialized contiguously, while
float, bool, integer arrays with another dtype, and JSON lists are rejected
instead of being silently quantized.
WebSocket compression is disabled because image bytes are already compact.
Both ends enforce a 16 MiB message limit by default, before MessagePack request
validation. Set the client value with
`PRISM_POLICY_MAX_MESSAGE_SIZE_BYTES` and the server value with
`--max-message-size-bytes`.

An `infer` request carries the two ordered current views, prompt, state,
`stream_id`, `memory_generation`, and the previous cycle's eight canonical executed
actions plus validity mask; it never carries historical images. History is
precomputed during the interval between policy decisions:

1. Every episode or CALVIN subtask starts with `reset_history(stream_id)`.
2. Generation 0 inference creates invalid zero `[1, 16, 512]` memory directly. It
   does not encode placeholder images.
3. Immediately after local action-chunk offsets 2 and 5, the client sends a transient
   two-camera `push_history_observation` for slots 0 and 1 of the next generation.
4. The server applies image preprocessing and the shared vision encoder immediately.
   At the square baseline, it retains one `[128, 1024]` visual-token tensor per slot,
   not the source images.
5. When slot 1 arrives, the server runs the Q-Former with ages `[6, 3]`, retains only
   the resulting `[1, 16, 512]` memory and mask, and releases both visual-token slots.
6. The next `infer` waits for both background push acknowledgements and consumes the
   already prepared memory. Successful inference releases that generation's memory.

“No history images are saved” means images are neither cached nor carried across a
replan boundary. Each pushed observation is still transmitted briefly so the model
server can run its image processor and vision encoder. Moving that encoder into the
simulator process would duplicate the model environment and is not the accepted
deployment. Evaluation video frames are a separate, optional artifact path.

History state is bounded and local to one WebSocket connection: at most two encoded
visual observations or one final memory are retained. A stream reset drops partial
or ready tokens, and generation checks reject missing, duplicate, stale, or
cross-stream observations. Explicit CALVIN subtask resets are required because task
success can be detected after an O2/O5 observation was already pushed.

The same connection owns the task-state runtime: eight width-512 state tokens and
independent Mamba convolution/SSM caches for each token. `reset_history` resets both
history memory and planning state. The server commits advanced task/Mamba state only
after inference and generation bookkeeping both succeed, so retrying a failed
generation sees the previous state. The 16 width-512 plan tokens are returned only
in debug output today; downstream Bridge consumption and their training objectives
are intentionally deferred.

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
The client records those clipped motion values together with the thresholded
canonical `open_01` gripper as the next request's executed-action history. Initial
requests use eight zero rows with an all-false validity mask; dummy actions are not
fed to the updater.

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

The server has no authentication and refuses non-loopback bind addresses by
default. Keep it on `127.0.0.1` and use SSH port forwarding when the client is
on another host:

~~~bash
ssh -L 9000:127.0.0.1:9000 user@model-server
~~~

`--allow-non-loopback` is an explicit acknowledgement for a trusted private
network; it does not add authentication or encryption.

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

Values in `profile_env` are defaults. Existing shell environment variables
take precedence, so an exported camera resolution or result path is not
silently replaced by YAML. Dotted CLI overrides such as
`--overrides profile_env.PRISM_LIBERO_CAMERA_RESOLUTION=320` change an
individual profile default.

Policy connections and the initial metadata handshake are bounded by
`PRISM_POLICY_CONNECT_TIMEOUT_SECONDS` (default 30 seconds). Every inference
round trip is bounded by `PRISM_POLICY_INFERENCE_TIMEOUT_SECONDS` (default 120
seconds). The same bound covers reset, inference, and waiting for history-precompute
acknowledgements. A timeout is a fatal infrastructure failure: evaluation aborts and
the client does not silently count it as task failure or reconnect with hidden
state.

LIBERO camera resolution and video rate are explicit run settings:
`PRISM_LIBERO_CAMERA_RESOLUTION` defaults to 448 and
`PRISM_LIBERO_VIDEO_FPS` defaults to 30. The resolved values, including both
timeouts, are stored in each result summary. Summary updates reuse metadata
created once at evaluation start and publish through an fsynced temporary file
plus atomic replacement, so an interrupted update cannot truncate the previous
valid JSON result.

The smoke profiles only validate configuration:

```bash
experiments/libero/run_eval.sh experiments/libero/configs/smoke.yaml
experiments/calvin/run_eval.sh experiments/calvin/configs/smoke.yaml
```

A real runtime smoke test must additionally create the simulator, reset a task,
serialize both camera views and state, exchange one policy request, and execute
at least one environment step.

The repository includes opt-in runtime integration tests that go further: each
test runs nine control steps from two eight-action policy responses, verifies that
offset-2 and offset-5 observations are pushed for generation 1, verifies that both
inference requests contain current images only, round-trips all payloads through
MessagePack, and asserts that the control-step budget is not mistaken for a
planning-step budget. Run them in their simulator environments:

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
