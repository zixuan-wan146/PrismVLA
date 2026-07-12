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
├── cache_indices/
├── token_caches/
├── checkpoints/
├── eval/
└── runs/
```

The model server runs in the model environment. Benchmark clients run in their own
Python environments and communicate through the transport-neutral policy client;
WebSocket is the default cross-process transport.

The wire protocol follows StarVLA's deployment boundary: MessagePack envelopes,
native NumPy arrays encoded as dtype/shape/raw bytes, a metadata handshake, and
structured request IDs and errors. Images must remain contiguous `uint8` arrays;
do not convert them to JSON lists. WebSocket compression and message-size limits
are disabled because image bytes are already compact and request validation occurs
after MessagePack decoding.

Start a trained Prism policy server:

```bash
../miniforge3/envs/Evo1/bin/python -m prism.serve.server \
  --ckpt_dir local_data/checkpoints/POLICY_NAME \
  --host 127.0.0.1 \
  --port 9000 \
  --vlm_local_files_only
```

Run a benchmark client from the repository root:

```bash
scripts/eval_benchmark.sh libero configs/experiment/libero_eval.yaml
scripts/eval_benchmark.sh calvin configs/experiment/calvin_eval.yaml
```

Use `PRISM_LIBERO_PYTHON` or `PRISM_CALVIN_PYTHON` to override the default
sibling environment. Use `PRISM_SERVER_URI` to connect to a server other than
`ws://127.0.0.1:9000`.

The smoke profiles only validate configuration:

```bash
scripts/eval_benchmark.sh libero configs/experiment/libero_smoke.yaml
scripts/eval_benchmark.sh calvin configs/experiment/calvin_smoke.yaml
```

A real runtime smoke test must additionally create the simulator, reset a task,
serialize both camera views and state, exchange one policy request, and execute
at least one environment step.
