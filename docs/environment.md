# Remote `prsim` environment

All executable work runs on the remote data disk. From the repository root, create or refresh the environment with:

```bash
bash scripts/setup_env.sh
```

The script creates the environment at the sibling data-disk path `../envs/prsim`, keeps Conda, pip, Hugging Face, Transformers, and temporary caches on the data disk, and installs the repository in editable mode with the `data` and `dev` extras. The data extra declares the pandas/Parquet/HDF5/video dependencies used by the retained LIBERO and CALVIN readers instead of relying on packages inherited from an older environment.

Activate it explicitly with:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ../envs/prsim
```

The accepted core versions are defined by `pyproject.toml`:

```text
Python       3.10
PyTorch      2.5.1
torchvision  0.20.1
Transformers 5.13.1
NumPy        1.26.4
```

Before model downloads, export data-disk cache paths when running commands outside the setup script:

```bash
export HF_HOME=../hf-home
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PIP_CACHE_DIR=../pip-cache
export TMPDIR=../tmp
```

When the accepted backbone is already cached and the server has no overseas
network route, force offline resolution so Hugging Face does not spend time on
remote HEAD retries:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Do not enable those variables while initially populating the cache.

Use the official Hugging Face endpoint by default. If it is unstable, set `HF_ENDPOINT=https://hf-mirror.com` for that command. The server network acceleration script may be sourced before overseas downloads, but the pip index should remain explicit:

```bash
source /etc/network_turbo
PRSIM_PIP_INDEX_URL=https://pypi.org/simple bash scripts/setup_env.sh
```

The benchmark simulator environments remain separate. `prsim` owns model training, model inference, protocol tests, and the policy server; LIBERO and CALVIN clients run in their benchmark-specific environments through the MessagePack/WebSocket boundary.

Download the accepted backbone into the data-disk cache and run the opt-in real-checkpoint test with:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3.5-0.8B", cache_dir="../hf-home/hub")
PY

PRISM_RUN_MODEL_INTEGRATION=1 pytest -q \
  tests/models/test_qwen35_checkpoint_integration.py
```

This test loads the physically truncated 16-block model on CUDA, verifies the
real processor grids and parameter count, runs the mixed-bfloat16/float32
query-memory encoder, and checks square, padded-batch, and non-square inputs.
