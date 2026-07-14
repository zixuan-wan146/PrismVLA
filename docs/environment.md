# Remote `prsim` environment

All executable work runs on the remote data disk. From the repository root, create or refresh the environment with:

```bash
bash scripts/setup_env.sh
```

The script creates the environment at the sibling data-disk path `../envs/prsim`, keeps Conda, pip, Hugging Face, Transformers, Triton, TorchInductor, and temporary caches on the data disk, and installs the repository in editable mode with the `data` and `dev` extras. The data extra declares the pandas/Parquet/HDF5/video dependencies used by the retained LIBERO and CALVIN readers instead of relying on packages inherited from an older environment.

The setup also installs the two CUDA components required by the Transformers
Qwen3.5 fast path:

```text
fla-core       0.3.2
causal-conv1d  1.5.0.post8+cu12torch2.5cxx11abiFALSE
```

These versions are intentional compatibility pins. `fla-core==0.3.2` provides
the gated-delta-rule and fused gated RMSNorm kernels used by Qwen3.5 while
remaining compatible with the accepted PyTorch 2.5.1 / Triton 3.1 stack. The
causal-conv1d artifact is the official prebuilt CPython 3.10 Linux x86-64 wheel,
so the server does not need `nvcc`. The setup installs both with `--no-deps`
after the project dependencies to prevent kernel packages from replacing the
accepted PyTorch stack. The higher-level `flash-linear-attention` package is not
needed: Qwen imports its kernels from `fla-core` directly.

To install from a wheel already stored on the data disk, override the causal
convolution requirement without changing the script:

```bash
PRSIM_CAUSAL_CONV1D_REQUIREMENT=../wheels/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  bash scripts/setup_env.sh
```

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
export TRITON_CACHE_DIR=../triton-cache
export TORCHINDUCTOR_CACHE_DIR=../torchinductor-cache
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

Verify that Transformers selected the Qwen3.5 fast path with:

```bash
python - <<'PY'
from transformers.models.qwen3_5 import modeling_qwen3_5

assert modeling_qwen3_5.is_fast_path_available
print("Qwen3.5 fast path is available")
PY
```

Use the official Hugging Face endpoint by default. If it is unstable, set `HF_ENDPOINT=https://hf-mirror.com` for that command. The server network acceleration script may be sourced before overseas downloads, but the pip index should remain explicit:

```bash
source /etc/network_turbo
PRSIM_PIP_INDEX_URL=https://pypi.org/simple bash scripts/setup_env.sh
```

The benchmark simulator environments remain separate. `prsim` owns model training, model inference, protocol tests, and the policy server; LIBERO and CALVIN clients run in their benchmark-specific environments through the MessagePack/WebSocket boundary.

Create or converge those benchmark environments from their verified,
fully-pinned Linux/Python 3.8 lock files:

~~~bash
bash scripts/setup_libero_eval_env.sh
bash scripts/setup_calvin_eval_env.sh
~~~

The scripts place environments at ../envs/libero and ../envs/calvin, pip
caches at ../pip-cache, and temporary files at ../tmp. Override them with
PRISM_LIBERO_ENV_PREFIX, PRISM_CALVIN_ENV_PREFIX, PIP_CACHE_DIR, and TMPDIR.
PRISM_ENABLE_NETWORK_TURBO=1 opts into /etc/network_turbo;
PRISM_PIP_INDEX_URL selects the pip index. PRISM_CONDA_BIN selects Conda, and
PRISM_CALVIN_ROOT selects the pinned CALVIN source checkout.

requirements/libero-eval.lock.txt reproduces the verified LIBERO 0.1.1 /
robosuite 1.4.0 / MuJoCo 3.2.3 environment.
requirements/calvin-eval.lock.txt is paired with CALVIN commit
fa03f01f19c65920e18cf37398a9ce859274af76 and its calvin_env submodule
commit 1431a46bd36bde5903fb6345e68b5ccc30def666. The setup script clones those
sources on the data disk and refuses mismatched or tracked-dirty checkouts.
Simulator dependencies intentionally do not appear as project extras because
their Python/Torch stacks conflict with the Python 3.10 model environment.

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
