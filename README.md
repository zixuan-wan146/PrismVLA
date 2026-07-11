# PrismVLA

PrismVLA is the reconstructed package form of the active bridge-VLA code path.

Target package:

```python
import prism
```


## Canonical Commands

```bash
python scripts/build_cache.py --config configs/experiment/libero_stage1.yaml
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
```

All scripts are thin entry points. Model, data, training, evaluation, and serving logic lives under `prism/`.
