# PrismVLA — Restructure Design & Execution Plan

**Source:** `Prism-Bridge-VLA` (current)
**Target:** `PrismVLA` (new repo, import name `prism`)
**Purpose:** Hand-off document for an implementing agent (codex). Everything below is prescriptive: layout, contracts, migration mapping, execution phases, and acceptance criteria.

**Benchmark correction:** RMbench is no longer part of the PrismVLA target. The target benchmarks are LIBERO and CALVIN. Existing RMbench code, configs, scripts, and docs should be treated as abandoned unless the user explicitly requests a separate RMbench restoration later.

---

## 1. Design principles

The current repo suffers from three things: (a) a `cli/` subpackage that mirrors `scripts/` (47 + 48 files doing the same work), (b) `training/{stage1,stage2}/{common,libero}/` over-nesting for ~26 files, and (c) ~15 flat docs at one level with stale cross-references. The restructure applies four rules:

1. **Package at repo root, no `src/` wrapper.** Follows the Isaac-GR00T convention: `prism/` sits at the top level so `import prism.models.policy` works with a one-line `pyproject.toml` package spec. No dual-tree confusion.
2. **Single canonical entry-point tree.** Business logic lives in `prism/`. `scripts/` is thin dispatchers only (parse args, load config, call into `prism`). No `cli/` subpackage.
3. **≤ 8 single-word subpackages, ≤ 2 levels of nesting** inside `prism/`. Anything deeper is a smell.
4. **One config file until it hurts.** Start with `prism/config.py` as a dataclasses module. Only split into `prism/config/{model,data,training}.py` when it exceeds ~1500 lines.

Reference repos surveyed: openpi (Physical Intelligence, π₀/π₀.₅), Isaac-GR00T (NVIDIA GR00T N1.7), openvla, Vlaser (ICLR 2026), and **StarVLA** (April 2026 paper, "Lego-like codebase" for VLA development). All ship 5–8 single-word subpackages with flat entry points. StarVLA in particular is the primary structural exemplar for the "high cohesion / low coupling" target and is already the LIBERO training-recipe reference cited throughout the current `docs/` — see §7.7 for how it is handled.

---

## 2. Naming

| Item | Value |
|---|---|
| Repo folder | `PrismVLA/` |
| Python package | `prism` |
| Import path | `import prism.<subpkg>.<module>` |
| PyPI name (if published) | `prismvla` |
| Entry-point CLI (optional) | `prism-train`, `prism-eval`, `prism-serve` |

---

## 3. Target directory layout

```
PrismVLA/
├── README.md
├── AGENTS.md                    # kept at root; coding agents scan for it here
├── LICENSE
├── pyproject.toml               # single source for deps + extras
├── uv.lock                      # or requirements.lock
├── .gitignore
│
├── prism/                       # THE package. import prism.<x>
│   ├── __init__.py
│   ├── config.py                # ALL config dataclasses live here
│   │
│   ├── models/                  # model building blocks
│   │   ├── __init__.py
│   │   ├── vlm.py               # InternVL3 embedder wrapper
│   │   ├── memory.py            # short visual-token memory (S_t)
│   │   ├── planner.py           # progress-state planner + updater (M_t, P_t)
│   │   ├── action_head.py       # direct bridge-attention + flow-matching head
│   │   └── policy.py            # top-level assembly, entry point for training/inference
│   │
│   ├── data/                    # datasets, loaders, caches
│   │   ├── __init__.py
│   │   ├── libero.py            # LIBERO episode reader + warmup dataset
│   │   ├── calvin.py            # CALVIN episode reader + warmup dataset
│   │   ├── cache.py             # token-cache read/write, shard IO
│   │   └── segments.py          # action-segment slicing, masks
│   │
│   ├── training/                # training loops (flat, no stage subdirs)
│   │   ├── __init__.py
│   │   ├── trainer.py           # single generic trainer; stage/dataset from config
│   │   ├── warmup.py            # progress-state warmup loop (separate)
│   │   └── loss.py              # flow-matching + plan/state/mem-pool losses
│   │
│   ├── eval/                    # benchmark runners
│   │   ├── __init__.py
│   │   ├── libero.py            # LIBERO runner + obs/action adapter
│   │   ├── calvin.py            # CALVIN runner + adapter
│   │   └── runner.py            # shared BenchmarkRunner base
│   │
│   ├── serve/                   # inference / runtime
│   │   ├── __init__.py
│   │   ├── engine.py            # inference engine, feature extractor
│   │   └── server.py            # websocket server + wire protocol
│   │
│   └── utils/                   # genuinely shared helpers ONLY
│       ├── __init__.py
│       ├── paths.py             # AUTODL_TMP resolution, run-dir helpers
│       ├── seeding.py           # deterministic seed setup
│       └── logging.py           # logger config
│
├── configs/                     # YAML experiment configs (Hydra/OmegaConf-style)
│   ├── model/
│   │   └── prism_base.yaml
│   ├── data/
│   │   ├── libero.yaml
│   │   └── calvin.yaml
│   └── experiment/
│       ├── libero_warmup_w4.yaml
│       ├── libero_stage1.yaml
│       ├── calvin_warmup_w4.yaml
│       ├── calvin_stage1.yaml
│       ├── calvin_smoke.yaml
│       └── libero_smoke.yaml
│
├── scripts/                     # thin entry points, NO business logic
│   ├── train.py                 # python scripts/train.py --config configs/experiment/X.yaml
│   ├── warmup.py                # progress warmup entry
│   ├── eval.py                  # LIBERO / CALVIN eval entry
│   ├── build_cache.py           # token-cache builder
│   ├── serve.py                 # websocket server launcher
│   └── check.sh                 # ruff + pytest wrapper (repo gate)
│
├── examples/                    # concrete end-to-end walkthroughs
│   └── libero_quickstart.md
│
├── tests/                       # mirrors prism/ layout, no deeper nesting
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── eval/
│   └── serve/
│
├── docs/                        # grouped, not flat
│   ├── design/                  # how it works
│   │   ├── memory_and_planner.md
│   │   ├── action_head.md
│   │   └── bridge_himem.md
│   ├── training/
│   │   └── libero_recipe.md
│   ├── benchmarks/
│   │   ├── benchmark_plan.md
│   │   └── benchmark_contracts.md
│   ├── research/                # surveys, external notes
│   │   ├── vla_survey_2026h1.md
│   │   └── vla_adapter_notes.md
│   └── engineering.md           # merges engineering_reproducibility + project_structure
│
└── third_party/                 # reference material, single home
    ├── papers/                  # was reference-paper/ (87MB PDFs)
    └── vendored/                # was reference-repo/, properly vendored
        └── vla_adapter/
```

**Impact estimate:** ~470 files → ~200 files. Directory depth 4 → 2. Top-level entries shrink to the target root only; keep `reconstruct.md` at root while this migration remains active.

---

## 4. Module contracts

Each subpackage owns exactly one concern. If a change spans two, revisit the split before adding a helper module.

| Subpackage | Owns | Does NOT own |
|---|---|---|
| `prism.config` | All dataclass schemas, YAML → dataclass hydration, validation | Any tensor math, IO, or side effects |
| `prism.models` | Neural modules and their forward passes | Datasets, training loops, optimizers |
| `prism.data` | Reading raw episodes, building warmup caches, token-cache IO, action-segment ops | Model architecture, training state |
| `prism.training` | Optimizer, scheduler, gradient step, checkpointing, logging | Model internals, dataset internals |
| `prism.eval` | Benchmark runners, obs→model→action adapters | Training, serving |
| `prism.serve` | Inference engine, feature extractor, websocket protocol | Training, benchmark orchestration |
| `prism.utils` | Path resolution, seeding, logging setup | Anything that could reasonably live in a domain package |

**Rule:** a module in one subpackage may import from `prism.config` and `prism.utils` freely, but cross-subpackage imports (e.g. `prism.training` importing `prism.eval`) require justification.

---

## 5. Config strategy

**Two-layer split, no ambiguity:**

1. **`prism/config.py`** — Python dataclasses that define the *schema*. Field names, types, defaults, validation. Example:
   ```python
   @dataclass
   class ModelConfig:
       H: int = 32
       R: int = 16
       vlm_backbone: str = "internvl3_2b"
       plan_slots: int = 8
       ...

   @dataclass
   class TrainingConfig:
       stage: Literal["warmup", "stage1", "stage2"]
       batch_size: int
       lr: float
       ...

   @dataclass
   class PrismConfig:
       model: ModelConfig
       data: DataConfig
       training: TrainingConfig
       runtime: RuntimeConfig
   ```

2. **`configs/**/*.yaml`** — actual *values* per experiment. Loaded via `OmegaConf` or plain `yaml.safe_load` and validated against the dataclass schema on entry.

**Guardrail:** no `if os.environ.get(...)` reads inside model or training code. All environment coupling happens in `prism/utils/paths.py` and in the config-loading step of `scripts/*.py`.

---

## 6. Entry-point strategy

Every `scripts/*.py` follows the same 4-step template:

```python
# scripts/train.py
import argparse
from prism.config import load_config
from prism.training.trainer import Trainer
from prism.utils.seeding import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(cfg.runtime.seed)
    Trainer(cfg).run()

if __name__ == "__main__":
    main()
```

That's the whole file. No conditionals, no model imports, no dataset construction. Same shape for `warmup.py`, `eval.py`, `build_cache.py`, `serve.py`.

**Optional:** register these as `[project.scripts]` entries in `pyproject.toml` so they become `prism-train`, `prism-eval`, etc. — cosmetic, not required.

---

## 7. Migration mapping (current → PrismVLA)

### 7.1 `src/himem_bridge_vla/` → `prism/`

| Current path | New path | Notes |
|---|---|---|
| `model/himem_bridge_vla.py` | `prism/models/policy.py` | Rename class to `PrismPolicy` or similar |
| `model/action_head/` (all `.py`) | `prism/models/action_head.py` | Fold into one file; if > 1000 lines, keep as `action_head/` subpackage with `__init__.py` re-exporting the public API |
| `model/bridge/` | Merge into `prism/models/action_head.py` | Bridge attention is part of the head |
| `model/himem/` | `prism/models/memory.py` | Short visual-token memory only (long memory lives in `planner.py` per current design) |
| `model/planner/` | `prism/models/planner.py` | Progress-state updater + planner + condition builder + action-segment AE |
| `model/internvl3/` | `prism/models/vlm.py` | InternVL3 wrapper |
| `dataset/libero*.py` | `prism/data/libero.py` | Consolidate `libero.py` + `libero_progress_warmup.py` |
| `dataset/calvin*.py` | `prism/data/calvin.py` | Consolidate `calvin.py` + `calvin_progress_warmup.py`; RMbench is abandoned |
| `dataset/memory_token_cache.py`, `memory_replay*.py`, `cache_utils.py` | `prism/data/cache.py` | Merge the four related modules |
| `dataset/action_segments.py` | `prism/data/segments.py` | 1:1 rename |
| `dataset/simulation_dataset.py`, `validation.py`, `config_utils.py` | Fold into `prism/data/__init__.py` or the domain files they support | Do not create a `prism/data/misc.py` grab-bag |
| `training/stage1/**`, `training/stage2/**`, `training/common/**` | `prism/training/trainer.py` | Stage is a config field, not a directory. Extract truly shared helpers into a single `_common.py` only if needed |
| `training/progress_warmup.py` | `prism/training/warmup.py` | Separate loop, keep separate file |
| `training_loss.py` | `prism/training/loss.py` | 1:1 move |
| `benchmarks/libero/`, `benchmarks/calvin/`, `benchmarks/base.py` | `prism/eval/{libero,calvin,runner}.py` | Flatten one level; RMbench is abandoned |
| `evaluation/run_metadata.py`, `evaluation/__init__.py` | Merge into `prism/eval/runner.py` | The two-file `evaluation/` package is over-modularized |
| `runtime/contract.py`, `feature_extractor.py`, `inference_engine.py`, `memory_builder.py` | `prism/serve/engine.py` | One file; split only if > 1500 lines |
| `runtime/websocket_server.py`, `server_protocol.py` | `prism/serve/server.py` | Merge server + wire protocol |
| `bridge_himem_config.py`, `experiment_config.py`, `runtime_config.py`, `training_config.py` | `prism/config.py` | All four → one dataclass module |
| `core/{constants,errors,paths,registry,types}.py` | Distribute: constants/types stay reachable via `prism/__init__.py`, `paths.py` → `prism/utils/paths.py`, `errors.py` → drop or inline where raised, `registry.py` → keep only if truly used | The current 5-file `core/` package is mostly cosmetic |
| `path_utils.py`, `reproducibility.py`, `image_preprocessing.py` | `prism/utils/paths.py`, `prism/utils/seeding.py`, and move `image_preprocessing.py` into `prism/data/` since it's dataset-side | Do not keep loose modules at the package root |
| `cli/**` (47 files) | **DELETED** | Their logic already exists in `prism/`; scripts import from there |

### 7.2 `scripts/` — collapse from 48 files to 6

| Current | New |
|---|---|
| `scripts/train/**` (4 files) | `scripts/train.py` (single file, config-dispatched) |
| `scripts/cache/**` (11 files) | `scripts/build_cache.py` (single file, config-dispatched) |
| `scripts/eval/**` (11 files) | `scripts/eval.py` (single file, benchmark from config) |
| `scripts/serve/**` (2 files) | `scripts/serve.py` |
| `scripts/quality/**` (9 files) | `scripts/check.sh` + move config-validation logic into `prism/config.py` (raise on invalid load) |
| `scripts/setup/**` (4 files) | Keep only `scripts/setup_libero.sh` if genuinely needed; move rest to `docs/training/` as instructions |
| `scripts/report/**` (5 files), `automation/**` (1), `maintenance/**` (1) | Drop unless actively used; if kept, move to `tools/` (separate from `scripts/`) |

### 7.3 Configs

The current `configs/{datasets,experiments,models,runtime,training}/` (5 subdirs) becomes `configs/{model,data,experiment}/` (3 subdirs). Runtime-profile YAMLs (`configs/runtime/libero_profiles/` and `configs/runtime/calvin_profiles/`) fold into `configs/experiment/` since a "runtime profile" is just an experiment variant. Do not migrate RMbench configs into PrismVLA.

### 7.4 Docs

15 flat files in `docs/` regroup as:

| Current | New location |
|---|---|
| `bridge_himem_design.md`, `direct_bridge_attention_design_zh.md`, `progress_state_planner_design_zh.md`, `vla_adapter_bridge_attention_notes_zh.md` | `docs/design/` |
| `libero_direct_bridge_training_recipe_zh.md`, `starvla_libero_training_recipe_research_zh.md` | `docs/training/` |
| `benchmark_plan.md`, `benchmark_research_report_2026-06-23.md`, `benchmark_contracts/**` | `docs/benchmarks/` |
| `vla_training_recipe_survey_2026_h1.md` | `docs/research/` |
| `engineering_reproducibility.md`, `project_structure.md`, `architecture/**`, `engineering/**` | Merge into `docs/engineering.md` (single file) |
| `current_project_state.md` | Delete after migration completes (state lives in git history + README status section) |
| Root `Plan.md`, `Structure.md` | Delete; salvage relevant bits into `README.md` and `docs/engineering.md` |

RMbench-specific benchmark contracts and research notes should not be migrated.
If benchmark documentation is needed for the second benchmark, create or keep
CALVIN-focused docs under `docs/benchmarks/`.

### 7.5 Third-party

| Current | New |
|---|---|
| `reference-paper/*.pdf` (87MB) | `third_party/papers/` |
| `reference-repo/vla_adapter/` | `third_party/vendored/vla_adapter/` |
| `reference-repo/starVLA` (dangling gitlink, no `.gitmodules`) | **Vendor as pinned source snapshot** at `third_party/vendored/starvla/`. See §7.7 below — this one gets its own treatment because 24 doc citations depend on it and the current state has been silently broken for every clone. |
| `evaluations/legacy/` | `third_party/legacy_eval/` if still needed, otherwise **delete** |

### 7.7 StarVLA — dedicated treatment

StarVLA (`github.com/starVLA/starVLA`) is not a passive reference in this project. It is:

1. The **primary training-recipe reference** for LIBERO in this repo. `docs/starvla_libero_training_recipe_research_zh.md` cites 8 specific files from it and analyzes its trainer, dataloader, DeepSpeed config, and gradient-accumulation semantics.
2. A **structural exemplar** for the "Lego-like, high cohesion / low coupling" design PrismVLA is targeting. Its own layout (top-level `starVLA/` package, `examples/<benchmark>/` for recipes, benchmark-specific `train_files/`) validated most of the choices in §3.
3. **Silently broken** in the current repo: committed as a git gitlink (mode `160000`, commit `6dc01d0`) with no `.gitmodules` file, so every clone lands an empty folder and all 24 doc citations are dead paths.

**Decision: vendor as a pinned source-only snapshot.** Same treatment as `reference-repo/vla_adapter/` (which is currently a `040000 tree`, i.e. properly vendored — that's why it works). Do NOT re-add as a submodule; submodule tracking of an actively-evolving upstream is precisely how the current mess was created.

**Concrete steps during Phase 7:**

1. Remove the gitlink entry:
   ```bash
   git rm --cached reference-repo/starVLA
   ```
2. Clone upstream at the pinned commit into a relative scratch location:
   ```bash
   git clone https://github.com/starVLA/starVLA starvla-src
   cd starvla-src && git checkout 6dc01d0
   ```
3. Copy **only the paths referenced by docs** into `third_party/vendored/starvla/`:
   ```
   third_party/vendored/starvla/
   ├── SNAPSHOT.md                 # commit hash, upstream URL, date, license, "do not edit"
   ├── examples/LIBERO/
   │   ├── README.md
   │   ├── data_preparation.sh
   │   └── train_files/
   │       ├── run_libero_train.sh
   │       ├── starvla_cotrain_libero.yaml
   │       └── data_registry/data_config.py
   └── starVLA/
       ├── training/train_starvla.py
       ├── dataloader/lerobot_datasets.py
       └── config/deepseeds/{ds_config.yaml,deepspeed_zero2.yaml}
   ```
   Skip large binaries, checkpoints, media, and files that no doc references. Follow the same exclusion rules already in `.gitignore` for `reference-repo/**/*.{pdf,mp4,pkl,...}`.
4. Rewrite all 24 doc citations from `reference-repo/starVLA/...` → `third_party/vendored/starvla/...` (sed pass, one commit).
5. Write `third_party/vendored/starvla/SNAPSHOT.md`:
   ```markdown
   # StarVLA Snapshot

   - Upstream: https://github.com/starVLA/starVLA
   - Commit:   6dc01d0
   - Snapshot date: 2026-06-29 (matches docs/design/libero_recipe.md)
   - License: (copy upstream LICENSE header)
   - Purpose: reference-only. Do not edit. Refresh only via a full snapshot
     replacement, and update SNAPSHOT.md + docs together.
   ```
6. Add licensing note to the vendored tree if starVLA's license requires attribution (check upstream LICENSE — MIT would need retention).

**Refresh policy:** when the LIBERO recipe research needs to track a newer starVLA commit, do a **full replacement** of the vendored subtree in a single commit that bumps the commit hash in `SNAPSHOT.md` and updates any doc claims that changed. Never partial-update files, never re-add as a submodule.

**Acceptance for §7.7:** `find third_party/vendored/starvla -type f | wc -l` returns a small number (the ~8 doc-cited files plus SNAPSHOT.md), `grep -r "reference-repo/starVLA" .` returns nothing, all doc links resolve to actual files, and `git ls-tree HEAD | grep 160000` is empty.

### 7.8 Bonus pattern from StarVLA — `**/bar/` gitignored scratch dirs

StarVLA uses a nice convention: any directory named `bar/` anywhere in the tree is git-ignored, giving users a well-known place for local scratch scripts (e.g. `examples/LIBERO/train_files/bar/my_train.sh`) without polluting the repo. Adopt this: add `**/bar/` to `.gitignore` during Phase 0. Zero cost, one line, and it channels ad-hoc user scripts away from `scripts/` so the canonical entry-point tree stays clean.

### 7.6 Requirements

Consolidate `requirements.txt`, `requirements-dev.txt`, `requirements-libero.txt`, `requirements-policy.json` into `pyproject.toml` with extras:

```toml
[project]
name = "prismvla"
dependencies = [ ... base runtime deps ... ]

[project.optional-dependencies]
dev = [ ... ruff, pytest, ... ]
libero = [ ... libero-only deps ... ]
```

If `requirements-policy.json` is a policy-server config (not a pip file), move it to `configs/serve/policy.json`.

---

## 8. Execution phases (ordered for codex)

Each phase is a separate PR/commit. Do not merge phases. Tests must pass after each phase.

### Phase 0 — Scaffold
- Create empty `PrismVLA/` skeleton per §3.
- Create `prism/` and every subpackage with an empty `__init__.py`.
- Add `pyproject.toml` (name `prismvla`, package `prism`), `.gitignore` (copy from source), `README.md` stub, `AGENTS.md` (copy from source, edit paths).
- Add a placeholder `tests/test_import.py` that imports every subpackage.
- **Acceptance:** `pip install -e .` succeeds; `python -c "import prism.models, prism.data, prism.training, prism.eval, prism.serve, prism.utils, prism.config"` succeeds; `pytest` collects and passes.

### Phase 1 — Config unification
- Create `prism/config.py` with dataclasses covering the union of the four current `*_config.py` files.
- Add YAML loader (`load_config(path, overrides)`) with dataclass validation.
- Port one experiment YAML (`configs/experiment/libero_smoke.yaml`) to prove the loader.
- **Acceptance:** `python -c "from prism.config import load_config; c = load_config('configs/experiment/libero_smoke.yaml'); print(c)"` prints the hydrated config.

### Phase 2 — Data layer
- Move `dataset/libero*.py` → `prism/data/libero.py` (merge).
- Move `dataset/calvin*.py` → `prism/data/calvin.py` (merge).
- Do not migrate RMbench dataset modules.
- Move `memory_token_cache.py`, `memory_replay*.py`, `cache_utils.py` → `prism/data/cache.py` (merge).
- Move `action_segments.py` → `prism/data/segments.py`.
- Update all imports to `prism.data.*`.
- **Acceptance:** existing dataset-side tests pass unchanged after import rewrite.

### Phase 3 — Model layer
- Move `model/himem_bridge_vla.py` → `prism/models/policy.py`.
- Merge `model/action_head/**` + `model/bridge/**` → `prism/models/action_head.py`.
- Move `model/himem/**` → `prism/models/memory.py`.
- Move `model/planner/**` → `prism/models/planner.py`.
- Move `model/internvl3/**` → `prism/models/vlm.py`.
- Rewrite all internal imports.
- **Acceptance:** existing model-side tests pass; policy forward pass runs on a fixture batch.

### Phase 4 — Training layer
- Merge `training/stage1/**` + `training/stage2/**` + `training/common/**` into `prism/training/trainer.py`. Stage becomes a config field.
- Move `training/progress_warmup.py` → `prism/training/warmup.py`.
- Move `training_loss.py` → `prism/training/loss.py`.
- **Acceptance:** existing training tests pass; `python scripts/train.py --config configs/experiment/libero_smoke.yaml` runs one step.

### Phase 5 — Eval + serve layers
- Fold LIBERO/CALVIN benchmark modules + `evaluation/**` → `prism/eval/`.
- Do not migrate RMbench benchmark modules.
- Fold `runtime/**` + `server_protocol.py` → `prism/serve/`.
- **Acceptance:** existing eval tests pass; websocket server starts and answers a health check.

### Phase 6 — Scripts + CLI kill
- Write the 6 thin `scripts/*.py` files per §6.
- Delete `src/himem_bridge_vla/cli/` entirely.
- Delete `scripts/{cache,eval,quality,report,serve,setup,train,automation,maintenance}/` subtrees.
- **Acceptance:** all four canonical workflows (build cache → warmup → train → eval) run end-to-end from the new scripts.

### Phase 7 — Docs + third-party
- Regroup `docs/**` per §7.4.
- Move `reference-paper/` and `reference-repo/` under `third_party/`.
- Resolve the dangling `starVLA` submodule (vendor or delete).
- Delete `Structure.md`, `Plan.md`, `current_project_state.md`.
- **Acceptance:** no broken links in any `.md`; `git ls-tree HEAD | grep 160000` returns nothing (no dangling submodule gitlinks); starVLA lives at `third_party/vendored/starvla/` with a `SNAPSHOT.md` pinning commit `6dc01d0`, and every doc citation resolves to a real file.

### Phase 8 — Requirements + lint
- Fold requirements into `pyproject.toml` extras.
- Run `ruff check .` clean.
- Run full `pytest` clean.
- **Acceptance:** `scripts/check.sh` passes.

---

## 9. Guardrails (do NOT do these)

1. Do not recreate a `cli/` subpackage. Every workflow enters through a `scripts/*.py` shim.
2. Do not add a `common/` folder anywhere. If code is truly shared, put it in `prism/utils/` or make it a top-level file in the owning subpackage.
3. Do not split `prism/config.py` into a package until it actually exceeds ~1500 lines.
4. Do not introduce hardcoded paths. All environment coupling goes through `prism/utils/paths.py`.
5. Do not add `stage1/`, `stage2/`, `libero/`, `calvin/`, or `rmbench/` as directories inside `prism/training/`. Stage and dataset are config fields. RMbench is abandoned for this target.
6. Do not re-add a git submodule without a matching committed `.gitmodules` file.
7. Do not resurrect retired designs (H64 suffix planner, transition-trigger, Dual-FIFO, `PlanTokenQueue`). Cleanup was already correct in the current repo — preserve that.
8. Do not mix refactor commits with behavior changes. Each phase is either a move (imports rewritten, semantics identical) or a genuine change (with tests) — never both.

---

## 10. Acceptance checklist (end state)

- [ ] `import prism` and every subpackage import works from a fresh clone + `pip install -e .`.
- [ ] Repo root contains only target entries: `AGENTS.md`, `LICENSE`, `README.md`, `pyproject.toml`, `.gitignore`, `reconstruct.md`, `prism/`, `configs/`, `scripts/`, `examples/`, `tests/`, `docs/`, `third_party/`, plus an optional lockfile.
- [ ] `prism/` has exactly 6 subpackages: `models`, `data`, `training`, `eval`, `serve`, `utils`, plus `config.py`.
- [ ] No directory inside `prism/` nests deeper than 2 levels.
- [ ] `scripts/` contains ≤ 6 `.py` files plus `check.sh`.
- [ ] No `cli/` directory exists anywhere.
- [ ] No dangling git submodule (`git ls-tree HEAD | grep 160000` empty, or matching `.gitmodules` entry).
- [ ] `pyproject.toml` is the single dependency source; no `requirements-*.txt` files.
- [ ] `ruff check .` and `pytest` both green.
- [ ] Every YAML in `configs/experiment/` loads and validates against `prism.config`.
- [ ] The four canonical workflows run end-to-end from the new scripts:
  1. `scripts/build_cache.py --config configs/experiment/libero_stage1.yaml`
  2. `scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml`
  3. `scripts/train.py --config configs/experiment/libero_stage1.yaml`
  4. `scripts/eval.py --config configs/experiment/libero_smoke.yaml`
- [ ] CALVIN benchmark plumbing is available through the same canonical scripts:
  1. `scripts/build_cache.py --config configs/experiment/calvin_stage1.yaml`
  2. `scripts/warmup.py --config configs/experiment/calvin_warmup_w4.yaml`
  3. `scripts/eval.py --config configs/experiment/calvin_smoke.yaml`
- [ ] No RMbench source modules, configs, scripts, or docs remain in the PrismVLA target.

---

## 11. Notes for the implementing agent

- Follow the existing `AGENTS.md` rules (relative paths, no hardcoded absolute paths, config-not-code for parameters, tests after protocol changes). They already apply to the new repo without modification.
- When a merge of multiple current files into one new file is ambiguous, keep the new file split by clear top-of-file comment banners (`# --- from action_head/... ---`) during the move commit, then remove banners in a follow-up cleanup commit. This preserves reviewability of the move.
- If you hit a case where a current module truly does not fit any of the seven target subpackages, stop and raise the case rather than inventing a new subpackage. That's a design signal, not an implementation detail.
- Preserve git history where possible: use `git mv` for renames, not delete+add.
