# LIBERO and CALVIN data-protocol acceptance record

Status: accepted for training-data and closed-loop evaluation use

Date: 2026-07-13

## Purpose

This record is the final acceptance evidence for the LIBERO and CALVIN data roots,
normalization artifacts, and simulator-facing protocol. It supersedes the earlier
Phase 0 inventory that described LIBERO before materialization and CALVIN ABC while
its numeric files were incomplete.

All paths below are relative to the repository root and point to sibling directories
on the data disk. The accepted storage version is LeRobot v2.1 only.

## Final status

| Benchmark use | Accepted root or asset | Result |
| --- | --- | --- |
| LIBERO training | `../benchmarks/libero/lerobot-v2.1-rotate180` | accepted |
| LIBERO evaluation | installed LIBERO task definitions and initial states | accepted |
| CALVIN ABC training | `../benchmarks/calvin/lerobot/task_ABC_D_complete` | accepted |
| CALVIN scene-D LeRobot reference data | `../benchmarks/calvin/lerobot/task_D_D` | accepted, never used for ABC training statistics |
| CALVIN closed-loop evaluation | `../benchmarks/calvin/runtime/dataset/task_ABC_D/validation` | accepted |

The original `../benchmarks/calvin/lerobot/task_ABC_D` snapshot remains provenance
input only. It is not an accepted training root because it contains only 7,870 of the
17,870 declared Parquet files. Consumers must use the `_complete` root.

There is no remaining data or simulator-environment blocker. The remaining blockers
to a real training run are model and experiment decisions: the accepted values for
`action_hidden_size`, `num_attention_heads`, and `ffn_ratio` are still unresolved in
`configs/model/qwen35_query_memory.yaml`; no production training profile or trained
policy checkpoint exists yet. The checkpoint inference backend is implemented, but a
production launcher must load the selected policy weights and inject that loaded
policy into it.

## LIBERO final training root

### Inventory

Each suite has 10 language tasks and 50 demonstrations per task.

| Suite | Episodes | Frames | Parquet files | MP4 files |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 500 | 62,250 | 500 | 1,000 |
| `libero_object` | 500 | 74,507 | 500 | 1,000 |
| `libero_goal` | 500 | 63,728 | 500 | 1,000 |
| `libero_10` | 500 | 138,090 | 500 | 1,000 |
| **Total** | **2,000** | **338,575** | **2,000** | **4,000** |

Every suite declares `codebase_version: v2.1`. The strict reader validated the
metadata episode count, summed frame count, task map, and every expected Parquet and
two-view video path. Representative episodes from every suite decoded both canonical
views as contiguous `uint8` arrays with shape `[1, 128, 128, 3]`.

The materialization retained every source frame; it did not apply no-op filtering.
The stable state is `[ee_position, ee_axis_angle, left_gripper, right_gripper]` with
eight values. The first six action values remain the source OSC pose deltas. The
source gripper convention `-1=open, +1=close` was materialized once as canonical
`0=close, 1=open` using `(1 - source) / 2`.

### Image and view contract

Both training videos are materialized with `rotate_180`. The canonical view mapping
is:

| Canonical view | LeRobot physical key | Simulator observation key |
| --- | --- | --- |
| `primary` | `observation.images.image` | `agentview_image` |
| `wrist` | `observation.images.wrist_image` | `robot0_eye_in_hand_image` |

The evaluation adapter applies the same 180-degree rotation to both simulator views
before constructing a request. A source-to-video spot check measured mean absolute
pixel errors of about 1.94 and 2.09 against the rotated source images, versus 70.77
and 75.44 against the unrotated images. The small residual is the expected lossy AV1
encoding error.

### Materialization identities

| Suite | Plan SHA-256 | Run SHA-256 |
| --- | --- | --- |
| `libero_spatial` | `b5e907cb017986f5d5c56650a68de0565f9c3b7c17478c96ca0131a80a32bc12` | `5e94898d50bfc1f132f87b3d34c0d6329fa16d6e6d23b3425766e18876d1a992` |
| `libero_object` | `0efc15ab2c5240ee07484502ca463e3e76b14c99e62d44c71ae9731936385e31` | `066686aaa55901bb11b9834cd96bc17fd96faf72f904e97e02311bb57a9a0d56` |
| `libero_goal` | `b86b09df806337615d64ff9255690c8931468432e8d8e7949e0853b64a396abd` | `08682c6d692aafcef3b84dd9ad8df0e6f5822872268f9d2a537a347b6df50e70` |
| `libero_10` | `f3cfb87aaaf9145dcde9c8c4c8270c54526dbbce4b90f2403c27f8e5a9ef53f3` | `0ee20f7ddac47a09df3e9c9ce7858d6aa05b12103db5e387f9fb22a6df34eaf2` |

Each suite stores its plan, per-episode journal, artifact sizes, frame counts, and
artifact SHA-256 values under `.materialization` and `meta/materialization.json`.

### Normalization artifact

The accepted artifact is
`../benchmarks/libero/lerobot-v2.1-rotate180/statistics.json`.

- format: `prism-normalization-v1`
- group: `libero`
- datasets: all four accepted suites
- state/action count: 338,575
- schema SHA-256:
  `b394006ed02018e887c4266ccdfd48fdb198aa0b63a6d175143f9444d20bc781`
- canonical content SHA-256:
  `e21a88c25ad4e666973435a9f535c141b22b6c2a98c8c73b50355de43c6314ce`
- file SHA-256:
  `92456dba6899ecf792cb012cb980e315d1bd0cb10902a6128df6295a5bfd27cc`

The first six action values use q01/q99 scaling with a hard `[-1, 1]` clip. The
seventh value is identity canonical `open_01` with threshold 0.5. Quantiles that
coincide with saturated physical action bounds are recorded as such and are not a
completeness failure.

## CALVIN ABC reconstruction and final training root

### Pinned sources and mapping

The reconstruction is reproducible from two pinned repositories:

- target ordering, metadata, and videos:
  `CollisionCode/calvin_abc_d_lerobot_v2.1` at
  `7e206b2aa210c5166276b8e9777955bfd1a1e8ac`
- complete numeric donor:
  `Traly/calvin_abc_d-lerobot` at
  `92bf05b93a4ba8a8825f2bffb1b78ff4cb4e6c63`

The deterministic episode mapping matches task text, episode length, and exact
numeric signatures while preserving the Collision target ordering.

- mapping SHA-256:
  `9d4d5d29c6710981f4e4983f8a2ae3d4f42697c03227553a656afad3c20690b7`
- materialization plan SHA-256:
  `fd309d2cf6651e85f767d0af966a55596a8a9d9c933bc26cfaec4277ec6db0f0`
- materialization run SHA-256:
  `1aca55a97f0c9527560b93749437a6029d91256d6c85aaaeebc8a9b0a808201b`

The numeric conversion uses `float32(action.relative)` without a one-row shift,
including row zero. State is converted to
`float32([observation.state[:6], 0, observation.state[6]])`; timestamps are
`float32(frame_index) / float32(10)`, and indices are regenerated in Collision target
order.

### Existing-data gate and generated episodes

Before reuse, all 7,870 physically present target episodes were compared against the
mapped donor rows with exact equality. The gate covered 473,294 frames and reported:

| Check | Result |
| --- | ---: |
| state mismatches | 0 |
| action mismatches | 0 |
| first-action mismatches | 0 |
| scalar/index mismatches | 0 |

The schema fingerprint is
`7431c1d23c11091895ddfced52ed82415e38a955c564c1b1506e681fa990aea5`.
Only after this bit-exact gate passed were those 7,870 Parquet files hard-linked into
the isolated output. The other 10,000 Parquet files were generated from the mapped
and pinned Traly numeric source.

All 35,740 target videos are present and hard-linked into the final root. Three wrist
videos missing from the initial local snapshot were fetched from the pinned Collision
revision and verified against their LFS SHA-256 identities before plan construction.
The original incomplete root was not overwritten.

### Final inventory and validation

The accepted root is `../benchmarks/calvin/lerobot/task_ABC_D_complete`.

| Episodes | Frames | Tasks | Parquet files | MP4 files |
| ---: | ---: | ---: | ---: | ---: |
| 17,870 | 1,071,743 | 389 | 17,870 | 35,740 |

The strict LeRobot v2.1 reader passed the entire root. Final validation decoded
episodes 0, 8,935, and 17,869 and verified their numeric row counts. The decoded
canonical views have shapes `[1, 200, 200, 3]` for `primary` and
`[1, 84, 84, 3]` for `wrist`.

The on-disk state layout is `[tcp_xyz, tcp_euler_xyz, constant_zero, gripper_width]`.
The on-disk action layout is six relative TCP values plus signed gripper command
`-1=close, +1=open`. The DataSpec canonicalizes that seventh value once with
`(source + 1) / 2` before policy training.

### Normalization artifact

The accepted artifact is
`../benchmarks/calvin/lerobot/task_ABC_D_complete/statistics.json`.

- format: `prism-normalization-v1`
- group: `calvin_abc`
- dataset: `calvin_abc`
- train/eval provenance: A/B/C for training and D for evaluation
- state/action count: 1,071,743
- schema SHA-256:
  `3b59dd638fa3b2de7fa904af96c772ea45049863844e475e656cd5a1d0296c95`
- canonical content SHA-256:
  `17a9fca394fb830f30f6edeb64826d437b7ddad8622a29aec2a97ac7bcdba790`
- file SHA-256:
  `0f02628d203285e5b0f85348b5fadc1ebe09e01a6df7d1a9401da7ff76569abe`

Only ABC training frames contribute to these statistics. Scene D is excluded.

## CALVIN scene-D reference data and evaluation runtime

The complete LeRobot D root at `../benchmarks/calvin/lerobot/task_D_D` contains:

| Episodes | Frames | Tasks | Parquet files | MP4 files |
| ---: | ---: | ---: | ---: | ---: |
| 5,124 | 308,918 | 389 | 5,124 | 10,248 |

It passes the same strict reader and two-view decode contract. It contains scene-D
demonstrations for numeric reference; it is not an admissible substitute for ABC
demonstrations and is not the reset source used by the closed-loop runner. The
training/statistics configuration rejects the `task_D_D` root and any split-D
leakage.

Closed-loop CALVIN evaluation uses the official runtime validation directory rather
than the LeRobot D root. The runtime directory contains 99,022 physical NPZ frames;
`ep_start_end_ids.npy` describes four ranges whose inclusive lengths also sum to
99,022. Its merged environment config selects
`calvin_table_D/urdf/calvin_table_D.urdf`.

The similar names are intentional: `task_ABC_D` denotes the official ABC-to-D
protocol, and its `validation` directory is scene D. The accepted ABC training root
is the separate LeRobot `_complete` directory and contains no validation frames.

## Shared policy and simulator protocol

Both benchmarks expose ordered canonical views `("primary", "wrist")`, an
eight-dimensional raw canonical state, and a seven-dimensional action. The policy
predicts eight actions in parallel with shape `[B, 8, 7]`; motion dimensions use the
stored q01/q99 statistics, while gripper remains canonical `open_01`.

Gripper decoding uses the strict predicate `prediction > 0.5`. A prediction equal to
0.5 is closed. The binary canonical result is mapped to each simulator as follows:

| Benchmark | Closed | Open | Mapping |
| --- | ---: | ---: | --- |
| LIBERO | `+1` | `-1` | `1 - 2 * open` |
| CALVIN | `-1` | `+1` | `2 * open - 1` |

There is no configurable CALVIN gripper mode and no action-autoencoder step in this
accepted path.

The CALVIN evaluation adapter projects raw simulator state to the training layout as
`[robot_obs[:6], 0, robot_obs[6]]`. The LIBERO adapter builds the same state ordering
used during materialization and applies the accepted 180-degree transform to both
views.

## Real evaluation-environment acceptance

The installed LIBERO benchmark assets expose all four accepted suites, 10 tasks per
suite, and 50 initial states per task. The opt-in real integration test passed in the
dedicated LIBERO environment with robosuite/MuJoCo: it executed nine control steps
from two policy decisions and verified canonical state, view ordering, sparse history,
and control-step budgeting.

The CALVIN opt-in integration test passed in the dedicated simulator environment. It
loaded the scene-D PyBullet environment through EGL on the NVIDIA GPU, executed nine
control steps from two policy decisions, and verified the same request/history and
budget contracts.

The data acceptance gate is therefore complete for LIBERO training/evaluation and
CALVIN ABC training plus D evaluation. Future failures caused by unresolved model
dimensions, missing trained weights, or production launcher composition are not data
or simulator-data failures.
