# LIBERO and CALVIN data-protocol acceptance record

Status: accepted for training-data and closed-loop evaluation use
Date: 2026-07-13

## Scope

This record accepts the dataset roots, normalization artifacts, view/action
adapters, and simulator split boundaries used by PrismVLA. It supersedes the earlier
inventory made before LIBERO materialization and CALVIN ABC reconstruction were
complete.

This is a data-protocol decision only. It does not validate model capacity or
training hyperparameters. In particular, the currently configured action-head
values `512 / 8 / 4` are a provisional runnable baseline and are outside this
record's acceptance scope.

All paths are relative to the repository root and remain on the data disk. Training
storage uses LeRobot v2.1.

## Accepted roots

| Use | Root or asset | Status |
| --- | --- | --- |
| LIBERO training | `../benchmarks/libero/lerobot-v2.1-rotate180` | accepted |
| LIBERO evaluation | installed LIBERO task definitions and initial states | accepted |
| CALVIN A/B/C training | `../benchmarks/calvin/lerobot/task_ABC_D_complete` | accepted |
| CALVIN D numeric reference | `../benchmarks/calvin/lerobot/task_D_D` | reference only; excluded from training statistics |
| CALVIN D closed-loop evaluation | `../benchmarks/calvin/runtime/dataset/task_ABC_D/validation` | accepted |

The incomplete `../benchmarks/calvin/lerobot/task_ABC_D` snapshot is provenance
input only: it contains 7,870 of 17,870 declared Parquet files. Consumers must use
the `_complete` root.

## LIBERO

### Inventory and schema

| Suite | Episodes | Frames | Parquet | Videos |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 500 | 62,250 | 500 | 1,000 |
| `libero_object` | 500 | 74,507 | 500 | 1,000 |
| `libero_goal` | 500 | 63,728 | 500 | 1,000 |
| `libero_10` | 500 | 138,090 | 500 | 1,000 |
| **Total** | **2,000** | **338,575** | **2,000** | **4,000** |

Each suite declares LeRobot v2.1, 10 tasks, and 50 demonstrations per task. The
strict reader verified metadata counts, task maps, all expected Parquet/video paths,
and representative two-view decoding.

State is `[ee_position, ee_axis_angle, left_gripper, right_gripper]`. Actions contain
six OSC pose deltas and canonical gripper `0=close, 1=open`; materialization converts
the source convention `-1=open, +1=close` exactly once with `(1 - source) / 2`.

Both stored views use `rotate_180`:

| Canonical view | LeRobot key | Simulator key |
| --- | --- | --- |
| `primary` | `observation.images.image` | `agentview_image` |
| `wrist` | `observation.images.wrist_image` | `robot0_eye_in_hand_image` |

The evaluation adapter applies the same transform before sending a request.

### Materialization and normalization identities

| Suite | Plan SHA-256 | Run SHA-256 |
| --- | --- | --- |
| `libero_spatial` | `b5e907cb017986f5d5c56650a68de0565f9c3b7c17478c96ca0131a80a32bc12` | `5e94898d50bfc1f132f87b3d34c0d6329fa16d6e6d23b3425766e18876d1a992` |
| `libero_object` | `0efc15ab2c5240ee07484502ca463e3e76b14c99e62d44c71ae9731936385e31` | `066686aaa55901bb11b9834cd96bc17fd96faf72f904e97e02311bb57a9a0d56` |
| `libero_goal` | `b86b09df806337615d64ff9255690c8931468432e8d8e7949e0853b64a396abd` | `08682c6d692aafcef3b84dd9ad8df0e6f5822872268f9d2a537a347b6df50e70` |
| `libero_10` | `f3cfb87aaaf9145dcde9c8c4c8270c54526dbbce4b90f2403c27f8e5a9ef53f3` | `0ee20f7ddac47a09df3e9c9ce7858d6aa05b12103db5e387f9fb22a6df34eaf2` |

Detailed per-episode journals and artifact hashes live under `.materialization` and
`meta/materialization.json` in each suite.

Normalization artifact:
`../benchmarks/libero/lerobot-v2.1-rotate180/statistics.json`

- format/group/count: `prism-normalization-v1` / `libero` / 338,575
- schema SHA-256: `b394006ed02018e887c4266ccdfd48fdb198aa0b63a6d175143f9444d20bc781`
- canonical content SHA-256: `e21a88c25ad4e666973435a9f535c141b22b6c2a98c8c73b50355de43c6314ce`
- file SHA-256: `92456dba6899ecf792cb012cb980e315d1bd0cb10902a6128df6295a5bfd27cc`

The first six action dimensions use q01/q99 scaling and clip to `[-1, 1]`;
canonical gripper uses identity normalization.

## CALVIN A/B/C reconstruction

The complete root is reproduced from pinned authorities:

- Collision target ordering, metadata, and videos:
  `CollisionCode/calvin_abc_d_lerobot_v2.1@7e206b2aa210c5166276b8e9777955bfd1a1e8ac`
- Traly numeric donor:
  `Traly/calvin_abc_d-lerobot@92bf05b93a4ba8a8825f2bffb1b78ff4cb4e6c63`
- episode mapping SHA-256: `9d4d5d29c6710981f4e4983f8a2ae3d4f42697c03227553a656afad3c20690b7`
- plan SHA-256: `fd309d2cf6651e85f767d0af966a55596a8a9d9c933bc26cfaec4277ec6db0f0`
- run SHA-256: `1aca55a97f0c9527560b93749437a6029d91256d6c85aaaeebc8a9b0a808201b`

All 7,870 existing target Parquets were bit-compared with their mapped donor rows;
state, action, first-action, and scalar/index mismatch counts were zero. The
remaining 10,000 Parquets were generated from the pinned donor. All 35,740 target
videos are present. The incomplete source root was not overwritten.

| Episodes | Frames | Tasks | Parquet | Videos |
| ---: | ---: | ---: | ---: | ---: |
| 17,870 | 1,071,743 | 389 | 17,870 | 35,740 |

The on-disk state layout is
`[tcp_xyz, tcp_euler_xyz, constant_zero, gripper_width]`. Actions contain six
relative TCP values plus signed gripper `-1=close, +1=open`; the DataSpec converts
the seventh value once with `(source + 1) / 2`.

Normalization artifact:
`../benchmarks/calvin/lerobot/task_ABC_D_complete/statistics.json`

- format/group/count: `prism-normalization-v1` / `calvin_abc` / 1,071,743
- schema SHA-256: `3b59dd638fa3b2de7fa904af96c772ea45049863844e475e656cd5a1d0296c95`
- canonical content SHA-256: `17a9fca394fb830f30f6edeb64826d437b7ddad8622a29aec2a97ac7bcdba790`
- file SHA-256: `0f02628d203285e5b0f85348b5fadc1ebe09e01a6df7d1a9401da7ff76569abe`

Only A/B/C training frames contribute to these statistics.

## Scene-D and shared simulator protocol

`../benchmarks/calvin/lerobot/task_D_D` contains 5,124 episodes and 308,918 frames.
It is a numeric reference only and cannot replace A/B/C training data or supply
training statistics. Closed-loop evaluation instead resets from the official
scene-D runtime validation directory.

Both benchmark adapters expose ordered views `("primary", "wrist")`, an
eight-dimensional canonical state, and seven-dimensional actions. The policy returns
eight actions per request with shape `[B, 8, 7]`. Gripper decoding uses
`prediction > 0.5`; exactly 0.5 is closed.

| Benchmark | Closed | Open | Mapping from canonical `open` |
| --- | ---: | ---: | --- |
| LIBERO | `+1` | `-1` | `1 - 2 * open` |
| CALVIN | `-1` | `+1` | `2 * open - 1` |

The CALVIN adapter maps simulator state to `[robot_obs[:6], 0, robot_obs[6]]`. The
LIBERO adapter uses the materialized state order and applies the same 180-degree
view transform used for training.

Recorded opt-in integration tests exercised nine simulator steps from two policy
decisions in each benchmark and checked view order, state/action mapping, sparse
history, and control-step budgeting. Those environment tests remain the acceptance
evidence; they are not implied to run during every static or CPU-only review.

The data and simulator-data gate is complete. Model capacity experiments, production
training, and trained checkpoint quality are tracked separately and cannot change
the status of this data-protocol record.
