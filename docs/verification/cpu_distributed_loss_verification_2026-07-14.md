# Outstanding CPU Distributed-Loss Verification

Status: environment-limited; not accepted as passing
Date: 2026-07-14

## Scope

The only unfinished verification from the 2026-07-13 static-review fixes is
`tests/training/test_distributed_loss_integration.py`. This is a real two-process
CPU DDP integration test for the globally masked loss denominator. It does not
require a GPU.

The implementation and its ordinary CPU coverage are complete. This note does
not classify the test as passed and does not identify a numerical assertion
failure.

## What happened on the current host

The test was attempted twice. In both attempts, worker rank 1 was terminated by
external `SIGKILL` after approximately 66--68 seconds. The launcher consequently
returned a nonzero status before the test could inspect its result file.

The container memory limit is 2,147,483,648 bytes (2 GiB). During a monitored
attempt, cgroup usage reached approximately 1.85 GiB; immediately before the
worker disappeared, the launcher used approximately 376 MiB RSS and each worker
used approximately 555 MiB RSS. This is strongly consistent with the host memory
ceiling, but it is not recorded as a proven cgroup OOM because `memory.events`
did not report an OOM event.

No Python exception, failed numerical assertion, or distributed protocol error
preceded the signal.

## Completed surrounding verification

On the same CPU-only host:

- the suite excluding this one test completed with 244 passed and 6 skipped;
- the targeted review-fix tests completed with 55 passes;
- the two-process checkpoint failure-propagation test passed;
- Ruff and the whitespace/diff check passed.

These results do not substitute for the unfinished real-DDP loss test.

## Required rerun

Rerun on a CPU host or container with materially more than 2 GiB of available
memory (4 GiB is the recommended minimum for this isolated test):

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=1 \
  pytest -q tests/training/test_distributed_loss_integration.py
```

Acceptance requires all of the following:

1. `torch.distributed.run` exits with status 0;
2. both learned weights equal approximately `[0.96, 0.96]`;
3. `total_l1` equals approximately `7 / 175`;
4. `gripper_transition_recall` equals approximately `1.0`.

Until that rerun succeeds, report the item as **environment-limited and
unverified**, not passed and not a confirmed code defect.
