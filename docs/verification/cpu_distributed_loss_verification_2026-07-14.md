# CPU Distributed-Loss Verification

Status: accepted; passed on the remote data-disk environment
Date: 2026-07-14

## Scope

`tests/training/test_distributed_loss_integration.py` is a real two-process CPU
DDP integration test for the globally masked loss denominator. It does not
require a GPU. The test is now accepted after a successful rerun on the remote
data-disk environment with sufficient memory.

## Accepted rerun

The test passed with exit status zero on 2026-07-14 using:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=1 \
  ../envs/prsim/bin/pytest -q \
  tests/training/test_distributed_loss_integration.py
```

The rerun exercised both CPU ranks, unequal valid-element counts, empty local
transition populations, and two-step gradient accumulation. No numerical,
distributed-protocol, or process-lifecycle failure occurred.

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

These surrounding checks remain complementary to the accepted real-DDP rerun.

## Acceptance criteria

The accepted test covers all of the following:

1. `torch.distributed.run` exits with status 0;
2. both learned weights equal approximately `[0.96, 0.96]`;
3. `total_l1` equals approximately `7 / 175`;
4. `gripper_transition_recall` equals approximately `1.0`.

The earlier `SIGKILL` attempts remain useful evidence about the 2 GiB host
limit, but they no longer block acceptance and were not a confirmed code defect.
