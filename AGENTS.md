# AGENTS.md

This file defines the working rules for AI agents in this repository.

## 1. Remote Environment and Execution Requirements

1. All practical work must be performed on the remote server, including code changes, testing, training, downloads, builds, and output generation. The local machine should only be used for personal code review, file synchronization, and pushing commits to the remote repository.

2. Key-based SSH authentication has already been configured for the remote server. The private key is located at:

```bash
$HOME/.ssh/id_ed25519_autodl
```

The public key is located at:

```bash
$HOME/.ssh/id_ed25519_autodl.pub
```

3. After connecting to the remote server, switch to the data disk `autodl-tmp/` before working. Code, datasets, models, caches, logs, and output files must be stored on the data disk to avoid occupying the system disk.

4. Before downloading overseas resources, the server-provided network acceleration script may be enabled when available.

Note: this may slow down downloads from domestic mirrors. Disable it when using domestic sources.

5. If the official Hugging Face endpoint is slow or unstable, use the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Hugging Face caches, model weights, and large files must be stored on the data disk. Do not write them to the default system-disk cache paths.

6. The shutdown command is:

```bash
shutdown
```

Do not run high-risk operations such as shutdown, deleting large directories, clearing caches, or overwriting checkpoints unless explicitly requested.

## 2. Path Rules

1. Project-internal paths must use relative paths. Code, configs, scripts, documentation, tests, data paths, and output paths should be written relative to the repository root or the current working directory.

2. Do not hard-code personal absolute project paths in code, configs, or scripts.

## 3. Basic Principles

1. Understand before modifying. Before changing code, search and read the relevant implementation, configs, tests, and documentation. Do not guess based on prior assumptions.

2. Do not write temporary patches, hard-coded shortcuts, bypasses, or vague fallbacks just to make the current issue appear fixed.

3. Do not mix unrelated responsibilities in the same file, function, or script. Data processing, training, inference, evaluation, configuration, and CLI entry points must be clearly separated.

4. Do not copy and paste existing logic to create duplicate implementations. If duplicated logic is found, refactor it for reuse, or delete the old implementation after the new module is stable.

5. Do not put core logic inside scripts. Scripts should only parse arguments and call implementations from the main package.

6. Do not hard-code parameters inside code logic. Paths, models, datasets, hyperparameters, and run modes must come from config files, command-line arguments, or clearly defined constants.

7. If something is uncertain, search, verify, and explain the uncertainty. Do not make blind guesses.

## 4. Separation of Parameters and Logic

1. Code should define logic; configuration should define changeable behavior. Do not switch experiment settings by editing source code.

2. Changeable parameters should be externalized, including but not limited to:

   * data paths
   * output paths
   * checkpoint paths
   * model names
   * batch size
   * learning rate
   * training steps
   * evaluation settings
   * benchmark settings
   * cache settings

3. Do not introduce hidden parameters, magic numbers, unexplained paths, or hard-coded values that only work for a single experiment.

## 5. Modification Workflow

1. Before making changes, check the git status to avoid overwriting existing work:

```bash
git status -sb
```

2. Before modifying code, identify which layer the issue belongs to:

   * configuration
   * data
   * training
   * inference
   * evaluation
   * script entry points
   * engineering structure

3. Do not mix refactoring with behavior changes. Prefer splitting work into separate steps:

   * move or rename files without changing behavior
   * extract interfaces without changing behavior
   * change behavior
   * add tests
   * delete old code

4. Each change should be small but complete. Do not leave behind half-migrated, half-compatible, or half-deprecated parallel implementations.

5. Once an obsolete module has been fully replaced and tests pass, delete it. If it cannot be deleted immediately, mark it clearly as legacy and document the replacement module and removal condition.

## 6. Testing and Verification

1. All tests must be run in the remote project environment.

2. After modifying code, run the relevant tests. Do not simply say “it should work.”

3. Changes involving protocols, shapes, masks, dimensions, normalization, paths, config parsing, or input/output formats must include corresponding tests.

4. Before committing, run:

```bash
git diff --check
```

5. If the full test suite cannot be run, explain why and run the smallest meaningful alternative test.

## 7. Reproducibility

1. Training, evaluation, builds, downloads, and service startup must be reproducible through explicit commands.

2. Large files, model weights, datasets, caches, logs, videos, and experiment outputs must not be committed to git unless explicitly required by the repository and requested by the user.

3. Output files must be written to clear and intentional directories. Do not scatter outputs randomly across the repository.

## 8. Forbidden Behaviors

The following behaviors are strictly forbidden:

1. Modifying code without searching and reading the relevant implementation first.
2. Guessing the meaning of fields without understanding the existing logic.
3. Using temporary patches to hide structural problems.
4. Mixing multiple responsibilities in one script or function.
5. Writing core business logic inside scripts.
6. Replacing configuration with hard-coded values.
7. Using personal absolute paths inside the project.
8. Copying old code to create a second implementation.
9. Keeping deprecated code without cleanup.
10. Making large changes without adding or running tests.
11. Hiding unverified, unfinished, or uncertain work.


