# Flappy Bird with PyTPG

This is an isolated Flappy Bird experiment powered by
[Ryan-Amaral/PyTPG](https://github.com/Ryan-Amaral/PyTPG). It does not import the
compact `tpg.py`; both algorithms import the same benchmark from
`../shared_flappy/`.

## What is included

- The repository-wide shared deterministic Flappy Bird environment.
- Headless training with the real PyTPG `Trainer` and `Agent` APIs.
- Procedural Pygame rendering, so no image assets are required.
- Best-agent and resumable trainer checkpoints.
- Evaluation over unseen pipe-layout seeds.
- Environment unit tests that do not require a display.

## Isolation from the parent project

PyTPG-specific source, dependencies, and generated artifacts live below
`pytpg_flappy/`; only the shared environment/evaluator lives in
`../shared_flappy/`. The import loader removes the parent repository from
Python's package search path after the shared benchmark is imported. This
avoids the parent `tpg.py` shadowing PyTPG's `tpg` package.

Use a dedicated virtual environment inside this directory so installing PyTPG
does not change the parent project's environment.

## Installation on Windows

Python 3.10 or 3.11 is recommended for the upstream project's older
`setup.py`-based package.

```powershell
cd C:\code\flappy-bird-tpg\pytpg_flappy
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not use `pip install tpg`: the package with that name on PyPI is unrelated.
The requirements file installs PyTPG directly from its official GitHub
repository, pinned to the official commit tested by this project
(`7295f90ececbfc34fdbc1d73e032a9c2407a182c`). It also constrains NumPy to 1.x
because that PyTPG source imports the legacy `numpy.NINF` symbol removed in
NumPy 2.x.

## Train

```powershell
.\.venv\Scripts\python.exe train.py --generations 200
```

Useful options:

```powershell
.\.venv\Scripts\python.exe train.py `
  --generations 300 `
  --population 80 `
  --episodes 3 `
  --validation-episodes 5 `
  --seed 42
```

Outputs are written to `artifacts/shared_env/`:

- `best_agent.pkl`: best policy encountered so far.
- `latest_trainer.pkl`: complete population for resuming training.
- `training.json`: last-generation statistics and configuration.

Resume a saved population:

```powershell
.\.venv\Scripts\python.exe train.py `
  --generations 100 `
  --resume artifacts\shared_env\latest_trainer.pkl
```

`--generations` means additional generations when `--resume` is used.

## Watch the trained agent

```powershell
.\.venv\Scripts\python.exe play.py
```

Use `Esc` or close the window to stop. To replay a fixed pipe layout:

```powershell
.\.venv\Scripts\python.exe play.py --seed 42 --episodes 5
```

## Evaluate on unseen layouts

```powershell
.\.venv\Scripts\python.exe evaluate.py --episodes 100 --seed 100000
```

Evaluation reports reward and passed-pipe distributions. Evaluation seeds
should not overlap the training seeds.

## Test the environment

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Observation, action, and fitness

The five normalized observations shared with the compact implementation are:

1. bird vertical position;
2. bird vertical velocity;
3. horizontal distance to the next pipe;
4. next gap top edge;
5. next gap bottom edge.

Actions are `0` (do nothing) and `1` (flap). Fitness is the episode return:

- `+0.1` per surviving frame;
- `+5` per passed pipe;
- `-10` on collision.

Both implementations call the exact same `shared_flappy.evaluate_agent`
function. Every agent and every generation sees the same training layouts. The
best-agent checkpoint is selected on a second, fixed validation seed set, so a
lucky training layout does not overwrite a stronger saved policy.

## Suggested first experiment

The defaults intentionally use shorter programs and a smaller population than
PyTPG's Atari-oriented defaults. First run 50 generations as a smoke test, then
increase to 200-500 generations. For a meaningful comparison with the parent
implementation, use the same number of environment steps, at least 10 training
seeds, and a separate set of 100 evaluation seeds.
