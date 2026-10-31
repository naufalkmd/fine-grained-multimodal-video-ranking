# Team code

Code we wrote, kept apart from the organizers' starter kit. Run it from the repository root
with `python -m`, so that it can import the kit's modules.

## What is where

| Location | Contents |
|---|---|
| Repository root, `tests/`, `docs/ko/` | The organizers' starter kit, with an English translation of its docs |
| `docs/brief.md`, `docs/validation.md`, `AGENTS.md`, `CLAUDE.md` | Our rules summary and working instructions |
| `team/` | Our code that isn't part of the submission |
| `cuda/` | Local GPU setup and local files. Data, models and outputs stay out of git |

Commit `a9a4179` is the kit as we received it, plus the translation and our docs. To see
everything changed since then, run `git diff a9a4179 --stat`.

## Changing kit files

The submission only contains what `export.py` copies, so inference code can't live in
`team/` without changing `export.py` and `package.py`. List each changed kit file here with
the reason.

| Kit file | Change | Why |
|---|---|---|
| `.gitignore` | Added `Data/` | Keeps the dataset out of git on case-sensitive systems |

## Files

- `negatives.py`: generates negative descriptions from the organizers' data with a fixed seed.
- `lexical.py`: hub-corrected word-overlap similarity used by the `lexical` negative type.
- `train_negatives.py`: training negatives, k per episode with a set count from each type,
  resampled every epoch from the training split only, and saved as a record.
- `rank_eval.py`: scores an exported submission on fixed 5-candidate validation sets, and runs
  the scoring, text-only leak and frame-order checks.
- `tests/`: tests for the above.

```bash
python -m pytest tests team/tests
python -m team.rank_eval --config cuda/config.json --run-dir cuda/outputs/baseline \
    --submission-dir cuda/sample_submission --trained-on random
python -m team.train_negatives --config cuda/config.json --run-dir cuda/outputs/baseline \
    --k 4 --mix lexical=2,similar_episode=2 --epochs 20
```
