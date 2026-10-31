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
| `.gitignore` | Added `Data/` and `*_submission/` | Keeps the dataset and exported submissions out of git |
| `package.py` | Added `ranker_model.py` as an optional file | Ranking models need it, and the ZIP otherwise drops files that aren't listed |

## Files

- `negatives.py`: generates negative descriptions from the organizers' data with a fixed seed.
- `lexical.py`: hub-corrected word-overlap similarity used by the `lexical` negative type.
- `train_negatives.py`: training negatives, k per episode with a set count from each type,
  resampled every epoch from the training split only, and saved as a record.
- `rank_eval.py`: scores an exported submission on fixed 5-candidate validation sets, and runs
  the scoring, text-only leak and frame-order checks.
- `ranker_model.py`: the `mean_pool` and `temporal` models. Exported with the submission.
- `train_ranker.py`: trains a model on generated negatives, using a listwise ranking loss.
- `submission_ranker.py`: exported as the submission's `submission.py`.
- `export_ranker.py`: creates a submission folder from a `train_ranker.py` checkpoint.
- `tests/`: tests for the above.

```bash
python -m pytest tests team/tests
python -m team.rank_eval --config cuda/config.json --run-dir cuda/outputs/baseline \
    --submission-dir cuda/sample_submission --trained-on random
python -m team.train_negatives --config cuda/config.json --run-dir cuda/outputs/baseline \
    --k 4 --mix lexical=2,similar_episode=2 --epochs 20

python -m team.train_ranker --config cuda/config.json --splits cuda/outputs/baseline/splits.json \
    --train-negatives reorder,swapped_action,similar_episode,random --model temporal \
    --output-dir cuda/outputs/temporal_v2
python -m team.export_ranker --checkpoint cuda/outputs/temporal_v2/best.pt \
    --output-dir cuda/temporal_v2_submission
python package.py --submission-dir cuda/temporal_v2_submission --output cuda/temporal_v2.zip
```
