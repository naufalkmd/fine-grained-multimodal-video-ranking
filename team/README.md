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
- `word_swap.py`: finds replacement words from the descriptions themselves, for `word_swap` negatives.
- `train_negatives.py`: training negatives, k per episode with a set count from each type,
  resampled every epoch from the training split only, and saved as a record.
- `train_rank.py`: trains the kit model on 1 correct + k negatives with `--loss listwise` or
  `--loss bce`, picking checkpoints by the 5-candidate score. Its checkpoints work with `export.py`.
- `rank_eval.py`: scores an exported submission on fixed 5-candidate validation sets, and runs
  the scoring, text-only leak and frame-order checks.
- `ranker_model.py`: the `mean_pool` and `temporal` models. Exported with the submission.
- `train_ranker.py`: trains a model on generated negatives, using a listwise ranking loss. With
  `--mix` and `--k` it uses fixed counts per type from `train_negatives.py` and ranks 1 + k
  candidates; without them it draws 4 negatives of random types as before.
  With `--qwen-candidates` it also trains on the short positives and single-change negatives
  from `qwen_negatives.py`, on the share of episodes set by `--qwen-probability`.
- `qwen_negatives.py`: asks a local Qwen3-4B-Instruct (float16, fits a T4) to rewrite each
  description as one short sentence and make four single-change negatives. Only code rules filter
  the output. The JSONL keeps the raw output and checks for inspection. Data goes on Drive, not git.
- `qwen_candidates.py`: adds the test-like `qwen` set to a validation candidates file and checks
  it for text-only cues.
- `submission_central.py`, `export_central.py`: a text-only submission that ranks candidates by
  how similar each is to the others, used to probe how the test negatives were made.
- `submission_ranker.py`: exported as the submission's `submission.py`.
- `export_ranker.py`: creates a submission folder from a `train_ranker.py` checkpoint.
- `final_colab.ipynb`: produces the final model on a T4. Built from the organizers' notebook by
  `make_final_notebook.py`, so edit that script rather than the notebook.
- `tests/`: tests for the above.

Qwen candidates, built once (about 35 minutes on an RTX 5070) and shared on Drive:

```bash
python -m team.qwen_negatives --config cuda/config.json --splits cuda/outputs/baseline/splits.json \
    --model-dir cuda/models/qwen3-4b-instruct-2507 --output cuda/outputs/qwen/candidates_v3.jsonl
python -m team.qwen_candidates --generated cuda/outputs/qwen/candidates_v3.jsonl \
    --candidates cuda/outputs/validation/candidates_lexical_csls.json \
    --output cuda/outputs/validation/candidates_qwen.json
python -m team.train_ranker --config cuda/config.json --splits cuda/outputs/baseline/splits.json \
    --train-negatives reorder,swapped_action,similar_episode,random,lexical --model temporal \
    --flip-probability 0.5 --epochs 60 --patience 60 \
    --qwen-candidates cuda/outputs/qwen/candidates_v3.jsonl --qwen-probability 0.5 \
    --candidates cuda/outputs/validation/candidates_qwen.json --output-dir cuda/outputs/temporal/Q1_h_qwen50
```

Download the model first with `huggingface_hub.snapshot_download("Qwen/Qwen3-4B-Instruct-2507",
local_dir="cuda/models/qwen3-4b-instruct-2507")`. It is used for augmentation only and is never
part of a submission.

Validation candidates with the `lexical` set: `rank_eval.py` builds a candidates file the first
time it is pointed at a path that doesn't exist, using every type in `negatives.py`. Generation
is fixed by the seed, so every machine gets the same file. Build it once without a submission:

```bash
python -m team.rank_eval --config cuda/config.json --run-dir cuda/outputs/baseline \
    --trained-on random --candidates cuda/outputs/validation/candidates_lexical_csls.json
```

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

python -m team.train_ranker --config cuda/config.json --splits cuda/outputs/baseline/splits.json \
    --mix lexical=3,similar_episode=2,random=1,reorder=2 --k 8 --model temporal \
    --no-cross-attention --flip-probability 0.5 --epochs 60 --patience 60 \
    --candidates cuda/outputs/validation/candidates_lexical_csls.json \
    --output-dir cuda/outputs/temporal/T2_k8_l3s2r1o2

python -m team.train_rank --config cuda/config.json --output-dir cuda/outputs/listwise_lex2_sim2 \
    --k 4 --mix lexical=2,similar_episode=2 --loss listwise \
    --candidates cuda/outputs/validation/candidates_lexical_csls.json
```
