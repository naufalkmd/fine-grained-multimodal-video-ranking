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
| `package.py` | Added optional files for ranking, SigLIP, centrality and reference videos | These submissions need them, and the ZIP otherwise drops files that aren't listed |

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
  `--qwen-reverse-actions` also lets the reversed-action sentence be correct on reversed frames.
- `qwen_negatives.py`: asks a local Qwen3-4B-Instruct (float16, fits a T4) to rewrite each
  description as one short sentence and make four single-change negatives. Only code rules filter
  the output. The JSONL keeps the raw output and checks for inspection. Data goes on Drive, not git.
- `qwen_candidates.py`: adds the test-like `qwen` set to a validation candidates file and checks
  it for text-only cues.
- `action_pairing.py`: checks whether reversing the frames flips a submission's choice between a
  short positive and its reversed-action sentence. Wording cancels out, so chance is 0.5.
- `submission_central.py`, `export_central.py`: a text-only submission that ranks candidates by
  how similar each is to the others, used to probe how the test negatives were made.
- `submission_ranker.py`: exported as the submission's `submission.py`. With `central_weight` in
  `inference_config.json` (set by `export_ranker.py --central-weight`), it adds candidate centrality.
- `export_ranker.py`: creates a submission folder from a `train_ranker.py` checkpoint.
- `final_colab_text.ipynb`: produces a text-only final submission (`FINAL_RULE` "central" or
  "regime") on a T4, after the organizers' baseline training that package.py needs. Built by
  `make_final_notebook.py --kind text`.
- `compare_public.py`: compares public per-episode CSVs with a baseline (paired difference, sign
  test, rank moves). Keep the CSVs in `cuda/public_csv/`, which git ignores.
- `final_colab.ipynb`: produces the final model on a T4. Built from the organizers' notebook by
  `make_final_notebook.py`, so edit that script rather than the notebook.
- `submission_siglip.py`, `export_siglip.py`: ranks with SigLIP's image-text match, with no training.
- `submission_siglip_central.py`: adds SigLIP and centrality ranks, weighted by `central_rank_weight`.
- `submission_combined.py`, `export_combined.py`: adds a trained ranker's rank and SigLIP's rank.
- `submission_siglip_diff.py`: SigLIP on only the words that differ between candidates. Dropped.
- `submission_evidence.py`, `export_evidence.py`: ranks by what this video adds, using fixed reference videos.
- `submission_shortlist.py`: centrality makes a shortlist, and image evidence orders it.
- `smolvlm_scorer.py`: scores candidates by SmolVLM's likelihood of writing them after the frames,
  minus a weighted no-frame likelihood. Exported with the submission.
- `train_smolvlm.py`: fine-tunes SmolVLM on the correct descriptions only, with frames left out of
  some batches so that it also learns the no-frame likelihood.
- `smolvlm_check.py`: scores SmolVLM with the right and a wrong video, at prior weights 0 and 1,
  and runs action pairing. Saves the raw scores.
- `train_smolvlm_contrast.py`: continues fine-tuning so that the video, not the wording, separates
  one-word edits; compares each edit on the episode's frames and another episode's.
- `time_submission.py`: times an exported submission and projects the 1,910-episode test.
- `text_rules.py`: text-only rules (centrality, familiar or unfamiliar wording, interleaving, a
  structure gate) scored on `qwen_long_negatives.py`'s LLM-edited validation sets.
- `submission_fusion.py`, `export_fusion.py`: text signals (centrality, unfamiliar words or characters
  under counts from the training descriptions, shortness) combined by one signal, interleaving or
  public rank shares (`bayes`), optionally with the structure gate. Set in `weights/rule.json`.
- `order_check.py`: checks locally whether a candidate's events fit the frames in the stated order
  (SigLIP, no training). `submission_order.py`, `export_order.py`: the same as an upload probe.
- `merge_bounds.py`: exact worst, best and overlap-fitted scores for merging two rules, from their
  public per-episode CSVs.
- `submission_regime.py`, `export_regime.py`: centrality first when some candidate's words all
  survive in the others (the rest look like edits of it), otherwise unfamiliar wording first; the
  two alternate after that. Text only.
- `submission_smolvlm_gate.py`, `export_smolvlm_gate.py`: centrality decides near-identical
  candidates and SmolVLM decides pairs that describe different scenes (public 0.5502); with
  `--video-margin` it also decides pairs it scores far apart (public 0.5382).
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

SmolVLM gate (fine-tuning about 25 minutes on an RTX 5070; the check needs the Qwen files above).
The fine-tuned model is on the team Drive next to the Qwen data, as the folder `ft_v1` (494 MB,
float16, with `training_info.json`); ask njunu17 for the link. Put it at
`cuda/outputs/smolvlm/ft_v1/` to skip the training step below.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct', local_dir='cuda/models/smolvlm-256m-instruct')"
python -m team.train_smolvlm --splits cuda/outputs/baseline/splits.json --output-dir cuda/outputs/smolvlm/ft_v1
python -m team.smolvlm_check --model-dir cuda/outputs/smolvlm/ft_v1/model --output cuda/outputs/smolvlm/check_ft_v1.json
python -m team.export_smolvlm_gate --model-dir cuda/outputs/smolvlm/ft_v1 \
    --kit-submission cuda/sample_submission --output-dir cuda/smolvlm_gate_v1_submission --max-frames 4
python package.py --submission-dir cuda/smolvlm_gate_v1_submission --output cuda/smolvlm_gate_v1.zip
```

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
