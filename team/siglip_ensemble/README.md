# Kyounghoon: submitted SigLIP sentence-residual ensemble

This folder preserves the training logic and the exact ZIP used for the reported
public submission. It is a recorded experiment, not a claim of improved hidden-test
performance.

| Public result (reported by participant) | Value |
| --- | ---: |
| Score | **0.3630** |
| Evaluated episodes | 573 / 573 |
| Errors displayed | 0 / 0 |
| Time | 59.94 seconds |

## Download the submitted model

`siglip_minimal_3seed_submission.zip` (379,382,302 bytes, stored with **Git LFS**) was
removed from the current tree after the competition so that clones stay small. It is kept
in the history at commit `1cd0063`: download it from
[its file page at that commit](https://github.com/naufalkmd/fine-grained-multimodal-video-ranking/blob/1cd0063/team/kyounghoon/siglip_minimal_3seed_submission.zip),
or run `git checkout 1cd0063 -- team/kyounghoon/siglip_minimal_3seed_submission.zip`
followed by `git lfs pull` (it lands at its old path, `team/kyounghoon/`). Do not submit the small LFS pointer text file or the
repository's source archive.

SHA256:

```text
ac7e50137f7efefb54ee8475d3a9eab0255c6b6ca58d1c9b87d3df4962ace8ee
```

The ZIP includes offline SigLIP weights, tokenizer and image processor, the three
trained residual heads, and the API v2 inference code. Its root contains
`submission.py` and `submission.json`. It contains no training images, captions,
feature caches, or validation answers.

## Contents

- `training/`: the exact 36 Python source files archived by the completed training
  run, including augmentation, sampling, caching, training, evaluation and tests.
  `request.json` and `environment.json` record the actual run configuration.
- `notebooks/siglip_minimal_training_colab.ipynb`: the same training notebook with
  its caption-bearing metadata moved to an external file. Training logic is
  unchanged. See the required inputs below.
- `inference/`: small inference files extracted from the actual submitted ZIP.
- `export/` and `notebooks/siglip_minimal_submission_export_colab.ipynb`: export
  and offline verification helpers for these exact selected checkpoints.
- `results/`: aggregated local comparisons and the participant-reported public result.

## Model and training

The frozen encoder is `google/siglip-base-patch16-224`, pinned to revision
`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. Every frame is retained.
The original whole-description 64-token cosine score is retained. The correction
path encodes all sentence spans, splitting long spans into ordered token windows,
and aligns each span to all frames using similarity weights. A small MLP and
one sentence Transformer layer produce a bounded correction of up to 0.1.
This E0 model has no frame Transformer.

Three independently trained heads (seeds 2026, 2027, 2028) share one frozen encoder
at inference. Their final scores are averaged equally. Selected checkpoint epochs
are **4, 3, 4**; these are `best.pt`, not the last training epoch.

The fixed split is 3,336 training and 834 validation episodes. Each training row
uses one official positive and four negatives: two random original training
captions, one minimal edit, and one similar-episode training caption. The minimal
edit replaces the existing lexical slot; unavailable/colliding edits fall back
to that original slot. Rules cover color, shape, nouns, action/state opposites and
left/right, with training-vocabulary and ambiguity checks. They remain weak
negative assumptions, not visually verified false statements. The original rule
tables and implementation are preserved for inspection. Qwen event additions
were **not used** for this submitted model.

Epoch selection uses the existing team validation reward, then random validation
reward and team loss. The newer minimal-edit validation is final comparison only.
No teacher predictions are training labels. Each test episode is scored
independently; no test-time learning, memory update or centrality heuristic is used.

## Reproduce on free Colab T4

Open the training notebook and provide these private inputs in Drive:

1. `/content/drive/MyDrive/inthon2026_train.zip` — organizer-provided training data.
2. `/content/drive/MyDrive/portable_models.json` — the exact metadata sidecar from
   the original notebook/runtime. Its SHA256 is
   `57250ff11b7f487506f1ea497e54c32733a28ad4b784bbb11151ae9c0097e34c`.
   It contains validation candidate text and historical predictions, so it is
   intentionally not committed. In an existing completed runtime, copy
   `/content/inthon-siglip-minimal-training/portable_models.json` to that Drive path.
3. `/content/drive/MyDrive/factorial_weights.zip` — prior E0 checkpoints for the
   reused-control comparison, or set `TRAIN_CONTROL=True` to retrain that control.
   The metadata sidecar is still required with `TRAIN_CONTROL=True`.

Python 3.12.3 and the competition package versions are prepared by the notebook.
No training data is downloaded from this repository. Keep the generated training
data, schedules and provenance in the run folder for organizer inspection.
Running the training notebook is distinct from running the export notebook;
the latter reuses already-trained heads and never trains them.

The upload of this specific ZIP and its embedded weights was explicitly requested
by the participant. All other model/data files remain excluded from Git.

## Local results and limits

Training used the mixture described above. Evaluation used fixed random captions,
the team's lexical-plus-similar-caption mixture, and code-generated minimal edits.
These are development proxies, not estimates of leaderboard performance.

| Equal-score 3-seed ensemble | Previous E0 | Submitted minimal-edit E0 |
| --- | ---: | ---: |
| Random negatives, 834 episodes | 0.977218 | 0.978118 |
| Team negatives, 834 episodes | 0.918016 | 0.917866 |
| Minimal-edit 5-choice, 399 episodes | 0.549812 | 0.600877 |

The public score is 0.3630 despite local gains. Color/object changes improved more
than action direction, and event-addition negatives were not covered. Local
scores must not be presented as public performance.

The export helper checks the exact checkpoint hashes and recomputes raw-image
validation in a separate process with an empty model cache and blocked network.
The participant's resulting `offline_verification.json` is not included here
because it has not been supplied. The public report establishes successful
execution on those 573 episodes; it does not certify the duration of the entire
hidden test. The Python inference modules passed eight local CPU contract tests;
actual encoder/T4 checks happen in the Colab export workflow.
