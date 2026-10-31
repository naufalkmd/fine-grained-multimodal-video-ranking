# Findings and status (njunu17's line of work)

What this line of work found, what is ready to submit, and what is still open, as of 2026-11-01
01:30 KST. Every number here is also a row in [validation.md](validation.md), with its settings.
Local scores use our own validation candidates; public scores come from the leaderboard.

## Summary

- **Best public score so far: centrality alone, 0.5515.** It is text only and never looks at the
  video.
- Every video model scored near chance on public, and adding one to centrality lowered the score
  even when it only broke near-ties (0.5312 with w = 4).
- Pretrained SigLIP, used without training, didn't help either: ranked equally with centrality it
  scored 0.4682 publicly (see "Findings from SigLIP and image evidence" in
  [validation.md](validation.md)).
- Recommended final candidate: `central_submission.zip`, until some video signal beats it on public.

## Public scores so far

| Submission | What it is | Public |
|---|---|---:|
| `final_rehearsal.zip` | H: temporal, cross-attention, flipping 0.5, 60 epochs | 0.3586 |
| `temporal_word_swap_v1.zip` | H + `word_swap` | 0.3968 |
| `central_submission.zip` | Text only: ranks candidates by centrality, no video, no training | **0.5515** |
| `q1_central05.zip` | Q1 + centrality, w = 0.5 (the model leads) | 0.4068 |
| `q2_central4.zip` | Q2 + centrality, w = 4 (centrality leads, the model breaks near-ties) | 0.5312 |

Chance is 0.375. Centrality ranked the correct text first in 40.8% of episodes and last in 11.3%.

## What the test candidates seem to be

The slide example has five candidates that describe the same scene in one short sentence and differ
in one place: a reversed verb ("buttoning" / "unbuttoning") or an added action ("and then sits
down"). Our earlier validation sets mostly use other videos' descriptions as negatives, so a model
can pass them by matching objects. That skill is useless when all five candidates share the scene,
which explains why models scoring about 0.8 locally sit at chance on the public set.

The test candidates may be long rather than one short sentence. The team lead found that the
organizers wrote the training positives from the frames at 42-75 words and started negatives the
same way, so test candidates are probably about 60 words (unconfirmed; see validation.md). If so,
the Qwen sets, with a median of 12 words, are shorter than the test.

## Centrality: a text-only cue

When every negative is one edit of the correct text, the correct text is the one most similar to
all the others. `team/submission_central.py` ranks by each candidate's mean word-level similarity
to the other candidates. It uses only the episode's own input, so it is within the rules, but it
never looks at the video.

| Validation set | How negatives are made | Centrality |
|---|---|---:|
| `added_action`, `mixed`, `qwen` | Edits of the correct text | 0.997-1.000 |
| `swapped_action` | One sentence replaced | 0.971 |
| `similar_episode`, `random` | Other videos' descriptions | 0.37-0.39 |

Public came out at 0.5515, far above the video models but well below 0.99. So many test
negatives are edits of the correct text, but not all: in 11% of episodes the correct text was the
least central candidate. Local sets overstate centrality.

## Test-like validation set built with Qwen

`team/qwen_negatives.py` asks a local Qwen3-4B-Instruct (float16) to rewrite each description as
one short sentence and make four single-change negatives: reversed action, added action, swapped
order and changed detail. Only code rules filter the output; the prompt has no example sentences.
`team/qwen_candidates.py` turns the validation split into the `qwen` set (323 complete items).

Video-only models trained without Qwen data score on it about what they score on public (T2 0.379,
H + `word_swap` 0.390 against public 0.3968). It overstates models trained on Qwen data, because
they can pick up Qwen's wording, and it overstates centrality (0.997 locally against 0.5515).

## Measuring whether a model reads the video

`team/action_pairing.py` checks whether reversing the frames flips a model's choice between a short
positive and its reversed-action sentence ("folds" / "unfolds"). Wording alone cancels out, so
chance is exactly 0.5 and only reading the action's direction from the video can raise it. It is a
better local check of video use than the `qwen` set, which text cues inflate.

| Model | Action pairing (n = 566) |
|---|---:|
| H + `word_swap` | 0.530 |
| Q1 | 0.544 |
| Q2 | **0.640** |

## Training on Qwen data (Q1, Q2)

Q1: H settings, plus Qwen's short positive and single-change negatives on half of the training
episodes (`--qwen-candidates`, `--qwen-probability 0.5`). Flipping uses the swapped-order sentence.

Q2: Q1 plus `--qwen-reverse-actions`. On reversed frames the reversed-action sentence becomes the
correct text and the positive a negative, because playing "folds" backwards looks like "unfolds".
Each such sentence is then right on one frame direction and wrong on the other, so its wording
can't give it away.

| | H + `word_swap` | Q1 | Q2 |
|---|---:|---:|---:|
| `qwen` set | 0.390 | 0.650 | 0.622 |
| `qwen` set with the wrong video | - | 0.500 | 0.456 |
| Action pairing | 0.530 | 0.544 | **0.640** |
| Clean score (`similar_episode`, `lexical`) | 0.775 | 0.790 | 0.800 |
| Order pairing | 0.730 | 0.639 | 0.630 |

Q2 relies less on wording and more on the video than Q1 locally. Both still peaked at the last of
60 epochs.

## Model plus centrality, and why it failed on public

`team/submission_ranker.py` takes `central_weight` from `inference_config.json`: each candidate
scores its standardized logit plus `central_weight` times its standardized centrality. With 0 it
behaves exactly as before. `team/export_ranker.py --central-weight` sets it.

- w = 0.5 with Q1 (0.4068): the model led, and its near-chance public accuracy overrode
  centrality's correct choices.
- w = 4 with Q2 (0.5312): against centrality alone, 118 of 573 episodes changed, 45 for the better
  and 73 for the worse (sign test p = 0.013). The largest loss was the correct text moving from
  1st to 2nd (31 episodes, against 14 moving from 2nd to 1st). Even on near-ties, the model chose
  the wrong candidate about twice as often as the right one.

What Q2 learned on PerceptionTest frames doesn't carry over to the test benchmark's videos. Frozen
DINOv2 frame summaries may also lack the hand-motion detail that separates "buttoning" from
"unbuttoning".

## Ready to submit (on njunu17's machine, not in git)

| ZIP | Contents | Public |
|---|---|---:|
| `cuda/central_submission.zip` | Centrality alone | 0.5515 |
| `cuda/q2_central4.zip` | Q2 + centrality, w = 4 | 0.5312 |
| `cuda/q1_central05.zip` | Q1 + centrality, w = 0.5 | 0.4068 |
| `cuda/q1_h_qwen50.zip` | Q1 alone | Not submitted |

All passed `validate.py`.

## Qwen data and T4 reproducibility

- Data, shared on Drive rather than git: `candidates_v3.jsonl` (4,170 episodes, 5.9 MB) and
  `candidates_qwen.json` (validation candidates with the `qwen` set). Commands are in
  [team/README.md](../team/README.md).
- Measured on an RTX 5070 in float16: 4.02 B parameters, peak 8.6 GiB allocated at batch 16, so it
  fits a T4 (16 GB). The installed torch includes sm_75 (T4), and attention runs as SDPA, which T4
  supports. 4,170 episodes took 35 minutes; a T4 should take about 2 to 4 times longer.
- Not yet checked on an actual T4. A 20-episode run on Colab would confirm the speed.

## Earlier work, for reference

- `lexical` negatives with a hub correction (CSLS): text-only leak fell from 0.485 to 0.397.
- `train_negatives.py` and `train_ranker.py --mix --k`: fixed counts per negative type.
- On the kit model, listwise beat BCE on every set, and the mix of types mattered more than k.
- T2 (temporal, k = 8, no `swapped_action`): `added_action` fell to 0.687 and the model learned a
  `reorder` text cue (0.644 with the wrong video). Keep `swapped_action` in training.

## Open questions

1. Whether any video signal helps on public. Models trained on our data haven't, and neither has
   SigLIP zero-shot at equal rank weight (0.4682). The team lead's simulation predicts a video
   signal only helps once it scores about 0.45 or more alone on public. Compare any new attempt's
   per-episode CSV against `central_submission.zip`.
2. Improving centrality itself. The 11% of episodes where the correct text is least central suggest
   some test negatives are edits of each other rather than of the correct text. Only public can
   test changes here.
3. The final model has to be produced on Colab. Centrality alone needs no training, so it only
   needs `export_central.py` and `package.py`.
