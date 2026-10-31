# Findings and status (njunu17's line of work)

What this line of work found, what is ready to submit, and what is still open, as of 2026-11-01
06:15 KST. Every number here is also a row in [validation.md](validation.md), with its settings.
Local scores use our own validation candidates; public scores come from the leaderboard.

## Summary

- **Best public score so far: centrality alone, 0.5515.** It is text only and never looks at the
  video.
- Every video model scored near chance on public, and adding one to centrality lowered the score
  even when it only broke near-ties (0.5312 with w = 4).
- Pretrained SigLIP, used without training, didn't help either: ranked equally with centrality it
  scored 0.4682 publicly (see "Findings from SigLIP and image evidence" in
  [validation.md](validation.md)).
- A fine-tuned SmolVLM that only decides between candidates describing different scenes
  (`smolvlm_gate_v1.zip`) scored 0.5502, the same as centrality alone: it changed only 2 episodes,
  because test candidates rarely differ that much. Contrast training on one-word edits, used only
  where SmolVLM was confident, scored lower still (0.5382; 16 episodes better, 33 worse).
- Recommended final candidate: `central_submission.zip`.

## Lessons

1. **Text encoders hide one-word differences.** A one-word edit of a 60-word description keeps its
   MPNet embedding at cosine 0.983, so models that compare a sentence vector with the video can't
   tell near-identical candidates apart. SmolVLM, which reads the text token by token while seeing
   the frames, is the only permitted model without this bottleneck.
2. **What the test candidates look like (inferred, not measured).** Long descriptions with small
   edits: almost no pair is below 0.6 word similarity. Most negatives aren't simple edits of the
   correct text, or centrality would score about 0.99 instead of 0.55, so in most episodes only
   the video can pick the answer.
3. **Local image use didn't carry over.** Every video model that used the image locally (the
   wrong-video score dropped) lost on public, and the more a submission relied on the image, the
   lower it scored. The wrong-video check shows a model uses the image, not that it reads the test
   videos correctly.
4. **Combining with centrality only pays if the video signal is right on the test.** Weighted sums,
   near-tie breaking, a similarity gate and a confidence gate all scored at or below centrality.

## Public scores so far

| Submission | What it is | Public |
|---|---|---:|
| `final_rehearsal.zip` | H: temporal, cross-attention, flipping 0.5, 60 epochs | 0.3586 |
| `temporal_word_swap_v1.zip` | H + `word_swap` | 0.3968 |
| `central_submission.zip` | Text only: ranks candidates by centrality, no video, no training | **0.5515** |
| `q1_central05.zip` | Q1 + centrality, w = 0.5 (the model leads) | 0.4068 |
| `q2_central4.zip` | Q2 + centrality, w = 4 (centrality leads, the model breaks near-ties) | 0.5312 |
| `smolvlm_gate_v1.zip` | Centrality, with fine-tuned SmolVLM deciding pairs below 0.6 word similarity | 0.5502 |
| `smolvlm_confidence_v1.zip` | The gate, plus contrast-trained SmolVLM deciding pairs whose scores differ by 4 or more | 0.5382 |

Chance is 0.375. Centrality ranked the correct text first in 40.8% of episodes and last in 11.3%;
`smolvlm_gate_v1` in 40.7% and 11.5%.

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

## SmolVLM likelihood and the similarity gate

**Why earlier video models couldn't separate the candidates.** Changing one word of a 60-word
description leaves its MPNet embedding almost unchanged: cosine 0.983 on average between a
description and its `word_swap` edit (0.999 at the 90th percentile, 300 descriptions), against 0.59
for another episode's description. The rankers compare each candidate's text with the video
separately, so near-identical candidates look the same to them whatever negatives they train on.

**SmolVLM likelihood.** `team/smolvlm_scorer.py` scores a candidate by how likely SmolVLM-256M is to
write it after seeing the frames, minus `prior_weight` times how likely it is with no frames
(pointwise mutual information at weight 1). It reads the text word by word, so a changed word
changes the score. `team/train_smolvlm.py` fine-tunes it to write the organizers' descriptions from
the frames, using only the correct descriptions (no generated negatives), with frames left out on
20% of batches so that the same model learns the no-frame prior. Vision encoder frozen, 2 epochs,
float16 mixed precision.

Measured with `team/smolvlm_check.py` on 120 validation episodes. The SmolVLM model is trained on
correct descriptions only, so every set is held out. Scores with the wrong video show how much
comes from the text alone:

| Set (right / wrong video) | Zero-shot, w = 1 | Fine-tuned, w = 1 | Centrality |
|---|---|---|---:|
| `word_swap` | 0.357 / 0.330 | 0.408 / 0.320 | 0.993 |
| `qwen` | 0.448 / 0.401 | 0.378 / 0.328 | 1.000 |
| `swapped_action` | 0.634 / 0.360 | 0.698 / 0.259 | 0.963 |
| `similar_episode` | 0.796 / 0.403 | **0.894** / 0.386 | 0.405 |
| Action pairing (chance 0.5) | 0.525 | 0.508 | - |

- With w below 1 the scores rise, but the wrong-video scores rise with them: the gain is how
  natural the text is, not the video. At w = 1 the less natural edited negatives gain, which is the
  same failure as image evidence's 0.3370 on public.
- Image use depends on how different two candidates are. Fine-tuned, w = 1, the correct text beats
  a negative 0.922 of the time with the right video against 0.513 with the wrong one when their word
  similarity is below 0.5, but 0.580 against 0.498 above 0.95. Centrality is the reverse: strong on
  edits, chance on other scenes.

**The gate.** `team/submission_smolvlm_gate.py` compares every pair of candidates once: below a word
similarity of 0.6 the higher SmolVLM score wins, otherwise the more central candidate, and the
ranking is by wins. SmolVLM only runs if some pair is below 0.6. Locally (150 episodes) it keeps
centrality on edits (`word_swap` 0.995 both), recovers other scenes (`similar_episode` 0.891 against
0.394) and a constructed case where the four negatives are edits of another episode's description
(0.968 against 0.000), but loses a little on short candidates (`qwen` 0.957 against 0.993).

**Public: 0.5502, against centrality's 0.5515.** Compared episode by episode with centrality's
public CSV, only 2 of 573 episodes changed, both for the worse (episode 407 from 1st to 2nd, 425
from 3rd to 5th). So almost no test pair falls below 0.6 word similarity: the test negatives are
close edits, and the 11% of episodes where the correct text is least central aren't other-scene
negatives. What's left to gain is in telling near-identical candidates apart, which no video model
here does yet.

Raising the threshold would hand those near-identical pairs to SmolVLM. Its local pairwise accuracy
there is 0.58 above 0.95 similarity and 0.62 at 0.85-0.95, while centrality's public rank
distribution implies the correct text beats a given negative about 0.65 of the time. Since local
scores have overstated public ones every time, that trade looks like a loss, and it wasn't tried.

**What the test structure seems to be.** Simulated with one-word edits of the organizers'
descriptions (300 descriptions), centrality scores 0.997 when each negative is an edit of the
correct text (0.995 with three edits each), 0.078 for a chain of edits and 0.412 when all five
candidates are edits of a hidden original. Public 0.5515 (first 40.8%, last 11.3%) therefore means
most test negatives are not simple edits of the correct text; in most episodes the text alone
doesn't single it out. Locally, short one-sentence candidates (`qwen`) have a pair below 0.6
similarity in 77% of episodes and sentence swaps in 96%, against 0% for one-word edits of long
descriptions, so the gate's 2 changed episodes also suggest long candidates with small edits. This
is inferred from simulations, not measured on the test.

**Contrast training (`contrast_v1`).** `team/train_smolvlm_contrast.py` continues from `ft_v1` for
one epoch with the correct description T, a one-word edit T' (`word_swap`, table from the training
split) and another batch episode's frames V2, asking for
[log p(T|V) - log p(T'|V)] - [log p(T|V2) - log p(T'|V2)] > 0, so text-only preferences cancel.
On 120 validation episodes, pairwise at w = 1 with the right / wrong video: `word_swap` (the trained
type) 0.624 / 0.452 against `ft_v1`'s 0.580 / 0.498; held out, `qwen` 0.527 / 0.446,
`added_action` 0.696 / 0.469, `similar_episode` 0.938 / 0.540; action pairing 0.525. Pairs whose
scores differ by 4 or more were right 0.96 of the time (6% of `word_swap` pairs).

**Confidence gate (`smolvlm_confidence_v1`).** The gate above, plus SmolVLM deciding any pair whose
scores differ by at least 4. Locally it kept one-word edits (`word_swap` 0.988 against 0.995) and
raised a constructed case where the negatives share one edit plus their own, so the correct text
isn't central, from 0.387 to 0.499 (wrong video 0.397). **Public: 0.5382**, below centrality:
49 of 573 episodes changed, 16 better and 33 worse (sign test p = 0.021), most often the correct
text moving from 1st to 2nd (13). Even SmolVLM's confident decisions were wrong about twice as
often as right on the test, as with every earlier video model.

## Ready to submit (on njunu17's machine, not in git)

| ZIP | Contents | Public |
|---|---|---:|
| `cuda/central_submission.zip` | Centrality alone | 0.5515 |
| `cuda/q2_central4.zip` | Q2 + centrality, w = 4 | 0.5312 |
| `cuda/q1_central05.zip` | Q1 + centrality, w = 0.5 | 0.4068 |
| `cuda/q1_h_qwen50.zip` | Q1 alone | Not submitted |
| `cuda/smolvlm_gate_v1.zip` | Centrality + fine-tuned SmolVLM gate, 773 MiB | 0.5502 |
| `cuda/smolvlm_confidence_v1.zip` | Confidence gate with `contrast_v1`, 773 MiB | 0.5382 |

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
   SmolVLM, used only between different-looking candidates, changed nothing (0.5502), and
   contrast-trained SmolVLM, used where it was confident, lost (0.5382).
2. Improving centrality itself. Simulations suggest most test negatives aren't edits of the correct
   text, so a text rule has little left to find; only public can test changes here.
3. The final model has to be produced on Colab. Centrality alone needs no training, so it only
   needs `export_central.py` and `package.py`.
