# Findings and status (njunu17's line of work)

What this line of work found, what is ready to submit, and what is still open, as of 2026-11-01
00:30 KST. Every number here is also a row in [validation.md](validation.md), with its settings.
Local scores use our own validation candidates; public scores come from the leaderboard.

## Public scores so far

| Submission | What it is | Public |
|---|---|---:|
| `final_rehearsal.zip` | H: temporal, cross-attention, flipping 0.5, 60 epochs | 0.3586 |
| `temporal_word_swap_v1.zip` | H + `word_swap` | 0.3968 |
| `central_submission.zip` | Text only: ranks candidates by centrality, no video, no training | **0.5515** |

Chance is 0.375. In the first 100 rows of the centrality submission's per-episode results, the
correct text was ranked first in 39% of episodes and last in 14%.

## What the test candidates seem to be

The slide example has five candidates that describe the same scene in one short sentence and differ
in one place: a reversed verb ("buttoning" / "unbuttoning") or an added action ("and then sits
down"). Our earlier validation sets mostly use other videos' descriptions as negatives, so a model
can pass them by matching objects. That skill is useless when all five candidates share the scene,
which explains why models scoring about 0.8 locally sit at chance on the public set.

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
negatives are edits of the correct text, but not all: in 14% of episodes the correct text was the
least central candidate. Local sets overstate centrality, so its weight has to be set on public.

## Test-like validation set built with Qwen

`team/qwen_negatives.py` asks a local Qwen3-4B-Instruct (float16) to rewrite each description as
one short sentence and make four single-change negatives: reversed action, added action, swapped
order and changed detail. Only code rules filter the output; the prompt has no example sentences.
`team/qwen_candidates.py` turns the validation split into the `qwen` set (323 complete items).

Video-only models score on it about what they score on public, which makes it a useful proxy for
them:

| Model | `qwen` set | Public |
|---|---:|---:|
| T2 (no Qwen data) | 0.379 | - |
| H + `word_swap` | 0.390 | 0.3968 |

It doesn't match for centrality (0.997 locally against 0.5515 publicly), as explained above.

## Training on Qwen data (Q1)

H settings, plus Qwen's short positive and single-change negatives on half of the training
episodes (`--qwen-candidates`, `--qwen-probability 0.5`). Flipping uses the swapped-order sentence.

| | H + `word_swap` | Q1 |
|---|---:|---:|
| `qwen` set | 0.390 | **0.650** |
| `qwen` set with the wrong video | - | 0.500 |
| Clean score (`similar_episode`, `lexical`) | 0.775 | 0.790 |
| Order pairing | 0.730 | 0.639 |
| `added_action` (held out) | 0.839 | 0.799 |

Part of the gain is text cues (0.500 with the wrong video), and about 0.15 needs the video. Order
pairing dropped from H's 0.733. The best epoch was the last of 60, so it may still be rising.

## Model plus centrality

`team/submission_ranker.py` now takes `central_weight` from `inference_config.json`: each candidate
scores its standardized logit plus `central_weight` times its standardized centrality. With 0 it
behaves exactly as before. `team/export_ranker.py --central-weight` sets it. Scores below come from
the exported folder itself:

| Validation set | Q1 alone | Q1 + centrality, w = 0.5 | Centrality alone |
|---|---:|---:|---:|
| `qwen` | 0.650 | 0.873 | 0.997 |
| `similar_episode` | 0.819 | 0.774 | 0.388 |
| `lexical` | 0.760 | 0.775 | 0.550 |
| `added_action` | 0.800 | 0.949 | 1.000 |

w = 0.5 rather than 1 because local sets overstate centrality. Q1 combines better than H +
`word_swap` at every weight tried (0.5, 1, 2, 4).

## Ready to submit (on njunu17's machine, not in git)

| ZIP | Contents | Validated |
|---|---|---|
| `cuda/q1_central05.zip` | Q1 + centrality, w = 0.5 | `validate.py` passed |
| `cuda/q1_h_qwen50.zip` | Q1 alone | `validate.py` passed |
| `cuda/central_submission.zip` | Centrality alone (submitted) | `validate.py` passed |

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

1. What weight to give centrality. Only public can settle it; w = 0.5 and w = 1 are the obvious
   pair to compare.
2. Whether Q1's gain holds on public. Submitting `q1_central05.zip` tests the combination; Q1
   alone would isolate the Qwen data.
3. The final model has to be produced on Colab, so Q1's training (with the Drive data) needs to go
   into `final_colab.ipynb` if it is chosen.
4. SigLIP is unused. Test sentences are short, so its 64-token limit no longer gets in the way; a
   zero-shot check on the `qwen` set would show whether it is worth adding.
