# Validation

How we measure experiments. This is the team's own approach, not an organizer
requirement. The competition rules are in [brief.md](brief.md).

Status: scoring is in `team/rank_eval.py` and negatives are in `team/negatives.py`.
The kit's `evaluate.py` still reports BCE and pair accuracy on random negatives, and
checkpoint selection in `train.py` still uses BCE.

## Text-only leak check

`rank_eval.py` also trains a text-only classifier on generated negatives, to see
which ones can be spotted without the video. A score above about 0.42 is a leak, and
a model trained on that type may learn the text cue instead of the video. The
classifier is only a diagnostic: never use its outputs to select negatives or train.

| Set | Text-only score | Cause |
|---|---:|---|
| `random` | 0.370 | None found |
| `lexical` | 0.397 | None found after the hub correction in `team/lexical.py`; plain TF-IDF neighbours scored 0.485 |
| `similar_episode` | 0.398 | None found |
| `reorder` | 0.451 | Not identified; down from 0.820 after keeping the opening sentence first |
| `mixed` | 0.487 | Contains the other leaking types |
| `swapped_action` | 0.544 | Not identified |
| `added_action` | 0.995 | Negatives are always longer, so picking the shortest scores 1.000 |

Reordering can't produce four distinct negatives for every episode, so `reorder` covers
622 of the 834 validation episodes.

## Method

- Split by episode, using the kit's 80:20 split, and never train on validation
  episodes.
- Give each validation episode its correct description and 4 negatives, in a
  random order, to avoid favouring a position when scores tie.
- Generate negatives from code with a fixed seed, and reuse the same set across
  experiments so that scores are comparable.
- Score each episode by the rank of the correct description, then take the mean:

  | Rank of correct answer | 1st | 2nd | 3rd | 4th | 5th |
  |---|---:|---:|---:|---:|---:|
  | Episode score | 1 | 0.5 | 0.25 | 0.125 | 0 |

- Random guessing averages 0.375, so a useful model must beat that.
- Select checkpoints by this score rather than BCE.

## Trap 1: training and validating on the same negatives

A model tested on the kinds of negatives it trained on will look better than it
really is.

- Hold at least one negative type out of training, and keep it in validation.
- Report the score on seen and held-out types separately.
- Treat the held-out score as the better guide to generalization.

## Trap 2: tuning only to our own negatives

We can't see the test candidates beyond the slide examples (reversed action and
added action), and the test comes from a different benchmark, so our negatives
are a guess.

- Use the public leaderboard as the outside check on local gains.
- If local scores rise but the public score doesn't, assume our negatives don't
  resemble the real ones. Change the negatives before tuning the model further.
- Don't over-tune to the public score either. It covers only 30% of the test
  set, and the private 70% decides the final ranking.

## Known weakness: reversed or shuffled frames

Reversed or shuffled frames are a cheap way to check that a model learns order,
but some actions look much the same in reverse, so some of these labels will be
wrong. Don't rely on this as the only negative type.

## Experiment log

Record every experiment that gets a local score.

| Date and time (KST) | Experiment | Trained on | Held out | Local: seen | Local: held out | Public |
|---|---|---|---|---:|---:|---:|
| 2026-10-31 15:33 | Kit baseline: mean-pooled frames, concat MLP, BCE; RTX 5070 | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721 | Not submitted |
| 2026-10-31 16:22 | Same kit baseline retrained on a second machine (RTX 5070), scored with a first `lexical` set (plain TF-IDF top 4, since replaced). Other sets match the 15:33 run exactly | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode`, `lexical` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721, `lexical` 0.579 | Not submitted |
| 2026-10-31 17:20 | Same checkpoint as 16:22, scored with `lexical` from `team/lexical.py` (CSLS-corrected TF-IDF, 4 random picks from the nearest 20). Other sets unchanged | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode`, `lexical` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721, `lexical` 0.658 | Not submitted |
