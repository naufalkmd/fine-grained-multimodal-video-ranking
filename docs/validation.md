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

## Ablation results

Each step adds one change to the step above it, except G and H, which each change F. All runs use seed 2026 unless noted and the same fixed candidates, and hold `added_action` out of training. These scores use our own generated negatives, so they rank these models against each other but don't predict the leaderboard.

- **Clean rank score:** mean of `similar_episode` and `lexical`, the two sets with no detected text leak. Wrong-video scores on both sit at about 0.36-0.39, near the 0.375 chance level.
- **Order pairing:** whether reversing the frames changes which text order the model prefers. Text cues cancel out, and chance is 0.5.

| Run | Change | Epochs | `similar_episode` | `lexical` | Clean rank score | Order pairing |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Kit model; `lexical` is from a second machine's run | 19 | 0.721 | 0.658 | 0.690 | About 0.5 |
| A | Ranking loss; random negatives | 50 | 0.789 | 0.734 | 0.761 | 0.500 |
| B | A + hard negatives | 50 | 0.769 | 0.690 | 0.730 | 0.500 |
| C | B + temporal layers | 26 | 0.793 | 0.740 | 0.766 | 0.569 |
| D | C + cross-attention | 15 | 0.800 | 0.731 | 0.765 | 0.534 |
| E | D + reverse loss | 22 | 0.796 | 0.725 | 0.760 | 0.593 |
| F | E + flipping | 46 | 0.809 | 0.754 | 0.781 | 0.668 |
| G | F without cross-attention | 60 | 0.829 | 0.772 | 0.800 | 0.658 |
| H | F + `lexical` negatives | 50 | 0.819 | 0.762 | 0.791 | 0.728 |
| H, seed 2027 | Repeat of H | 60 | 0.815 | 0.769 | 0.792 | 0.690 |
| H, seed 2028 | Repeat of H | 60 | 0.811 | 0.755 | 0.783 | 0.683 |
| G + `lexical` | G trained on `lexical` too | 60 | 0.826 | 0.777 | 0.802 | 0.631 |

What this shows so far:

- Hard negatives alone lower the clean score (A to B), and mostly teach text cues.
- Flipping gives the largest order gain, and removes most of the `reorder` text cue.
- Cross-attention raises the order signal: H averages 0.700 order pairing over three seeds, against 0.631 for G + `lexical` (one seed). It costs about 0.013 in clean rank score. `lexical` barely changes G.
- With flipping, order pairing stays near 0.5 until about epoch 25. Patience 8 can end a run before that, so don't use it with flipping. C, D and E also stopped early, so they may be understated.
- Baseline to A also changes the learning rate, batch size and number of negatives, so that gain can't be credited to the ranking loss alone.

Not covered: nothing here tests single-word differences such as "buttoning" and "unbuttoning", and no model has a public score yet.

## Experiment log

Record every experiment that gets a local score.

Submission names used by the team:

| Name | ZIP | Model |
|---|---|---|
| `b` | `cuda/sample_submission.zip` | Kit baseline |
| `h-v1` | `cuda/final_rehearsal.zip` | H, frozen selection rule, 60 epochs, trained locally |
| `temporal_word_swap_v1` | `cuda/temporal_word_swap_v1.zip` | H + `word_swap` negatives, 60 epochs, trained locally |

| Date and time (KST) | Experiment | Trained on | Held out | Local: seen | Local: held out | Public |
|---|---|---|---|---:|---:|---:|
| 2026-10-31 15:33 | Kit baseline: mean-pooled frames, concat MLP, BCE; RTX 5070 | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721 | **0.3469** (573 episodes, 0 errors, 57.72 s). Submitted as `b`. Other teams on the leaderboard have 0.3573 and 0.3497. |
| 2026-10-31 15:54 | `mean_pool`: same architecture, listwise loss, generated negatives | `random`, `reorder`, `similar_episode`, `swapped_action` | `added_action` | `random` 0.922, `reorder` 0.516, `similar_episode` 0.748, `swapped_action` 0.872 | `added_action` 0.780; `mixed` 0.741 includes it. Wrong video: `reorder` 0.496 | Not submitted |
| 2026-10-31 16:03 | `temporal`: temporal transformer, cross-attention, reverse loss | `random`, `reorder`, `similar_episode`, `swapped_action` | `added_action` | `random` 0.939, `reorder` 0.731, `similar_episode` 0.806, `swapped_action` 0.903 | `added_action` 0.813; `mixed` 0.852 includes it. Wrong video: `reorder` 0.692. Order pairing 0.54 | Not submitted |
| 2026-10-31 16:20 | `temporal` with frames and text flipped together on half of qualifying episodes | `random`, `reorder`, `similar_episode`, `swapped_action` | `added_action` | `random` 0.947, `reorder` 0.286, `similar_episode` 0.818, `swapped_action` 0.896 | `added_action` 0.871; `mixed` 0.599 includes it. Wrong video: `reorder` 0.319. Order pairing 0.595 | Not submitted |
| 2026-10-31 16:22 | Same kit baseline retrained on a second machine (RTX 5070), scored with a first `lexical` set (plain TF-IDF top 4, since replaced). Other sets match the 15:33 run exactly | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode`, `lexical` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721, `lexical` 0.579 | Not submitted |
| 2026-10-31 17:20 | Same checkpoint as 16:22, scored with `lexical` from `team/lexical.py` (CSLS-corrected TF-IDF, 4 random picks from the nearest 20). Other sets unchanged | `random` | `reorder`, `added_action`, `swapped_action`, `similar_episode`, `lexical` | 0.927 (`random`) | 0.552 (`mixed`); `reorder` 0.360, `added_action` 0.688, `swapped_action` 0.692, `similar_episode` 0.721, `lexical` 0.658 | Not submitted |
| 2026-10-31 17:50 | `team/train_rank.py`, kit model and features, BCE with the positive weighted k times, k=4, checkpoint picked on trained types; best was the last of 20 epochs, still rising. Candidates: `candidates_lexical_csls.json` | `lexical` 2, `similar_episode` 2 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.728, `similar_episode` 0.781 | `mixed` 0.564; `reorder` 0.389, `added_action` 0.589, `swapped_action` 0.632, `random` 0.895 | Not submitted |
| 2026-10-31 17:51 | `team/train_rank.py`, kit model and features, listwise softmax cross-entropy, k=4, checkpoint picked on trained types; best was the last of 20 epochs, still rising. Candidates: `candidates_lexical_csls.json` | `lexical` 2, `similar_episode` 2 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.762, `similar_episode` 0.809 | `mixed` 0.574; `reorder` 0.391, `added_action` 0.612, `swapped_action` 0.659, `random` 0.909 | Not submitted |
| 2026-10-31 18:14 | Ablation A: `mean_pool`, ranking loss. One seed; 50-epoch cap; same selection rule | `random` | All other types | `random` 0.952, `reorder` 0.391, `similar_episode` 0.789, `swapped_action` 0.715 | `added_action` 0.670; `mixed` 0.595. Wrong video: `reorder` 0.373. Order pairing 0.500 | Not submitted |
| 2026-10-31 18:14 | Ablation B: A + hard negatives. One seed; 50-epoch cap; same selection rule | `random`, `reorder`, `similar_episode`, `swapped_action` | `added_action` | `random` 0.922, `reorder` 0.558, `similar_episode` 0.769, `swapped_action` 0.876 | `added_action` 0.760; `mixed` 0.766. Wrong video: `reorder` 0.525. Order pairing 0.500 | Not submitted |
| 2026-10-31 18:14 | Ablation C: B + temporal layers, text averaged. One seed; 50-epoch cap; same selection rule | same as B | `added_action` | `random` 0.929, `reorder` 0.615, `similar_episode` 0.793, `swapped_action` 0.892 | `added_action` 0.801; `mixed` 0.809. Wrong video: `reorder` 0.555. Order pairing 0.569 | Not submitted |
| 2026-10-31 18:14 | Ablation D: C + cross-attention. One seed; 50-epoch cap; same selection rule | same as B | `added_action` | `random` 0.919, `reorder` 0.675, `similar_episode` 0.800, `swapped_action` 0.904 | `added_action` 0.813; `mixed` 0.827. Wrong video: `reorder` 0.632. Order pairing 0.534 | Not submitted |
| 2026-10-31 18:14 | Ablation E: D + reverse loss. One seed; 50-epoch cap; same selection rule | same as B | `added_action` | `random` 0.922, `reorder` 0.693, `similar_episode` 0.796, `swapped_action` 0.911 | `added_action` 0.846; `mixed` 0.838. Wrong video: `reorder` 0.684. Order pairing 0.593 | Not submitted |
| 2026-10-31 18:14 | Ablation F: E + flipping. One seed; 50-epoch cap; same selection rule | same as B | `added_action` | `random` 0.932, `reorder` 0.417, `similar_episode` 0.809, `swapped_action` 0.907 | `added_action` 0.853; `mixed` 0.679. Wrong video: `reorder` 0.384. Order pairing 0.668. `lexical` 0.754 | Not submitted |
| 2026-10-31 18:22 | Ablation G: F without cross-attention. One seed; 50-epoch cap; same selection rule | same as B | `added_action`, `lexical` | `random` 0.939, `reorder` 0.327, `similar_episode` 0.817, `swapped_action` 0.866 | `added_action` 0.809; `lexical` 0.762; `mixed` 0.605. Wrong video: `reorder` 0.312. Order pairing 0.538 | Not submitted |
| 2026-10-31 18:43 | Ablation H: F + `lexical` negatives. One seed; best epoch was the 50-epoch cap; same selection rule | same as B + `lexical` | `added_action` | `random` 0.934, `reorder` 0.502, `similar_episode` 0.819, `swapped_action` 0.871, `lexical` 0.762 | `added_action` 0.761; `mixed` 0.723. Wrong video: `reorder` 0.456. Order pairing 0.728 | Not submitted |
| 2026-10-31 18:50 | H with seed 2027. Early stopping ended it at epoch 15, before order pairing starts rising (around epoch 25 in F and H), so it isn't a fair repeat. G (16 epochs) was cut off the same way | same as H | `added_action` | `random` 0.925, `reorder` 0.273, `similar_episode` 0.805, `swapped_action` 0.849, `lexical` 0.757 | `added_action` 0.832; `mixed` 0.582. Wrong video: `reorder` 0.299. Order pairing 0.545 | Not submitted |
| 2026-10-31 19:15 | H with seed 2027, 60 fixed epochs, no early stopping | same as H | `added_action` | `similar_episode` 0.815, `lexical` 0.769 | `added_action` 0.812. Wrong video: `added_action` 0.498. Order pairing 0.690 | Not submitted |
| 2026-10-31 19:19 | Grid `k2_l1s1`: `team/train_rank.py` listwise, kit model, k=2, up to 40 epochs, best epoch 40. Candidates: `candidates_lexical_csls.json` | `lexical` 1, `similar_episode` 1 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.775, `similar_episode` 0.821 | `mixed` 0.584; `reorder` 0.403, `added_action` 0.590, `swapped_action` 0.676, `random` 0.914 | Not submitted |
| 2026-10-31 19:20 | Grid `k4_l4`: `team/train_rank.py` listwise, kit model, k=4, up to 40 epochs, best epoch 40. Candidates: `candidates_lexical_csls.json` | `lexical` 4 | `reorder`, `added_action`, `swapped_action`, `similar_episode`, `random` | `lexical` 0.778 | `mixed` 0.589; `reorder` 0.427, `added_action` 0.588, `swapped_action` 0.656, `similar_episode` 0.812, `random` 0.898 | Not submitted |
| 2026-10-31 19:21 | Grid `k4_s4`: `team/train_rank.py` listwise, kit model, k=4, up to 40 epochs, best epoch 40. Candidates: `candidates_lexical_csls.json` | `similar_episode` 4 | `reorder`, `added_action`, `swapped_action`, `lexical`, `random` | `similar_episode` 0.834 | `mixed` 0.576; `reorder` 0.394, `added_action` 0.564, `swapped_action` 0.680, `lexical` 0.777, `random` 0.922 | Not submitted |
| 2026-10-31 19:24 | Grid `k4_l3s1`: `team/train_rank.py` listwise, kit model, k=4, up to 40 epochs, best epoch 37. Candidates: `candidates_lexical_csls.json` | `lexical` 3, `similar_episode` 1 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.777, `similar_episode` 0.818 | `mixed` 0.590; `reorder` 0.419, `added_action` 0.590, `swapped_action` 0.662, `random` 0.908 | Not submitted |
| 2026-10-31 19:26 | Grid `k4_l2s2`: `team/train_rank.py` listwise, kit model, k=4, up to 40 epochs, best epoch 36. Candidates: `candidates_lexical_csls.json` | `lexical` 2, `similar_episode` 2 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.795, `similar_episode` 0.828 | `mixed` 0.602; `reorder` 0.415, `added_action` 0.594, `swapped_action` 0.669, `random` 0.919 | Not submitted |
| 2026-10-31 19:27 | Grid `k4_l1s3`: `team/train_rank.py` listwise, kit model, k=4, up to 40 epochs, best epoch 36. Candidates: `candidates_lexical_csls.json` | `lexical` 1, `similar_episode` 3 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.782, `similar_episode` 0.828 | `mixed` 0.598; `reorder` 0.412, `added_action` 0.593, `swapped_action` 0.682, `random` 0.915 | Not submitted |
| 2026-10-31 19:30 | Grid `k8_l4s4`: `team/train_rank.py` listwise, kit model, k=8, up to 40 epochs, best epoch 36. Candidates: `candidates_lexical_csls.json` | `lexical` 4, `similar_episode` 4 | `reorder`, `added_action`, `swapped_action`, `random` | `lexical` 0.792, `similar_episode` 0.840 | `mixed` 0.586; `reorder` 0.406, `added_action` 0.575, `swapped_action` 0.664, `random` 0.926 | Not submitted |
| 2026-10-31 19:33 | Grid `k8_l3s3r2`: `team/train_rank.py` listwise, kit model, k=8, up to 40 epochs, best epoch 37. Candidates: `candidates_lexical_csls.json` | `lexical` 3, `similar_episode` 3, `random` 2 | `reorder`, `added_action`, `swapped_action` | `lexical` 0.801, `similar_episode` 0.846, `random` 0.941 | `mixed` 0.605; `reorder` 0.422, `added_action` 0.612, `swapped_action` 0.675 | Not submitted |
| 2026-10-31 19:35 | G, seed 2026, 60 fixed epochs, no early stopping | same as B | `added_action`, `lexical` | `similar_episode` 0.829 | `added_action` 0.801; `lexical` 0.772. Wrong video: `added_action` 0.513. Order pairing 0.658 | Not submitted |
| 2026-10-31 20:01 | H with seed 2028, 60 fixed epochs, no early stopping | same as H | `added_action` | `similar_episode` 0.811, `lexical` 0.755 | `added_action` 0.759. Wrong video: `added_action` 0.447. Order pairing 0.683 | Not submitted |
| 2026-10-31 20:22 | G + `lexical`, seed 2026, 60 fixed epochs, no early stopping | same as H | `added_action` | `similar_episode` 0.826, `lexical` 0.777 | `added_action` 0.785. Wrong video: `added_action` 0.486. Order pairing 0.631 | Not submitted |
| 2026-10-31 20:47 | Rehearsal: H with the frozen selection rule, 60 fixed epochs, trained locally. `cuda/final_rehearsal.zip` | same as H | `added_action` | `similar_episode` 0.821, `lexical` 0.788 | `added_action` 0.765. Wrong video: `added_action` 0.444. Order pairing 0.733 | **0.3586** (573 episodes, 0 errors, 62.35 s) |
| 2026-10-31 20:52 | T2: G settings (temporal, no cross-attention, flipping 0.5, 60 fixed epochs, seed 2026) with `--k 8 --mix lexical=3,similar_episode=2,random=1,reorder=2`; best epoch 50, selection 0.700. Candidates: `candidates_lexical_csls.json` | `lexical` 3, `similar_episode` 2, `random` 1, `reorder` 2 | `added_action`, `swapped_action` | `similar_episode` 0.812, `lexical` 0.761 (clean 0.786), `random` 0.902, `reorder` 0.690 | `added_action` 0.687, `swapped_action` 0.624; `mixed` 0.606. Wrong video: `reorder` 0.644, others 0.34-0.41. Order pairing 0.643 | Not submitted |
| 2026-10-31 21:18 | H with `lexical` 3, `similar_episode` 2, `random` 1, `reorder` 2, 8 negatives, 60 fixed epochs | `lexical`, `random`, `reorder`, `similar_episode` | `added_action`, `swapped_action` | `similar_episode` 0.838, `lexical` 0.790 | `added_action` 0.600; `swapped_action` 0.625. Wrong video: `added_action` 0.327. Order pairing 0.605 | Not submitted |
| 2026-10-31 22:36 | H + `word_swap`, 60 fixed epochs, frozen rule, trained locally. `cuda/temporal_word_swap_v1.zip` | same as H + `word_swap` | `added_action` | `similar_episode` 0.804, `lexical` 0.745, `word_swap` 0.555 | `added_action` 0.839. Wrong video: `word_swap` 0.439, `added_action` 0.560. Order pairing 0.730 | **0.3968** (573 episodes, 0 errors, 64.21 s) |
| 2026-10-31 23:20 | Text-only `team/submission_central.py`: ranks candidates by mean word-level similarity to the other candidates; no video, no training. Candidates: `candidates_qwen.json` (adds `qwen`: Qwen3-4B short positive + 4 single-change negatives, 323 complete validation items) | None | All types | - | `added_action` 1.000, `mixed` 0.999, `qwen` 0.997, `swapped_action` 0.971, `lexical` 0.550, `reorder` 0.506, `similar_episode` 0.388, `random` 0.370 | Not submitted |
| 2026-10-31 23:27 | Q1: H settings + `--qwen-candidates candidates_v3.jsonl --qwen-probability 0.5` (Qwen3-4B short positive + single-change negatives on half the episodes), 60 fixed epochs, frozen rule; best epoch 60. `cuda/q1_h_qwen50.zip` | same as H + `qwen` | `added_action` | `similar_episode` 0.819, `lexical` 0.761, `qwen` 0.650 | `added_action` 0.799. Wrong video: `qwen` 0.500, `added_action` 0.426. Order pairing 0.639 | Not submitted |
| 2026-10-31 23:35 | T2 checkpoint (no Qwen data) rescored on `candidates_qwen.json` for reference | as T2 | as T2 | - | `qwen` 0.379. Wrong video: `qwen` 0.338 | Not submitted |
| 2026-11-01 00:05 | Team lead's H + `word_swap` weights (`cuda/requested_data`, public 0.3968) rescored on `candidates_qwen.json`, with and without centrality added (z-scored per episode, model + w x centrality) | as H + `word_swap` | `added_action` | - | `qwen` 0.390 alone; +centrality w=1: `qwen` 0.668, `similar_episode` 0.689. Q1 for comparison: `qwen` 0.650 alone; w=1: `qwen` 0.872, `similar_episode` 0.715, `lexical` 0.742 | 0.3968 (model alone) |
| 2026-11-01 00:20 | Q1 + centrality, w = 0.5 (`team/export_ranker.py --central-weight 0.5`), scored from the exported folder. `cuda/q1_central05.zip` | same as Q1 | `added_action` | `qwen` 0.873, `similar_episode` 0.774, `lexical` 0.775, `random` 0.891, `swapped_action` 0.945 | `added_action` 0.949 | Not submitted |
| 2026-11-01 00:20 | Centrality alone (`cuda/central_submission.zip`) | None | All types | - | as the 23:20 row | about 0.53 (estimated from the first 100 rows of the per-episode CSV) |
