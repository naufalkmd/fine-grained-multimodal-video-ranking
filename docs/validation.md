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

## Findings from SigLIP and image evidence

Written 2026-11-01. Local scores use our own negatives and have overpredicted public scores, so only public scores can rank these. These are the team lead's local runs on an RTX 5070, so none can be the final submission. `docs/findings.md` covers the Qwen and centrality work.

### `siglip_central_v1`

- **Method:** SigLIP ranks each candidate against the video, centrality ranks it against the other candidates, and the two ranks are added with equal weight. There's no training.
- **SigLIP score:** each candidate is split into sentences, because SigLIP reads at most 64 tokens. A sentence takes its best match over the frames, and a candidate takes the mean over its sentences. Frame order isn't used.
- **Tokenizer:** loaded from `tokenizer.json`, because `sentencepiece` isn't in the evaluation environment.
- **ZIP:** `cuda/siglip_central_v1.zip`, 709 MiB. It holds the kit's DINOv2 and MPNet only because `package.py` requires them; they aren't loaded. Speed is about 0.08 s per episode locally.
- **Public:** 0.4682 (reported without a filename, assumed to be this ZIP). Its per-episode results were shared, and aren't used for tuning.

| Set | SigLIP alone | With wrong video | `siglip_central_v1` (150 episodes) | With wrong video |
|---|---:|---:|---:|---:|
| `random` | 0.978 | - | 0.639 | 0.369 |
| `similar_episode` | 0.925 | - | 0.689 | 0.477 |
| `lexical` | 0.895 | - | 0.746 | 0.469 |
| `word_swap` | 0.423 | - | - | - |
| `reorder` | 0.324 | - | - | - |
| `qwen` | 0.419 | 0.345 | 0.802 | 0.710 |

It scored below centrality alone (0.5515), and close to the midpoint of centrality and chance (0.463). So SigLIP probably adds little on the test, despite its strong content matching, and equal rank weights let it pull the rule down. This is an inference, not a measurement. A teammate's trained SigLIP ensemble (`team/kyounghoon/`), with no centrality, scored 0.3630 publicly, which supports it.

### Predicting combinations

A simulation calibrated to the public scores, assuming the two methods' errors are independent, puts SigLIP at about 0.36 alone and 3:1 ranks at about 0.539. It predicts that combining with centrality only helps if the image model alone scores about 0.45 or more publicly. At 0.48 it predicts 0.562 at 2:1, and at 0.52 it predicts 0.577 at 1:1. The simulation matches one check: it predicts centrality ranks first 37% of the time, against 39% in a 100-row sample.

### Image evidence

`team/submission_evidence.py` scores a candidate by its logit on this video minus its mean logit on 32 fixed training videos stored in the submission. Text preferences that ignore the video cancel. It needs a model trained on realistic edits to show a gain: on `qwen` it barely changes SigLIP, `temporal_word_swap_v1` or `h-v1`.

| `qwen` set | Right video | Wrong video |
|---|---:|---:|
| Qwen + `word_swap` model, normal | 0.780 | 0.652 |
| Same model, image evidence | 0.684 | 0.315 |

**Public: 0.3370, the lowest so far and about 2.5 standard errors below chance (0.375).** Local reliance on the image didn't carry over to the test. Removing the text habit probably removed useful signal, leaving little else. The wrong-video test measures reliance on the image, not skill on another benchmark. Across submissions so far, more reliance on the image tended to score lower, though that's a small, uncontrolled comparison:

| Submission | Public |
|---|---:|
| `text_centrality` (no image) | 0.5515 |
| `siglip_central_v1` | 0.4682 |
| `q1_central05` | 0.4068 |
| `temporal_word_swap_v1` | 0.3968 |
| `team/kyounghoon` SigLIP ensemble | 0.3630 |
| `h-v1` | 0.3586 |
| `b` | 0.3469 |
| `temporal_qwen_evidence_v1` | 0.3370 |

Don't build on image evidence. The simulation says combining needs an image model of about 0.45 alone, so the shortlist ZIPs weren't uploaded.

That model was trained with an `--external-sets` option that has since been removed in favour of `--qwen-candidates`, which also flips Qwen episodes. Retraining on Colab needs one of them chosen, and retesting.

### Video contrast loss

`--video-contrast-weight` also asks which of 12 batch videos fits each correct text, which text alone can't answer. On the `qwen` set it lowered the wrong-video score (0.460 against 0.652), but order pairing fell to 0.533 from 0.743. It also uses `--qwen-candidates`, unlike the model it's compared with, so that isn't a single change. Not uploaded, and not recommended.

### Shortlist

`team/submission_shortlist.py` lets centrality pick the top candidates, then orders them by image evidence, so the video decides first place. Local results are in the experiment log. Shortlists of 3 depend more on the video than 2 on the `qwen` set (wrong video 0.521 against 0.724).

Not uploaded, because the image model alone scored 0.3370.

### Tried and dropped

| Idea | Result | Why it was dropped |
|---|---|---|
| `compare_v1`: words see how the other candidates match them | `word_swap` 0.852, wrong video 0.852 | Ignores the video entirely |
| SigLIP on only the words that differ | `qwen` 0.255, `added_action` 0.233 | Ranks correct answers low on edits |
| SmolVLM yes/no check, 4 frames | `qwen` 0.383, `similar_episode` 0.701, 0.51 s per episode | Weaker than SigLIP, and too slow |
| SigLIP tie-break for centrality | `word_swap` from 0.439 to 0.912, depending on the tie width | Unstable |

### Test descriptions are probably long

The organizers' reports show the training positives were written from the frames at 42-75 words (median 60), in 2-4 sentences, and `manifest.json` records negatives started in the same way (774 for 387 episodes; the text isn't in our data). If the test was made the same way, its candidates are about 60 words rather than the slide's short sentences. That fits `word_swap`, which edits long descriptions, gaining on public (0.3968), while Qwen sets with a median of 12 words overpredict it. This is unconfirmed. The next step is regenerating Qwen edits at full length.

### Writing style and its reversal

`text_style` favours the candidate whose word pairs are most familiar from the training descriptions. It scored **0.2411** publicly, far below chance (0.375) given noise of about ±0.015, so on the test the correct description tends to read less like the training text. This is the opposite of our generated negatives, where edits make the wording less familiar (`word_swap` 0.612 locally), so our sets don't match the test on this signal. Why the test behaves this way is unknown.

Reversing the rule and adding centrality at equal weight scored **0.5443**, a tie with centrality (0.5515). We predicted about 0.58. `text_style_central` shouldn't be uploaded, because it combines the signal the wrong way round.

The site gives a per-episode CSV per submission, so a score tells us the rank of the correct text. Used in aggregate only (no tuning to individual episodes), they show:

| Rule | Public | % of episodes with correct text at rank 1 / 2 / 3 / 4 / 5 |
|---|---:|---:|
| Centrality | 0.5515 | 40.8 / 16.4 / 17.5 / 14.0 / 11.3 |
| `text_blend` | 0.5521 | 41.7 / 16.2 / 14.8 / 13.4 / 13.8 |
| Reversed style, computed from `text_style`'s CSV, not uploaded | 0.5327 | 35.3 / 23.9 / 18.2 / 12.2 / 10.5 |
| Reversed style + centrality, rank sum 1:1 | 0.5443 | 35.3 / 26.0 / 18.0 / 13.4 / 7.3 |

- Reversed style is a strong second signal, and its episode scores have almost no correlation with centrality's (-0.008). The correct text is first under both in 77 episodes, against 82.5 expected if independent.
- The rank sum still lost top places. A simulation that draws the negatives' places reproduces 0.5443 only if the two rules often rank the negatives alike (about 35% of episodes), so the two rules seem to share a favourite negative. That's an inference: the CSVs show only the correct text's rank.
- Reversing isn't always a gain: `temporal_qwen_evidence_v1` (0.3370) would only reverse to 0.3667.
- Paired 95% intervals against centrality: `text_blend` +0.0007 [-0.019, +0.020], rank sum -0.0072 [-0.032, +0.018].

`text_interleave` keeps centrality's top pick, places the top of reversed style second, and continues in turn. On the two top places alone, public scoring gives it 0.5175 against centrality's 0.4904, which is known from the CSVs. The full score of about 0.595 is a simulation at the calibrated overlap. The rule was chosen after seeing public statistics, so it needs its own upload.

### Fill in the blank

`team/submission_fill.py` scores each token by log p(token | frames) - log p(token | no frames), and compares candidates only on the tokens they don't share. Same model and 100 episodes per set as below; noise is about ±0.035. "Pair win" is how often the correct text beats one negative.

| Set | Whole text, right / wrong video | Differing tokens, right / wrong video | Pair win, right / wrong video |
|---|---:|---:|---:|
| `word_swap` | 0.412 / 0.330 | 0.481 / 0.301 | 0.625 / 0.463 |
| `qwen` | 0.378 / 0.326 | 0.417 / 0.311 | 0.600 / 0.490 |
| `swapped_action` | 0.689 / 0.245 | 0.814 / 0.224 | 0.863 / 0.365 |
| `added_action` | 0.556 / 0.399 | 0.614 / 0.400 | 0.775 / 0.600 |
| `similar_episode` | 0.929 / 0.326 | 0.951 / 0.335 | 0.968 / 0.500 |

Differing tokens beat the whole text on every set, and this is the first image method to clear a 58% pair win on near-identical candidates. All 834 validation episodes are outside the training part of the split the fine-tune's settings name (loss per token 2.010 on train, 2.116 on validation).

Publicly, `fill_central_v1` (1:1 with centrality) scored 0.5044. That is the best equal-weight image combination so far (`siglip_central_v1` 0.4682) but below centrality. The earlier simulation, which assumes independent errors, implies about 0.41 for `fill_v1` alone, under the roughly 0.45 where combining starts to help. It ran within the time limit at 8 frames.

### Merging the text rules

The public CSVs give the correct text's rank under each rule, so merging rules can be compared with no upload. The negatives' places are unknown, so they're drawn, with overlap rho: the chance the two rules order the negatives alike. rho = 0.35 reproduces the observed 1:1 rank sum (0.5443), and the share ranked first by (centrality rank, flip rank) group is off by 0.034 on average.

| Merge with reversed style, rho = 0.35 | Centrality | `text_blend` |
|---|---:|---:|
| **Take turns** | **0.5945** | **0.6016** |
| Likelihood from public ranks | 0.5858 | 0.5912 |
| Points 1, 0.5, 0.25... (same as 1/rank here) | 0.5695 | 0.5696 |
| Rank sum | 0.5447 | 0.5394 |
| Take turns, worst case from the CSVs, no simulation | 0.5550 | **0.5639** |

The worst case assumes every place not forced by the two ranks goes against the correct text. If private behaves like public, `text_blend` first beats `text_blend` alone on the private 1,337 episodes with probability 0.88 at the worst case, and above 0.99 at any overlap we simulated. These were estimates made before upload. **`text_blend_interleave` then scored 0.5958 publicly,** inside the simulated range.

### The video deciding between two candidates

`team/submission_arbiter.py` keeps the turn-taking but lets SmolVLM put the second rule's pick first when the frames favour it by more than a threshold. 100 episodes per set; correct text against each negative. "Right" is how often the frames' margin picks the correct text.

| Margin above | Share of pairs | `word_swap` + `qwen`, right / wrong video | Share of pairs | `swapped_action` + `added_action` + `reorder`, right / wrong video |
|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.613 / 0.476 | 1.00 | 0.719 / 0.466 |
| 1 | 0.41 | 0.716 / 0.476 | 0.78 | 0.772 / 0.452 |
| 2 | 0.18 | 0.790 / 0.491 | 0.57 | 0.842 / 0.470 |
| 3 | 0.09 | 0.786 / 0.632 | 0.44 | 0.894 / 0.451 |

Larger margins are more accurate with the right video and stay near 0.5 with the wrong one, up to a margin of 2. Above that the pairs are few. Subtracting the same margin under 8 fixed training videos brings the wrong-video pair win nearer 0.5 on some sets, but isn't better overall: `word_swap` 0.685 / 0.537, `qwen` 0.568 / 0.512, `swapped_action` 0.907 / 0.485, `added_action` 0.650 / 0.450, `reorder` 0.627 / 0.480 (without it: 0.625 / 0.463, 0.600 / 0.490, 0.863 / 0.365, 0.775 / 0.600, 0.520 / 0.432).

The gain from overruling depends on how often the frames overrule each kind of pair. Public counts give 125 episodes where only the reversed-style pick is correct, and 157 where only centrality's is. Local stand-ins, 200 episodes per set across `word_swap`, `qwen`, `swapped_action`, `added_action` and `reorder`: the correct text being the least familiar, with the most central negative (58 pairs, mostly from `swapped_action` and `added_action`, so noisy), and the correct text against the least familiar negative (942 pairs). Gain = 0.5 x (125 x share overruled when right - 157 x share overruled when wrong) / 573.

| Threshold | Right video: overruled when right / when wrong, gain | Wrong video: same |
|---:|---:|---:|
| 0 | 0.724 / 0.315, +0.036 | 0.345 / 0.515, -0.033 |
| 1 | 0.621 / 0.132, +0.050 | 0.207 / 0.310, -0.020 |
| 2 | 0.466 / 0.063, +0.042 | 0.172 / 0.193, -0.008 |
| 3 | 0.379 / 0.027, +0.038 | 0.138 / 0.107, +0.000 |

A threshold of 3 keeps most of the gain and removes the loss when the frames carry no information, so `arbiter_v1` uses it. It wasn't tuned on public results. The gain is likely overstated for the test: near-identical pairs rarely reach a margin of 3 (9% of `word_swap` and `qwen` pairs, right 79% of the time), which alone would give about +0.005.

**Public evidence against this.** `smolvlm_confidence_v1`, on `feat/smolvlm-gate`, let contrast-trained SmolVLM overrule centrality when its margin was at least 4, and scored 0.5382: 49 episodes changed, 16 better, 33 worse. Centrality's public ranks imply it wins about 0.65 of pairs, so a model with no information would overrule correctly about 0.35 of the time, and it did so 0.33 of the time. So those confident decisions look uninformative on the test, not harmful in themselves, despite 0.96 locally. It's a different model and rule, but `arbiter_v1`'s expected gain should be taken as about zero unless a public result shows otherwise.

A 4-epoch fine-tune was stopped during its first epoch. At about 2 s per batch on an RTX 5070 with the GPU to itself, an epoch takes about 28 minutes, so a T4 at 2 to 4 times slower couldn't reproduce 4 epochs by the deadline. Two epochs would take about 2 to 4 hours there.

### The order probe and the base-candidate gates

`probe_order` scored 0.4295 (1,969 / 4,584), about 3.5 standard errors above chance. That doesn't show it reads event order: on pure-noise similarities its score falls from -0.125 at 2 events to -0.474 at 8, so it leans toward candidates with fewer events with no video at all. The lean cancels where candidates have the same events, as in order changes, but not elsewhere. A reversed-frames upload would separate order from length. Simulated with a probe independent of the text rules, adding it to `text_blend_interleave` moves the score by -0.009 to +0.005, whichever way it's merged.

`text_order_third` (on `feat/smolvlm-gate`: the text rule's top two, then turns among `text_blend`, unfamiliar wording and the order check) scored **0.5914** publicly, 0.0044 below `text_blend_interleave`. Simulated as written before that score arrived, from the public ranks with the check's mean near 0.4295, it changes the score by -0.003 to +0.001 if the check is independent of the text rules, and by +0.004 to +0.006 if all of its lift falls in the 32% of episodes where their top two miss. With paired noise of about 0.0025, the result fits the first and makes the second unlikely, so the check doesn't appear to complement the text rules where they fail. A perfect check in those places would give +0.034. `docs/audit.md` on that branch finds the same lean locally with the wrong video.

The larger lever is which top pick goes first. Public counts: first under both 77, only `text_blend` 162, only reversed style 125. Centrality's rank spread fits a mix where the correct text is the base in about 15-35% of episodes. Moving the least familiar pick first only where the correct text isn't the base gains +0.016 to +0.081 if a gate catches all of those episodes, and about half that if it catches 70% and wrongly moves 10% of the base episodes. These are estimates from a hypothesis about the test. On synthetic candidates, with shuffled order and random ties:

| Structure | Centrality right | `gate` leaves it alone | Lowest excess under 0.3 |
|---|---:|---:|---:|
| Correct is the base, 1-word edits | 0.99 | 1.00 | 1.00 |
| Correct is the base, mixed edits | 0.94 | 1.00 | 0.91 |
| Hidden original, 1-word edits | 0.21 | 0.01 | 0.00 |
| Hidden original, mixed edits | 0.20 | 0.48 | 0.72 |
| Edits of edits | 0.10 | 1.00 | 0.78 |
| Order changes only | 0.19 | 1.00 | 0.26 |

`text_blend_interleave_gated` uses `gate` alone, and `text_blend_interleave_gated_excess` also fires at excess 0.3 or more. Neither has a public score yet. `gate` costs nothing on order changes, because it leaves them alone; it acts only where no candidate looks like the base. Local scores can't judge them, because reversed style points the wrong way on our sets (no training, `lexical` falls from 0.711 to 0.496 under both).

### Audit of our assumptions

Checked on 2026-11-01. Per-episode CSVs were used in aggregate only.

| Assumption or idea | Result | Evidence |
|---|---|---|
| The leaderboard is the plain mean over episodes | Confirmed | CSV means match the public scores exactly; a score of 0.4682 matches `siglip_central_v1` |
| `smolvlm_ft_v1` didn't train on our validation episodes | Very likely | The split recomputes to `splits.json`, and all 834 candidate episodes are in it. Loss per token 2.010 on train, 2.116 on validation. Not checked against the friend's own file |
| Test order or position carries information | No | Score autocorrelation about 0 at lags 1, 2, 5 and 10, in the two CSVs checked; `predict` never receives the episode ID |
| Below-chance models all lean on familiar wording | Mostly no | Rank correlation among negatives only is mostly 0 to +0.18; the exception is `temporal_qwen_word_swap_v1` on `qwen` (+0.286). `temporal_qwen_evidence_v1` itself wasn't tested |
| `text_style` is a proxy for length | No | r = 0.011 with word count across validation positives |
| Gaps under 0.03 are ties | Only without CSVs | Paired comparisons are more sensitive, as with the sign test on `q2_central4` |

The organizers' `README.md` and reports say Codex wrote the training positives from the frames, with a separate Codex review and no human review. `manifest.json` records 774 negatives for 387 episodes, removed from our data. If the test was made the same way, its negatives are deliberate, natural-sounding edits, unlike the rule-based `word_swap`. This is an inference: we can't see the test text. Length is also untested publicly, because no `text_shortest` score has been reported to us.

## Public scores

All from the leaderboard (573 episodes, chance 0.375, gaps under about 0.03 are ties).

| Submission | Uses the image? | Public |
|---|---|---:|
| `text_blend_interleave` | No | **0.5958** |
| `text_order_third` | From third place on | 0.5914 |
| `text_blend` | No | 0.5521 |
| `text_centrality` / `central_submission` | No | 0.5515 |
| `smolvlm_gate_v1` | Rarely | 0.5502 |
| `text_style_flip_central` | No | 0.5443 |
| `q2_central4` | Slightly | 0.5312 |
| `fill_central_v1` | Yes | 0.5044 |
| `siglip_central_v1` (reported without a filename) | Yes | 0.4682 |
| `probe_order` | Yes, with a text lean | 0.4295 |
| `q1_central05` | Yes | 0.4068 |
| `temporal_word_swap_v1` | Yes | 0.3968 |
| `team/kyounghoon` SigLIP ensemble | Yes | 0.3630 |
| `h-v1` | Yes | 0.3586 |
| `b` | Yes | 0.3469 |
| `temporal_qwen_evidence_v1` | Yes | 0.3370 |
| `text_style` | No | 0.2411 |

Built and validated with no score reported to us: `text_blend_interleave_gated`, `text_blend_interleave_gated_excess`, `arbiter_v1`, `text_interleave`, `text_interleave_small`, `unfamiliar_v1`, `fill_v1`, `text_style_flip`, `text_style_central`, `compare_v1`, `siglip_central_3to1`, `text_shortest`, `text_consensus`, `text_meaning`.

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
| 2026-10-31 23:15 | H with `word_swap` 2, `similar_episode` 1, `lexical` 1, 60 fixed epochs, frozen rule, trained locally. `cuda/temporal_word_swap_half_v1.zip` | `lexical`, `similar_episode`, `word_swap` | `reorder`, `added_action`, `swapped_action`, `random` | `similar_episode` 0.786, `lexical` 0.730, `word_swap` 0.577 | `added_action` 0.749; `swapped_action` 0.666; `random` 0.874; `reorder` 0.254. Wrong video: `word_swap` 0.456. Order pairing 0.735 | Not submitted |
| 2026-10-31 23:30 | Diagnostic text rule `text_centrality`: picks the candidate sharing the most words with the others, ignoring the video. No training. Uploaded by a teammate | None | None | Not comparable: our own negatives are mostly edits of the correct text, so this rule nearly solves them | Not comparable | **0.5515** |
| 2026-10-31 23:20 | Text-only `team/submission_central.py`: ranks candidates by mean word-level similarity to the other candidates; no video, no training. Candidates: `candidates_qwen.json` (adds `qwen`: Qwen3-4B short positive + 4 single-change negatives, 323 complete validation items) | None | All types | - | `added_action` 1.000, `mixed` 0.999, `qwen` 0.997, `swapped_action` 0.971, `lexical` 0.550, `reorder` 0.506, `similar_episode` 0.388, `random` 0.370 | Not submitted |
| 2026-10-31 23:27 | Q1: H settings + `--qwen-candidates candidates_v3.jsonl --qwen-probability 0.5` (Qwen3-4B short positive + single-change negatives on half the episodes), 60 fixed epochs, frozen rule; best epoch 60. `cuda/q1_h_qwen50.zip` | same as H + `qwen` | `added_action` | `similar_episode` 0.819, `lexical` 0.761, `qwen` 0.650 | `added_action` 0.799. Wrong video: `qwen` 0.500, `added_action` 0.426. Order pairing 0.639 | Not submitted |
| 2026-10-31 23:35 | T2 checkpoint (no Qwen data) rescored on `candidates_qwen.json` for reference | as T2 | as T2 | - | `qwen` 0.379. Wrong video: `qwen` 0.338 | Not submitted |
| 2026-11-01 00:05 | Team lead's H + `word_swap` weights (`cuda/requested_data`, public 0.3968) rescored on `candidates_qwen.json`, with and without centrality added (z-scored per episode, model + w x centrality) | as H + `word_swap` | `added_action` | - | `qwen` 0.390 alone; +centrality w=1: `qwen` 0.668, `similar_episode` 0.689. Q1 for comparison: `qwen` 0.650 alone; w=1: `qwen` 0.872, `similar_episode` 0.715, `lexical` 0.742 | 0.3968 (model alone) |
| 2026-11-01 00:20 | Q1 + centrality, w = 0.5 (`team/export_ranker.py --central-weight 0.5`), scored from the exported folder. `cuda/q1_central05.zip` | same as Q1 | `added_action` | `qwen` 0.873, `similar_episode` 0.774, `lexical` 0.775, `random` 0.891, `swapped_action` 0.945 | `added_action` 0.949 | **0.4068** |
| 2026-11-01 00:06 | `siglip_v1`: SigLIP sentence-to-frame matching, no training, no order. `cuda/siglip_v1.zip` | None | All types | - | `similar_episode` 0.925, `lexical` 0.895, `random` 0.978, `word_swap` 0.423, `reorder` 0.324, `qwen` 0.419. Wrong video: `qwen` 0.345 | Not submitted |
| 2026-11-01 00:23 | `combined_v1`: rank average of `temporal_word_swap_v1` and SigLIP. `cuda/combined_v1.zip` | as `temporal_word_swap_v1` | `added_action` | `qwen` 0.433, `word_swap` 0.562, `similar_episode` 0.885, `lexical` 0.843 | `added_action` 0.926. Wrong video: `qwen` 0.372, `added_action` 0.465 | Not submitted |
| 2026-11-01 00:23 | `temporal_qwen_word_swap_v1`: H + `word_swap`, with Qwen sets on half of episodes, no flipping on those, 60 fixed epochs. `cuda/temporal_qwen_word_swap_v1.zip` | same as H + `word_swap`, `qwen` | `added_action` | `qwen` 0.780, `word_swap` 0.502, `similar_episode` 0.800, `lexical` 0.744 | `added_action` 0.810. Wrong video: `qwen` 0.654, `word_swap` 0.375. Order pairing 0.743 | Not submitted |
| 2026-11-01 00:27 | `siglip_central_v1`: rank average of SigLIP and centrality, no training. 150 episodes per set. `cuda/siglip_central_v1.zip` | None | All types | - | `qwen` 0.802, `similar_episode` 0.689, `lexical` 0.746, `random` 0.639. Wrong video: `qwen` 0.710, `similar_episode` 0.477 | **0.4682**, reported without a filename, assumed to be this ZIP |
| 2026-11-01 00:57 | `siglip_diff_central_v1`: SigLIP on the words that differ between candidates, rank averaged with centrality, no training. `cuda/siglip_diff_central_v1.zip` | None | All types | - | Difference matching alone: `qwen` 0.255, `added_action` 0.233, `word_swap` 0.599, `similar_episode` 0.836. With centrality: `qwen` 0.719, `added_action` 0.537, `similar_episode` 0.645. Wrong video with centrality: `qwen` 0.600 | Not submitted, because it ranks correct answers low on edits |
| 2026-11-01 01:04 | `siglip_central_3to1`: centrality rank counts 3 times SigLIP rank, no training. `cuda/siglip_central_3to1.zip` | None | All types | - | `qwen` 0.981, `word_swap` 0.995, `swapped_action` 0.979, `similar_episode` 0.485, `lexical` 0.599, `random` 0.428 (centrality alone: 0.997, 0.995, 0.974, 0.434, 0.555, 0.352). 200-323 episodes per set | Not submitted |
| 2026-11-01 01:16 | `temporal_qwen_evidence_v1`: `temporal_qwen_word_swap_v1` scored as logit on this video minus mean logit on 32 fixed training videos. `cuda/temporal_qwen_evidence_v1.zip` | same as H + `word_swap`, `qwen` | `added_action` | `qwen` 0.684, `similar_episode` 0.809 | Wrong video: `qwen` 0.315, `similar_episode` 0.421 | **0.3370** (573 episodes) |
| 2026-11-01 01:53 | Combinations of `temporal_qwen_evidence_v1` and centrality, on the `qwen` set: shortlist 2 / shortlist 3 / rank 1:1 / rank 2:1. `cuda/evidence_shortlist2_v1.zip`, `cuda/evidence_shortlist3_v1.zip` | as above | `added_action` | `qwen` 0.884 / 0.804 / 0.868 / 0.959 | Wrong video: `qwen` 0.724 / 0.521 / 0.599 / 0.821 | Not submitted |
| 2026-11-01 02:26 | `temporal_qwen_contrast_v1`: H + `word_swap`, `--qwen-candidates` on half of episodes, `--video-contrast-weight 1`, 60 fixed epochs, best epoch 57. `cuda/temporal_qwen_contrast_v1.zip` | same as H + `word_swap`, `qwen` | `added_action` | `qwen` 0.622, `word_swap` 0.489, `similar_episode` 0.829, `lexical` 0.749 | `added_action` 0.845. Wrong video: `qwen` 0.460. Order pairing 0.533 | Not submitted |
| 2026-11-01 02:26 | `temporal_qwen_contrast_evidence_v1`: the same model scored by image evidence. `cuda/temporal_qwen_contrast_evidence_v1.zip` | as above | `added_action` | `qwen` 0.537, `similar_episode` 0.840 (150 episodes) | Wrong video: `qwen` 0.320, `similar_episode` 0.383 | Not submitted |
| 2026-11-01 00:20 | Centrality alone (`cuda/central_submission.zip`) | None | All types | - | as the 23:20 row | **0.5515** (573 episodes) |
| 2026-11-01 00:44 | Q2: Q1 + `--qwen-reverse-actions` (on reversed frames the reversed-action sentence is correct and the positive is a negative, like swapped order), 60 fixed epochs; best epoch 60 | same as Q1 | `added_action` | `similar_episode` 0.827, `lexical` 0.772, `qwen` 0.622 | `added_action` 0.791. Wrong video: `qwen` 0.456, `added_action` 0.477. Order pairing 0.630. Action pairing (reversing frames flips the positive vs reversed-action preference; text cues cancel; chance 0.5, n = 566): 0.640, against Q1 0.544 and H + `word_swap` 0.530 | Not submitted |
| 2026-11-01 01:00 | Q2 + centrality, w = 4 (centrality leads; the model mostly breaks near-ties), scored from the exported folder. `cuda/q2_central4.zip` | same as Q2 | `added_action` | `qwen` 0.994, `similar_episode` 0.468, `lexical` 0.611, `random` 0.466, `swapped_action` 0.973 | `added_action` 1.000 | **0.5312** (573 episodes). Against centrality alone (0.5515), 118 episodes changed: 45 better, 73 worse (sign test p = 0.013); most losses moved the correct text from 1st to 2nd (31) or 2nd to 3rd (20) |
| 2026-11-01 02:40 | `smolvlm_zero_shot`: SmolVLM-256M-Instruct likelihood of each candidate given 8 frames, minus `prior_weight` times its likelihood with no frames (`team/smolvlm_check.py`, 120 episodes, `word_swap` built in the script). No training | None | All types | - | w = 1: `word_swap` 0.357, `qwen` 0.448, `added_action` 0.746, `swapped_action` 0.634, `similar_episode` 0.796. Wrong video: 0.330, 0.401, 0.622, 0.360, 0.403. Action pairing 0.525. Higher scores at w < 1 match the wrong-video scores (text naturalness) | Not submitted |
| 2026-11-01 02:55 | `smolvlm_ft_v1`: SmolVLM fine-tuned to write the correct description from 8 frames (`team/train_smolvlm.py`, 2 epochs, frames dropped on 20% of batches, vision encoder frozen); validation loss 2.03 with frames, 2.21 without (3.41 / 3.55 before). Same check | Correct descriptions only | All types | - | w = 1: `word_swap` 0.408, `qwen` 0.378, `added_action` 0.550, `swapped_action` 0.698, `similar_episode` 0.894. Wrong video: 0.320, 0.328, 0.400, 0.259, 0.386. Action pairing 0.508. Pairs below 0.5 word similarity: 0.922 (wrong video 0.513); above 0.95: 0.580 (0.498) | Not submitted alone |
| 2026-11-01 03:27 | `smolvlm_gate_v1`: pairs below 0.6 word similarity decided by `smolvlm_ft_v1` (4 frames, w = 1), other pairs by centrality, ranked by wins (`team/submission_smolvlm_gate.py`). 150 episodes. `cuda/smolvlm_gate_v1.zip`, 773 MiB | Correct descriptions only | All types | - | `word_swap` 0.995, `qwen` 0.957, `swapped_action` 0.972, `similar_episode` 0.891, negatives edited from another episode's description 0.968 (centrality: 0.995, 0.993, 0.980, 0.394, 0.000). 0.12 s per episode on an RTX 5070, worst 0.52 s | **0.5502** (573 episodes); against centrality's public CSV only 2 episodes changed, both worse (407: 1st to 2nd; 425: 3rd to 5th) |
| 2026-11-01 05:29 | `contrast_v1`: `smolvlm_ft_v1` plus one epoch of `team/train_smolvlm_contrast.py` (one-word edit T' of the correct text T, another episode's frames V2; loss on [log p(T\|V) - log p(T'\|V)] - [log p(T\|V2) - log p(T'\|V2)], plus captioning with and without frames). Checked with `team/smolvlm_check.py`, 120 episodes | Correct descriptions, `word_swap` | `qwen`, `added_action`, `swapped_action`, `similar_episode` | Pairwise at w = 1: `word_swap` 0.624 (wrong video 0.452; `ft_v1` 0.580 / 0.498) | Pairwise at w = 1: `qwen` 0.527 (0.446), `added_action` 0.696 (0.469), `swapped_action` 0.815 (0.354), `similar_episode` 0.938 (0.540). Action pairing 0.525 | Not submitted alone |
| 2026-11-01 06:00 | `smolvlm_confidence_v1`: `smolvlm_gate_v1`'s rule with `contrast_v1`, plus SmolVLM deciding pairs whose scores differ by 4 or more (`--video-margin 4`). 150 episodes. `cuda/smolvlm_confidence_v1.zip` | Correct descriptions, `word_swap` | Other types | `word_swap` 0.988 (centrality 0.995); negatives sharing one edit plus their own 0.499 (centrality 0.387, wrong video 0.397) | `qwen` 0.917 (0.993), `swapped_action` 0.932 (0.980), `similar_episode` 0.924 (0.394). 0.164 s per episode on an RTX 5070 | **0.5382** (573 episodes); against centrality 49 episodes changed, 16 better, 33 worse (sign test p = 0.021) |
| 2026-11-01 07:00 | LLM-edited validation sets: `team/qwen_long_negatives.py` asks Qwen3-4B (greedy) for four full-length incorrect versions of each description (reversed action, added action, changed detail, changed order); `--style edit` keeps the wording, `--style rewrite` asks for rewording. Scored with `team/text_rules.py` (bigram counts from the training split only) | None | All types | - | Edit set (125 episodes) / rewrite set (299): centrality 1.000 / 0.992, `style` 0.584 / 0.640, reversed style 0.124 / 0.119, `interleave` 1.000 / 0.988, `rank_sum` 0.690 / 0.663, content-word centrality 0.872 / 0.896, style outlier 0.086 / 0.072. Qwen kept the wording even when asked to reword. Neither set reproduces public (centrality 0.5515, `style` 0.2411), so LLM edits of the correct text don't look like the test | Not applicable |
| 2026-11-01 06:40 | `regime_v1`: `team/submission_regime.py`. If some candidate has no word missing from every other candidate (the others look like edits of it), centrality's order interleaved with unfamiliar wording; otherwise unfamiliar wording first. Bigram counts from all 4,170 training descriptions. `cuda/regime_v1.zip`, 349 MiB, about 5 ms per episode on CPU | None (counts only) | All types | - | The gate picks centrality first on 96-100% of edit-style sets, where a zero-unique candidate is always correct (`reorder` 87%); on other-scene sets it always picks unfamiliar first. Local unfamiliar scores aren't fair (counts include these descriptions) | Not uploaded yet |
| 2026-11-01 03:02 | `text_consensus`: mean share of the other candidates containing each of a candidate's words. No images, no training. `cuda/text_consensus.zip` | None | All types | - | `qwen` 0.736, `word_swap` 0.995, `swapped_action` 0.948, `added_action` 1.000, `reorder` 0.393, `lexical` 0.705, `similar_episode` 0.458, `random` 0.412 | No score reported |
| 2026-11-01 03:02 | `text_meaning`: mean MPNet cosine similarity to the other candidates. No images, no training. `cuda/text_meaning.zip` | None | All types | - | `qwen` 0.960, `word_swap` 0.940, `swapped_action` 0.907, `added_action` 0.882, `reorder` 0.482, `lexical` 0.625, `similar_episode` 0.424, `random` 0.382 | No score reported |
| 2026-11-01 03:05 | `text_blend`: rank sum of word-sequence similarity, word consensus and MPNet meaning similarity between candidates; no images, no training. `cuda/text_blend.zip` | None | All types | - | `qwen` 1.000, `word_swap` 0.995, `swapped_action` 0.969, `added_action` 0.999, `reorder` 0.488, `lexical` 0.698, `similar_episode` 0.432, `random` 0.393 | **0.5521** |
| 2026-11-01 03:52 | `text_style`: mean bigram log probability per word; the ZIP counts all 4,170 training descriptions. No images, no negatives. `cuda/text_style.zip` | None (counts only) | All types | - | Train-split counts: `word_swap` 0.612, `qwen` 0.586, `reorder` 0.581, `added_action` 0.426, `swapped_action` 0.424, `random` 0.369, `similar_episode` 0.352, `lexical` 0.336 | **0.2411**, far below chance |
| 2026-11-01 03:52 | `text_style_central`: `text_style` + centrality ranks, 1:1. `cuda/text_style_central.zip` | None (counts only) | All types | - | Train-split counts: `word_swap` 0.937, `qwen` 0.904, `added_action` 0.759, `swapped_action` 0.750, `reorder` 0.573, `lexical` 0.452, `similar_episode` 0.374, `random` 0.368 | Not submitted: combines style the wrong way round |
| 2026-11-01 03:52 | `compare_v1`: ranker whose words see how the other candidates match them. `cuda/compare_v1.zip` | See "Tried and dropped" | `added_action` | `word_swap` 0.852 | Wrong video: `word_swap` 0.852, so it ignores the video | No score reported |
| 2026-11-01 04:17 | Patch grid: `--patch-grid 4 --model temporal --reverse-weight 0.1 --flip-probability 0.5` (DINOv2 patches in a 4x4 grid per frame). Code uncommitted on `feat/patch-grid`, not exported | `reorder`, `swapped_action`, `similar_episode`, `random`, `lexical`, `word_swap` | `added_action` | `word_swap` 0.536 (`temporal_word_swap_v1` 0.555), clean rank score 0.793 (0.774) | `qwen` 0.350 (0.390). Wrong video: `word_swap` 0.419. Order pairing 0.685. No gain | Not submitted |
| 2026-11-01 04:18 | Fill in the blank, untrained SmolVLM (8 frames, w = 1, 100 episodes per set). Differing tokens only | None | All types | - | `word_swap` 0.396, `qwen` 0.477, `swapped_action` 0.731, `added_action` 0.835, `similar_episode` 0.782. Wrong video: 0.311, 0.335, 0.351, 0.649, 0.301. Pair win `word_swap` 0.570 (wrong video 0.512) | Not submitted |
| 2026-11-01 04:25 | Fill in the blank, `smolvlm_ft_v1`: see "Fill in the blank" | Correct descriptions only | All types | - | `word_swap` 0.481, `qwen` 0.417, `swapped_action` 0.814, `added_action` 0.614, `similar_episode` 0.951. Wrong video: 0.301, 0.311, 0.224, 0.400, 0.335 | Not submitted alone |
| 2026-11-01 04:33 | `fill_v1`: `team/submission_fill.py`, scored from the unpacked ZIP. `cuda/fill_v1.zip`, 773 MiB. 0.33 s per episode locally, about 10.5 min for 1,910 episodes | Correct descriptions only | All types | - | `word_swap` 0.481, `similar_episode` 0.951 (100 episodes each) | Not uploaded yet |
| 2026-11-01 04:33 | `fill_central_v1`: `fill_v1` + centrality ranks, 1:1. `cuda/fill_central_v1.zip`, 773 MiB. 0.33 s per episode locally | Correct descriptions only | All types | - | `word_swap` 0.856, `similar_episode` 0.621 (100 episodes each). `word_swap` is inflated, because centrality nearly solves it | **0.5044**, below centrality (0.5515) |
| 2026-11-01 04:50 | `text_style_flip`: `text_style` reversed (`style_sign` -1), direction chosen from its public score. `cuda/text_style_flip.zip` | None (counts only) | All types | - | Counts include these validation descriptions, so these are unfair: `similar_episode` 0.406, `lexical` 0.393, `swapped_action` 0.222, `added_action` 0.193, `qwen` 0.128, `word_swap` 0.086 | Not submitted |
| 2026-11-01 04:50 | `text_style_flip_central`: `text_style_flip` + centrality ranks, 1:1. `cuda/text_style_flip_central.zip` | None (counts only) | All types | - | Same unfair counts: `swapped_action` 0.608, `added_action` 0.574, `lexical` 0.502, `qwen` 0.461, `word_swap` 0.438, `similar_episode` 0.426 | **0.5443**, a tie with centrality (0.5515) |
| 2026-11-01 05:45 | `text_interleave`: centrality's top, then reversed style's top, and so on in turn; rule chosen from public CSV statistics. `cuda/text_interleave.zip` | None (counts only) | All types | - | Checked on 6,778 local rankings: first place always equals centrality's, second is as designed. Local scores aren't informative for this rule | Not uploaded yet; simulated about 0.595 |
| 2026-11-01 06:37 | `text_interleave_small`: `text_interleave` without the kit's encoders, which it never loads (`team/package_small.py`). `cuda/text_interleave_small.zip`, 0.2 MiB | None (counts only) | All types | - | Same rankings as `text_interleave` on 6,778 local rankings; passes the kit's interface check from another folder in isolated mode | Not uploaded yet |
| 2026-11-01 07:07 | `text_blend_interleave`: `text_blend`'s top, then reversed style's top, and so on in turn; rule chosen from public CSV statistics. `cuda/text_blend_interleave.zip`, 349 MiB. Word-pair counts match `text_interleave`. Rehearsed through the Colab steps locally | None (counts only) | All types | - | Checked on 6,778 local rankings: first place always equals `text_blend`'s, second is as designed. Local scores aren't informative for this rule | **0.5958** (573 episodes), uploaded by njunu17, the best so far. Worst case 0.5639, simulated 0.59-0.60. The site showed the mean only, with no per-episode CSV |
| 2026-11-01 08:18 | `text_blend_interleave`, built on a Colab Tesla T4 with `team/final_text_colab.ipynb`, run `20261031_230904_336fbf`, in `colab_runs/` on Drive. 349.1 MiB, sha256 `d3bbad91...5787`. Kit MLP on 12 episodes, not loaded by the rule | None (counts only) | All types | - | Against `cuda/text_blend_interleave.zip`: same code, word counts and encoder weights; rankings identical on all 6,778 local rankings | Not uploaded yet. It imports `submission_style` during ranking, and crashes in a stricter local load test where the ZIP's folder is importable only while loading. That risk looks unfounded: `feat/smolvlm-gate`, the source of the 0.5958 upload as far as we know, has the same import. The import later moved to load time (`1106017`), which changes no ranking |
| 2026-11-01 09:35 | `text_blend_interleave_gated`: `text_blend_interleave`, with reversed style's top first when no candidate has all its words in the others. `text_blend_interleave_gated_excess`: also when the lowest word edit-distance excess is 0.3 or more; the threshold comes from synthetic candidates. `cuda/text_blend_interleave_gated.zip` and `cuda/text_blend_interleave_gated_excess.zip`, 349 MiB, about 0.02 s per episode | None (counts only) | All types | - | 150 episodes per set, plain / `gate` / excess: `added_action` 1.000 / 1.000 / 0.950, `swapped_action` 0.980 / 0.974 / 0.960, `similar_episode` 0.486 / 0.463 / 0.463, `lexical` 0.711 / 0.496 / 0.496, `random` 0.405 / 0.377 / 0.377, `reorder` 0.398 / 0.398 / 0.239; `word_swap` 1.000, `mixed` 0.985 and `qwen` 1.000 under all three. Lower is expected where a gate fires, because reversed style points the wrong way locally | Not uploaded yet |
| 2026-11-01 07:21 | `arbiter_v1`: `text_blend_interleave`, with `smolvlm_ft_v1` putting the second pick first when the frames favour it by more than 3 (8 frames, w = 1). `cuda/arbiter_v1.zip`, 773 MiB. 0.34 s per episode locally | Correct descriptions only | All types | - | 100 episodes per set, text only / right video / wrong video, share overruled right / wrong: `word_swap` 1.000 / 0.995 / 0.995, 0.01 / 0.01; `qwen` 1.000 / 0.985 / 0.980, 0.03 / 0.04; `swapped_action` 0.980 / 0.970 / 0.810, 0.02 / 0.34; `added_action` 1.000 / 0.980 / 0.975, 0.04 / 0.05; `reorder` 0.499 / 0.489 / 0.489, 0.02 / 0.06; `similar_episode` 0.395 / 0.490 / 0.410, 0.37 / 0.33. These sets nearly hand the text rule the answer, so the override can rarely help on them | Not uploaded yet |
| 2026-11-01 06:11 | `unfamiliar_v1`: least expected wording under `smolvlm_ft_v1` with no frames, on tokens that differ, averaged per token. No images. `cuda/unfamiliar_v1.zip`, 773 MiB. Direction chosen from `text_style`'s public score. About 0.08 s per episode locally | Correct descriptions only | All types | - | 200 episodes per set; low is expected, because local negatives point the other way on this signal: `word_swap` 0.043, `swapped_action` 0.067, `qwen` 0.231, `added_action` 0.272, `lexical` 0.426, `similar_episode` 0.454. Rank correlation with word-pair familiarity reversed: +0.25 to +0.60 | Not uploaded yet |
