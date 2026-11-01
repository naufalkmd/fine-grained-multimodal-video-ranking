# Audit: what we know, how sure we are, and what to do next

## Status at a glance (10:20 KST)

- **Best public: `text_blend_interleave`, 0.5958.** Text only: `text_blend`'s top pick, then the
  least familiar wording's top pick, in turn. Its worst case was checked exactly from the public
  CSVs before upload (0.5639).
- **The SigLIP order check scored 0.4295 alone** (chance 0.375), but this is **not clean evidence of
  an image signal**: locally, with another episode's video, it still favours candidates with fewer
  events (correlation -0.15 to -0.31; `added_action` 0.486 with the wrong video). Only on
  same-length reorderings is the image part clear (0.470 right vs 0.376 wrong). The public mean
  can't separate the two.
- **`text_order_third.zip` scored 0.5914**: the 0.5958 rule's top two plus the order check from
  third place on, so no gain from the order check (the bar for replacing was about +0.02).
- **Final so far: `text_blend_interleave` (0.5958).**
- **Next upload: `text_blend_interleave_gated`.** It differs only in putting reversed style's pick
  first when no candidate has all its words in the others (no candidate looks like the base). It
  can't act where the correct text is the base, which is where `text_blend` wins, so its downside is
  small; public counts cap its gain at +0.109 (125 episodes only reversed style gets right). Earlier
  this audit dropped it because it does nothing on order-only negatives; that was a mistake, since
  doing nothing there costs nothing (the lead's point). Final if it reaches about 0.616; Colab
  rebuild with `METHOD = "blend_interleave_gated"` in `team/final_text_colab.ipynb`.
- **Built, not uploaded: the least familiar wording judged by word pairs and characters together**
  (`blend_interleave_chars`, `blend_interleave_gated_chars`; `team/submission_chars.py`). Character
  familiarity keeps case and punctuation, which the word-pair score drops. On our sets the two
  correlate 0.52-0.80 and the combined least familiar pick differs in 20-33% of episodes; whether it
  is better can only be read publicly. Upload the gated one if the gate wins, the plain one if not,
  and only if time allows a Colab rebuild by about 11:00.
- Final build on Colab: done by the team lead at 08:18 (`team/final_text_colab.ipynb`,
  `text_blend_interleave`, rankings identical to the uploaded ZIP). A final with the order check
  would need a new Colab build with SigLIP (`team/export_text_order.py`).
- The site no longer gives per-episode CSVs, only the mean.

Written 2026-11-01 about 07:50 KST, after rereading every source below. This is the main reference
for strategy from now on: update it when a fact changes, rather than adding conclusions elsewhere.
Levels used: **Fact** (documented or measured), **Strong** (follows from facts with little room for
another reading), **Weak** (one reading among several), **Retracted** (we said it; it was wrong or
unsupported).

Sources: [brief.md](brief.md), the organizers' README (`docs/ko/README.md`), `cuda/Data/Train/README.md`
and `manifest.json`, `SERVER_COMPATIBILITY.md`, [validation.md](validation.md) (including the team
lead's sections on `origin/feat/fill-in-blank`), [findings.md](findings.md), and the public CSVs in
`cuda/public_csv/`.

## 1. The task (Facts)

- Each test episode gives RGB frames and candidate texts with IDs, nothing else
  (`SERVER_COMPATIBILITY.md`). We return all IDs in order; the correct text at rank 1-5 scores
  1, 0.5, 0.25, 0.125, 0. The public score is the plain mean over 573 episodes (CSV means match).
- Train: 4,170 PerceptionTest episodes, 8-20 frames, one description each, 42-75 words (median 60).
  The organizers' data README says **Codex wrote the training descriptions from the frames**, with a
  Codex reviewer and no human review. `manifest.json` records 774 negatives for 387 episodes; their
  text is not in our data.
- Test: a different benchmark, 1 correct + 4 wrong candidates. Slide examples: a reversed action
  ("buttoning" / "unbuttoning") and an added action ("and then sits down"). **The organizers'
  README also says random negatives "do not represent hard wrong answers that only change the order
  of events"** (`docs/ko/README.md`, evaluation section). Nothing says who wrote the test text.
- Constraints: 15 minutes for 1,910 episodes on a T4, ZIP up to 1 GiB, the four permitted models,
  only organizers' data plus reproducible augmentation, no distillation, final model made on Colab.

## 2. Public measurements (Facts)

| Submission | Image | Public | Correct text at rank 1 / 5 |
|---|---|---:|---|
| `text_blend` (three centrality variants) | No | 0.5521 | 41.7 / 13.8 |
| Centrality | No | 0.5515 | 40.8 / 11.3 |
| `smolvlm_gate_v1` | Rarely | 0.5502 | 40.7 / 11.5 |
| Reversed style + centrality, rank sum | No | 0.5443 | 35.3 / 7.3 |
| `smolvlm_confidence_v1` | Yes | 0.5382 | 38.7 / 11.0 |
| Reversed style (computed from `text_style`'s CSV) | No | 0.5327 | 35.3 / 10.5 |
| `q2_central4` | Slightly | 0.5312 | |
| `fill_central_v1` | Yes | 0.5044 | |
| `siglip_central_v1` | Yes | 0.4682 | |
| `q1_central05` | Yes | 0.4068 | 21.3 / 16.9 |
| Trained video rankers, SigLIP ensemble, kit baseline | Yes | 0.3469-0.3968 | |
| `temporal_qwen_evidence_v1` | Yes | 0.3370 | |
| `text_style` (most familiar wording first) | No | **0.2411** | 10.5 / 35.3 |

Chance is 0.375; one score's noise is about ±0.015-0.017, and paired comparisons are tighter.

- Centrality's and reversed style's episode scores are uncorrelated (-0.008), yet their rank sum
  didn't beat centrality.
- Against centrality: `smolvlm_gate_v1` changed 2 episodes (both worse); `smolvlm_confidence_v1`
  changed 49 (16 better, 33 worse, p = 0.021); `q2_central4` changed 118 (45 better, 73 worse).
- Every submission that let an image model move centrality's choices lost.

## 3. Local measurements worth keeping (Facts, on our data, not the test)

- A one-word edit of a 60-word description keeps its MPNet embedding at cosine 0.983 on average.
- SmolVLM separates other-scene candidates well (0.89-0.94 pair win, wrong video about 0.5) and
  one-word edits poorly (0.58-0.62 pair win, wrong video 0.45-0.50).
- Centrality on our sets: about 1.0 whenever negatives are edits of the correct text, including
  LLM-written full-length edits; 0.478 on `reorder`, where every candidate has the same words.
- Familiar-first on our sets: 0.44-0.64, never far below chance as on the test.

## 4. Claims re-evaluated

| Claim | Level now | Why |
|---|---|---|
| Most test negatives are not one-place edits of the correct text | **Strong** | Any such structure gives centrality about 0.99 locally, including LLM edits; public is 0.55 |
| The correct test text reads less like the training descriptions | **Fact** for the bigram measure | `text_style` 0.2411. The mechanism is unknown |
| Test candidates are long (~60 words) with small differences | Strong | Organizers' process; the gate's 2 changes; short candidates would have triggered it often |
| Test negatives were written by Codex / an LLM | **Retracted** | No source says so. It was a story to explain `text_style` |
| The correct text comes from a different writer | **Weak, overrated** | One of several explanations of `text_style`; others: added clauses or swapped-in verbs that are common in the training text, or reordered events that read more typically. We built the regime gate and the probes on it |
| Video can't help on this test | **Overrated** | True for everything tried: whole-text matching (SigLIP, rankers), likelihoods (SmolVLM) and their gates. None checked the *order* of events against the frames, which is the organizers' stated hard case |
| Image models fail because they learned Codex style | Weak | H was trained mostly on style-matched negatives (other Codex descriptions) and still failed. A simpler reading: when all candidates describe the same scene, whole-text matching has nothing to use, and candidates with an added action match more of the frames, which pushes some models below chance |
| Local wrong-video gaps predict public gains | **Retracted** | Every model with a large local gap lost publicly |
| Qwen-made sets can stand in for the test | Retracted | They reproduce neither centrality nor `text_style` |
| The regime gate finds the episodes where centrality fails | Weak, untested | On order-only negatives every candidate has zero unique words, so it trusts centrality exactly where centrality is near chance |
| Combining two uncorrelated text signals gives a large gain | Weak | The rank sum lost; `text_interleave`'s 0.595 is a simulation with one fitted parameter |

## 5. What we underused

1. **Order-only negatives.** The organizers name them, the brief says "the model has to reason
   over temporal order", and they are the one negative type that text cannot solve: every
   candidate has the same words, so any text-only preference (familiarity, centrality) is either
   uninformative or biased. A check that each candidate's events appear in the frames in the
   stated order is style-neutral by construction. We never built one.
2. **Event timings in `episode.json`.** Each training episode lists its actions with frame ranges
   (for example "Showing something to the camera", frames 0-145, then 178-327). That is ground truth
   for event order in the training videos, usable to check or train an order model, and unused.
3. **Length.** Added-action negatives are longer. `text_shortest` was built but never scored.
4. **Per-episode CSVs of every submission.** Only centrality's, three of ours and the lead's
   summaries are in one place; joint analysis of all of them is still to do.

## 5a. New since the first version of this audit

**Order check (Fact, local).** `team/order_check.py` splits each candidate into events, scores
each event against each frame with SigLIP (no training), and compares the best total when events
must take frames in their stated order with the total when they needn't. On 200 validation
episodes:

| Set | Right video | Wrong video | Right video reversed | Chance |
|---|---:|---:|---:|---:|
| `reorder` (5 candidates, same words, different order) | **0.495** | 0.367 | 0.371 | 0.375 |
| Qwen `changed_order` vs the correct text, pair win | **0.620** | 0.453 | 0.425 | 0.5 |

It reads event order from the frames: the right video beats the wrong one, and reversing the
frames turns the preference around. Earlier image methods were at 0.51-0.53 on action pairing.
Candidates that only reorder events share every event, so text preferences can't enter. Variants
(sentence-level events, normalized scores, strict order) were within noise or added a wrong-video
bias; the plain version is kept. About 0.07 s per episode on an RTX 5070. Unknown: how many test
episodes have order-only negatives, and whether SigLIP sees order as well in the test's videos.

As an upload, `cuda/probe_order.zip` (`team/submission_order.py`, 709 MiB, 0.06 s per episode)
ranks by order fit alone. It is a probe, not a final rule: it scores only order, so content swaps
fool it (150 episodes, right / wrong video: `reorder` 0.477 / 0.407, `swapped_action` 0.403 /
0.476, `added_action` 0.555 / 0.521; one-event candidates tie and fall to centrality). What to
read from its public CSV: how often it puts the correct text first where centrality doesn't,
against about 20-25% by chance.

**The team lead's merge analysis (`origin/main` 2bd6d28), checked so far:**

Checked with `team/merge_bounds.py`, which enumerates all 4! x 4! placements of the negatives, on
the public CSVs of centrality, `text_blend`, `text_style` and the rank sum (each CSV's mean matches
its public score exactly):

| Claim | Status |
|---|---|
| 77 episodes have the correct text first under both centrality and reversed style, 157 only under centrality, 125 only under reversed style | **Verified** (77 / 157 / 125; with `text_blend`: 77 / 162 / 125) |
| Take-turns merging scores at least 0.5550 (centrality first) / 0.5639 (`text_blend` first) | **Verified exactly.** No assumption about the negatives: it is the minimum over every placement. Independent placements give 0.5803 / 0.5879, the best case 0.6213 / 0.6274 |
| 0.5945 / 0.6016 at overlap 0.35, fitted to the rank sum's 0.5443 | Close: the fit gives 0.30 here and 0.5926 for centrality first. 16 episodes of the uploaded rank sum fall outside our tie-breaking, so the fit is rough; the bound above doesn't depend on it |
| Letting SmolVLM swap the top two above a margin of 3 (`arbiter_v1`) adds about +0.005 | The lead's own estimate, now revised to about zero (SmolVLM's confident overrides lost 16:33 publicly). Agreed |
| The final candidates' code | **Committed** by the lead (`af829bc`) and merged here, with its own Colab notebook (`team/final_text_colab.ipynb`). The style score there (bigram 0.8 + add-one unigram 0.2, with an end token) is the one measured publicly; `team/submission_regime.py` and `team/submission_fusion.py` use a different smoothing that was never uploaded |

So `text_blend_interleave` beats every uploaded rule even in its worst case, by +0.012 over
`text_blend`, and probably by about +0.04. **Uploaded: 0.5958**, inside the predicted 0.59-0.60 and
the best public score so far. The site now shows only the mean, no per-episode CSV, so later uploads
can't be analysed episode by episode.

**Training the order aligner (Fact, local).** `team/train_order.py` trains low-rank corrections and
a position prior on frozen SigLIP features, with the training descriptions as positives and code-made
reorderings as negatives. Shuffling every event let the opening summary sentence give the order away
and made validation worse (LLM-written order changes 0.605 → 0.443). Keeping the opening fixed and
swapping two events: 0.605 → 0.59-0.63 over five epochs, training accuracy 0.25-0.30. That is
within noise, so the untrained check is kept.

## 6. Open questions, cheapest resolution first

| Question | How to answer | Cost |
|---|---|---|
| Is length a signal? | Upload `probe_short.zip` | 1 upload |
| Is character-level writing a stronger signal than word pairs? | Upload `probe_chars.zip` | 1 upload |
| Does interleaving the two text signals gain? | Lead uploads `text_interleave` | 1 upload |
| Can the frames tell event order on our data? | **Answered: yes, weakly** (section 5a) | Done |
| Does the order check carry over to the test? | Upload it alone, then compare its CSV with centrality's | 1 upload |
| Are the lead's merge numbers right? | `team/merge_bounds.py` on `text_style`'s and `text_blend`'s CSVs | Local, minutes, once the CSVs are shared |

## 7. Strategy from here

1. **`text_blend_interleave` is the final unless something beats 0.5958.**
2. The order check alone (`probe_order.zip`) scored **0.4295 publicly**, about three standard errors
   above chance. Part of that may be structure rather than the frames: it favours candidates with
   fewer events even with the wrong video (see Status). It is weaker alone than either text signal
   (0.55, 0.53), so it shouldn't decide first place. Next upload:
   `text_order_third.zip` (`team/submission_text_order.py`, `order_merge` "third"), which keeps
   `text_blend_interleave`'s top two and lets the order check fill places from third on. Expected
   gain is small (+0.005 to +0.02); final if it beats 0.5958. `text_blend_interleave_gated` is
   dropped: the gate does nothing on order-only negatives.
3. Spend remaining uploads on information: shortness, characters.
4. Build the final on Colab by 10:30 with the lead's `team/final_text_colab.ipynb` (it builds
   `text_blend_interleave` with the measured style score), adding SigLIP only if the order check
   is in.
