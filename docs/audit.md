# Audit: what we know, how sure we are, and what to do next

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

## 6. Open questions, cheapest resolution first

| Question | How to answer | Cost |
|---|---|---|
| Is length a signal? | Upload `probe_short.zip` | 1 upload |
| Is character-level writing a stronger signal than word pairs? | Upload `probe_chars.zip` | 1 upload |
| Does interleaving the two text signals gain? | Lead uploads `text_interleave` | 1 upload |
| Can the frames tell event order on our data? | SigLIP sentence-to-frame alignment on `reorder` and Qwen `changed_order`, right vs wrong video | Local, about 1 hour |
| How many test episodes are order-only? | Not observable directly; the order check's public result will show it | 1 upload, after the local check |

## 7. Strategy from here

1. Keep centrality as the safe final until something beats it in a paired comparison.
2. Locally, build and test an order check (section 5.1). It is the most direct route to the score
   the task intends, and the only image method not yet tried.
3. In parallel, spend uploads on information: interleave (lead), shortness, characters.
4. Combine only signals that won publicly on their own, with a rule chosen before seeing the
   combination's score, and confirm it against centrality's CSV.
5. Leave the final Colab run for 10:30 at the latest (`team/final_colab_text.ipynb`, plus any image
   model it needs).
