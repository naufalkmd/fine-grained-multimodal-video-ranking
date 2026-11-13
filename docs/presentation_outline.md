# Presentation outline (draft)

> Competition-day draft, kept as written. The **[PENDING]** items were never filled in here; the final numbers are in
> the [README](../README.md) (final: 0.6385 public, 0.6408 on the full test set).

For the judges. Every number is from `validation.md`. **[PENDING]** marks what the final choice or a public score
still has to fill in. The rules set no format, only the 12:30 deadline, and missing it rules out awards.

## 1. Title
- Team 07, InThon 2026 Datathon: ranking five near-identical descriptions of a video.
- Final public score: **[PENDING]**.

## 2. The task
- Input: a frame sequence and five candidate descriptions. Output: all five ranked.
- Score by the correct answer's rank: 1, 0.5, 0.25, 0.125, 0. Random ranking earns 0.375.
- Train: PerceptionTest, correct descriptions only. Test: a different benchmark, with four wrong answers we never see.

## 3. We built the test before the model
- Eight types of generated wrong answers, each reported separately.
- Three checks on every result: can text alone solve it, does the score fall with the wrong video, does it notice
  reversed frames.
- Checkpoint selection was frozen in advance, so we couldn't tune to our own test.

## 4. Strong local scores, weak public scores
- Temporal Transformer over frozen DINOv2 and MPNet: about 0.8 locally on content-based wrong answers, 0.36 to 0.40 publicly.
- Baseline `b` 0.3469, `h-v1` 0.3586, `temporal_word_swap_v1` 0.3968.
- Why: a one-word edit in 60 words leaves MPNet similarity at 0.983, so the candidates look identical to the model.
  Our generated wrong answers also differ from the real ones.

## 5. Discovery 1: the correct answer sits at the centre
- If wrong answers are edits of the correct one, it is the most similar to all the others.
- Centrality, with no video: **0.5515**. Two other similarity rules tied: `text_blend` 0.5521, `smolvlm_gate_v1` 0.5502.

## 6. Discovery 2: a very bad score is information
- "Reads most like the training text" scored **0.2411**, far below chance, about nine times the noise.
- Reversed, using the leaderboard's per-episode results: **0.5327**, with almost no correlation with centrality (-0.008).
- We don't know the cause. Our guess: invented wrong details are more typical than what a real video shows.

## 7. How you combine signals matters
- Adding ranks lost first places, 40.8% to 35.3%: **0.5443**. Both rules seem to favour the same wrong answer.
- Taking turns keeps the first rule's top pick and gives second place to the other rule.
- `text_blend_interleave`: **[PENDING]**. From per-episode results alone, about 0.564 at worst; simulated about 0.60.

## 8. Where the video fits
- Image models mostly hurt: SigLIP + centrality 0.4682, image evidence 0.3370.
- Fine-tuned SmolVLM, on only the words that differ: chooses correctly in 63% of pairs with the right video and 46%
  with the wrong one (locally, on `word_swap`). With centrality at equal weight: **0.5044** publicly, our best
  equal-weight image result, but below centrality.
- Using the video only when it clearly favours the other rule's top pick (`arbiter_v1`): **[PENDING]**. Locally, it
  overrules rarely, and it does harm with the wrong video but not with the right one (`swapped_action`: 0.970 right, 0.810 wrong).

## 9. Rules and reproducibility
- Permitted models only. No distillation. Data comes from the organizers' descriptions and code-generated edits.
- Each episode is ranked from its own input, with no updates during inference, offline.
- The final ZIP is built on a Colab T4: **[PENDING: confirm the run]**. The notebook checks that its word counts match
  the team's ZIP.

## 10. Limits
- **[If the final uses no video, say so plainly.]** It relies on how the candidates were written, so it wouldn't
  carry over to a test built differently, and it doesn't reason about the video as the task intends.
- Next step: a video model that can reliably separate two near-identical sentences.
