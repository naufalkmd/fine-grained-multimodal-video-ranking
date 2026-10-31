# Validation

How we measure experiments. This is the team's own approach, not an organizer
requirement. The competition rules are in [brief.md](brief.md).

Status: planned. The kit's `evaluate.py` still reports BCE and pair accuracy on
random negatives, so it does not produce this score yet.

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
