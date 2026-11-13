# Fine-grained multimodal video ranking

Team 07's entry to the **InThon 2026 Datathon** (final submission November 1, 2026). Given the frames of a short
video and five candidate descriptions that differ in a single detail, rank the descriptions from most to
least likely.

**Result:** **0.6408 on all 1,910 test episodes** (chance 0.375) with a text-only ranking rule: 0.6385 on
the public 30% and 0.6417 on the private 70%. The organizers reported the combined score; the private
score follows from it and the public one.
The more useful result is *why* it is text-only: every video model we built reached about 0.8 on our own
validation and fell to chance on the test set. The sections below explain what went wrong and what worked
instead.

## The task

| | |
|---|---|
| Input per episode | 8–20 RGB frames and 5 candidate descriptions (about 60 words each) |
| Output | The 5 candidates in ranked order |
| Score per episode | 1, 0.5, 0.25, 0.125, 0 for the correct description at rank 1–5; the leaderboard is the mean |
| Training data | 4,170 episodes from the PerceptionTest video benchmark, **one correct description each, no wrong ones** |
| Test data | 1,910 episodes from a **different benchmark** (30% public, 70% private), 1 correct + 4 wrong candidates |
| Constraints | Only four permitted pretrained models (DINOv2-base, all-mpnet-base-v2, SigLIP-base, SmolVLM-256M); inference for all 1,910 episodes in 15 minutes on one T4, offline; submission ZIP ≤ 1 GB; no distillation from other models; final model built on Colab |

The organizers' examples show what makes it hard. The wrong candidates describe the *same* scene as the
correct one and change one thing: a reversed action ("buttoning" vs "unbuttoning") or an added action
("... and then sits down"). Telling them apart needs the order of events in the video, not just its content.

## What we learned

1. **Local validation measured the wrong skill.** With no wrong descriptions in the training data, we made
   our own: mostly other videos' descriptions. A model passes those by matching objects in the scene. That
   skill is useless when all five candidates describe the same scene, so models at about 0.8 locally scored
   0.35–0.40 on the public set.
2. **Sentence embeddings hide one-word differences.** A one-word edit of a 60-word description keeps its
   MPNet embedding at cosine similarity 0.983 on average. Any model that compares one sentence vector with
   the video cannot see the edit.
3. **Every image signal that changed the text ranking lost.** Trained video rankers, SigLIP ensembles and
   a fine-tuned SmolVLM all scored at or below the text-only rules. The one image method with a real
   order signal, a training-free check that a candidate's events match the frames in their stated order
   (SigLIP per event, dynamic programming over frames), scored 0.4295 alone, above chance, but added
   nothing when combined.
4. **The candidates' text carries signal of its own.** Two text-only scores worked, and they were
   uncorrelated per episode:
   - **Centrality:** the candidate most similar to the other four comes first. If the wrong
     candidates are edits of the correct one, the correct one is the text they share.
   - **Reversed familiarity:** the candidate whose wording is *least* like the training descriptions comes
     first (a word-bigram model fitted on the training text). Ranking the most familiar first scored
     0.2411, far below chance, so the direction itself is a measured fact; its cause is not known.

   Taking turns between the two rankings, and a gate that decides which goes first, gave the final rule.
5. **Bounds before uploads.** With 30 leaderboard checks, we computed the worst case of a merge exactly from
   the per-episode public results before uploading it (`team/merge_bounds.py`): taking turns could not score
   below 0.5639, and it scored 0.5958.

## How the score improved

| Submission (public leaderboard) | Uses the video | Public |
|---|---|---:|
| Organizers' baseline, trained video rankers, SigLIP ensemble | Yes | 0.347–0.397 |
| Pretrained SigLIP + centrality | Yes | 0.4682 |
| Event-order check alone (SigLIP) | Yes | 0.4295 |
| Reversed familiarity alone | No | 0.5327 |
| Centrality alone | No | 0.5515 |
| Centrality, three measures combined (`text_blend`) | No | 0.5521 |
| Take turns: `text_blend` and reversed familiarity | No | 0.5958 |
| + gate: reversed familiarity first when no candidate looks like the one the others were edited from | No | 0.6368 |
| + gate uses familiarity of each candidate's own words (**final**) | No | **0.6385** |

One public score is uncertain by about ±0.016, so the last step (+0.0017) is within noise; it became the
final because it was the highest score. On the private 70% the final scored 0.6417, slightly above its
public score, so the rule did not overfit the public leaderboard. Every number, with its settings and the validation data behind
it, is in [docs/experiments/validation.md](docs/experiments/validation.md).

## Final method

For each episode, from the five candidate texts alone:

1. **`text_blend`:** rank by centrality, as the sum of three ranks: the share of each candidate's words
   found in the other candidates, its mean MPNet similarity to them, and word-sequence similarity.
2. **Reversed familiarity:** rank by how *unlike* the training descriptions each candidate's wording is,
   from a word-bigram model fitted on the training text. When the gate below fires, only the word pairs
   that touch a candidate's own words (words that match nothing in the other candidates under a
   word-by-word alignment) are scored.
3. **Gate:** if some candidate has all its words in the others, it looks like the base text the others
   were edited from, and `text_blend`'s pick goes first. Otherwise reversed familiarity's pick goes first.
4. **Interleave:** fill the ranking by taking the next unused candidate from each list in turn.

It runs in about 0.02 s per episode. Code: `team/submission_text_rules.py` and `team/text_rules.py`.

## Repository layout

```text
.
├── *.py, config.json      organizers' starter kit: baseline model, training, export, packaging, submission entry point
├── tests/                 the kit's tests
├── team/                  our code (run with python -m team.<module> from the root); index in team/README.md
│   ├── tests/             tests for our code
│   └── siglip_ensemble/   SigLIP training and the three-seed ensemble
├── notebooks/             Colab notebooks: the organizers' and our two final-build notebooks
├── docs/
│   ├── starterkit/        the kit's docs in English, Korean originals in ko/
│   ├── competition/       the rules we worked under and their changes
│   ├── experiments/       every experiment, the findings and the end-of-competition audit
│   └── competition-day/   handoff notes and the presentation outline
└── cuda/                  local GPU setup; data, models and outputs stay out of git
```

The kit's modules stay at the root because the submission ZIP needs `submission.py` at its top level and our
code imports the kit's `common.py` and `dataset.py`.

| Start here | For |
|---|---|
| [team/README.md](team/README.md) | What each of our scripts does |
| [docs/experiments/validation.md](docs/experiments/validation.md) | Every experiment: settings, validation data, local and public scores |
| [docs/experiments/audit.md](docs/experiments/audit.md) | The end-of-competition review: each claim graded as fact, strong, weak or retracted |
| [docs/competition/brief.md](docs/competition/brief.md) | The competition rules we worked under |
| [docs/starterkit/README.md](docs/starterkit/README.md) | The organizers' baseline and how a submission is built |

## Reproducing the final submission

The competition data is the organizers' and is not included. With it in Google Drive at
`MyDrive/data/Train.zip`:

1. Open `notebooks/final_text_colab.ipynb` in Colab (T4 runtime).
2. In section 2, set `METHOD = "blend_interleave_gated_own"` (the final rule), then run all cells. The notebook
   downloads the permitted models, builds the submission ZIP (about 349 MiB) and records the run.

Locally, on an NVIDIA GPU: set up the evaluation environment from [cuda/README.md](cuda/README.md)
(`cuda/requirements-eval.txt`, the package versions the organizers' server used), then run the export
scripts in `team/` (see [team/README.md](team/README.md)). The root `requirements.txt` is the kit's
original list (Python 3.11, torch 2.5.1) and lacks packages our code needs, such as scikit-learn.

## Team

- Naufal Kamaruddin ([@naufalkmd](https://github.com/naufalkmd)): team lead; starter-kit translation, the
  submission rules (interleaving, the gate and its variants), the final Colab build
- 노준우 ([@njunu17](https://github.com/njunu17)): the first text rules (centrality, familiarity), SmolVLM
  experiments, the event-order check, merge bounds, the audit and the validation log
- 한경훈 ([@opolo17](https://github.com/opolo17)): SigLIP training and the three-seed SigLIP ensemble
  (`team/siglip_ensemble/`)
- 민수홍 ([@m2nx6](https://github.com/m2nx6)): SigLIP hybrid ranker (`team/exp6_submission.py`)

The baseline code at the repository root is the organizers' starter kit; commit `a9a4179` is the kit as we
received it.
