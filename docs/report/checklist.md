# Report checklist: Fine-grained multimodal video ranking (InThon 2026)

Status and open items for the report. The report is `docs/report/report.tex` (figures in `docs/report/figures/`;
build inside `docs/report/` with `latexmk -pdf -outdir=build report.tex`, PDF in `build/`). Red `[TODO]` marks a
missing value, red `[verify]` a statement not yet rechecked. `[x]` done, `[ ]` open.

Last updated: 2026-11-13.

---

## A. Requirements (no handout: an arXiv-style technical report after the competition)

- [x] Two columns (`article`, `twocolumn`), about 6 pages including references (5 pages now).
- [x] Authors: Naufal Kamaruddin, Junu Noh (노준우), Kyounghoon Han (한경훈), Suhong Min (민수홍); Korea University.
      Kyounghoon Han is from his GitHub profile; Junu Noh and Suhong Min confirmed by Naufal (2026-11-13).
- [ ] **Each co-author agrees before it is posted anywhere.**
- [x] AI use: the competition rules restrict AI only for generating data (`docs/competition/brief.md`). No
      acknowledgement of drafting help (Naufal's decision, 2026-11-13).
- [x] No text from the organizers' data is quoted: the task figure is an illustration, and no episode excerpts appear.

## B. Terms (code name → report term; full list in `docs/glossary.md`)

| In our code and notes | In the report |
|---|---|
| centrality, `text_centrality` | medoid / minimum Bayes risk (MBR) selection |
| `text_blend` | medoid selection with three similarities fused by rank sum (Borda count) |
| familiarity, `text_style`, reversed style | n-gram language-model likelihood; lowest likelihood first |
| interleave, take turns | round-robin rank fusion |
| gate | rule-based switch |
| own words | unaligned tokens |
| event-order check, `probe_order` | monotonic event-to-frame alignment |
| text-only leak | hypothesis-only (text-only) baseline; annotation artifacts |
| merge bounds | worst-case bound by exhaustive enumeration |
| negatives | generated hard negatives |
| dev / validation episodes | held-out validation split (834 episodes) |
| public / private | public leaderboard (573 episodes) / private leaderboard (1,337 episodes) |

## C. Outline and where the numbers come from

**Challenges** (the spine: each method answers one, each result row names one):
- **C1 No wrong candidates for training.** The training data has one correct description per episode and no
  negatives, so negatives must be generated, and local validation measures our generator.
- **C2 Near-identical candidates.** Candidates differ in one detail (reversed or added action, order); a one-word
  edit of a 60-word description keeps its MPNet embedding at cosine 0.983.
- **C3 Benchmark shift.** The test comes from a different benchmark, with candidates written by an unknown
  process; text cues that hold locally reverse on the test.
- **C4 Scarce, noisy feedback.** 30 leaderboard checks on 573 public episodes, about ±0.016 per score.

| Section | Pages | Holds |
|---|---:|---|
| Abstract, 1 Introduction | 1 | task, C1–C4, what we did and found; Fig. 1 (task) |
| 2 Related work | 0.5 | fine-grained video-text benchmarks, encoders used, annotation artifacts, MBR, rank fusion |
| 3 Task, data and evaluation | 0.75 | data, score, chance, noise, generated negatives + text-only check (Table 1), how the leaderboard was used |
| 4 Video-based ranking | 1 | trained rankers (Table 2), zero-shot SigLIP, SmolVLM differing tokens, monotonic alignment; why they failed |
| 5 Text-only ranking | 1 | medoid (MBR) selection, LM likelihood, round-robin fusion, rule-based switch, worst-case bounds (Table 4) |
| 6 Results | 0.75 | Table 3 (public scores), Fig. 2, private score, paired comparisons |
| 7 Discussion and limitations | 0.5 | annotation artifacts, what the video signal lacked, what the rule does not do |
| 8 Conclusion, references, appendix | 0.5 | reproduction |

| Table / figure | Source |
|---|---|
| Fig. 1 task illustration | `docs/figures/make_figures.py` (no data) |
| Fig. 2 score progression | `docs/figures/make_figures.py` ← `docs/experiments/validation.md` "Public scores" |
| Table 1 negative types, text-only scores | `validation.md` "Text-only leak check" |
| Table 2 trained video rankers A–H (local), public b / h-v1 / word_swap | `validation.md` "Ablation results", "Public scores" |
| Table 3 public scores, rank-1/5 shares | `validation.md` "Public scores", "Writing style and its reversal"; `audit.md` §2 |
| Table 4 fusion simulation, worst case | `validation.md` "Merging the text rules" |
| Private score 0.6417 | combined 0.640772251309 (organizers) and public 2,927 / 4,584 |

## D. Still to fill in

- [ ] Co-authors' agreement.

## E. Case studies

None with real text: the organizers' data is not ours to publish. The task figure is an illustration.

## F. Before posting

Rigor:
- [x] Every rule was chosen with public-leaderboard statistics; the report says so, and gives the private score as
      the check on overfitting.
- [x] No "significant" without a test; no claim from a difference inside the noise (±0.016 per score, paired
      0.005–0.013).
- [x] Every number matches `docs/experiments/validation.md` or `audit.md` (check script: 121 of 121 traced).
- [x] Only permitted models named; the distillation and data rules stated.
- [x] No code names in the text; terms as in part B.

Presentation:
- [x] Figures vector or 200-dpi; same decimals per quantity (4 for public scores, 3 for local).
- [x] No `\todo` / `\verify` left; within about 6 pages (5).
