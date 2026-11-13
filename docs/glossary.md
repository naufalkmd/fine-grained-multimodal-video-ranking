# Glossary: our terms and their standard names

During the competition we named methods as we built them. The README and module docstrings use the
standard name first; the competition-day records in [experiments/](experiments/) and
[competition-day/](competition-day/) keep the names as written at the time. Code identifiers, config keys
and `--method` values keep our names, because the submitted configuration depends on them.

## Text-only ranking

| Our term (code name) | Standard name | What it is here |
|---|---|---|
| centrality (`central`, `centrality`) | **medoid / minimum Bayes risk (MBR) selection** [1] | Each candidate's mean similarity to the other four; the highest comes first. If the wrong candidates are edits of the correct one, the correct one is the consensus of the set. |
| consensus | **token overlap** with the other candidates | The share of a candidate's words that occur in each other candidate, averaged |
| meaning | **mean embedding cosine similarity** | Mean MPNet cosine similarity to the other candidates |
| word-sequence similarity | **Ratcliff/Obershelp (gestalt pattern matching) similarity** | `difflib.SequenceMatcher.ratio()` over word tokens |
| `text_blend`, "three measures combined" | **rank-sum fusion (Borda count)** | The sum of the ranks from consensus, meaning and word-sequence similarity |
| familiarity, style (`style`, `submission_style`) | **n-gram language-model likelihood** | Mean log-probability of a candidate's word pairs under an interpolated bigram model fitted on the training descriptions |
| reversed / unfamiliar wording | **lowest-likelihood (highest-perplexity) first** | The same score, ranked in the opposite direction; ranking the most likely first scored 0.2411, far below chance |
| character familiarity (`chars`) | **character n-gram language-model likelihood** (Witten-Bell interpolation) | The same idea on characters, keeping case and punctuation |
| own words (`gate_own_words`) | **unaligned tokens** (the candidate's **edit span**) | Words that match nothing in the other candidates under a word-level alignment |
| interleave, take turns (`interleave`, `merge`) | **round-robin rank fusion** | Fill the ranking by taking the next unused candidate from each of two rankings in turn |
| gate (`gate`), regime gate | **rule-based switch** | Decides which ranking goes first: the likelihood ranking when no candidate has all its words in the others (no candidate looks like the source of the edits) |
| base text, "edited from" | **source sentence** of the edits | The candidate the other four look like edits of |
| the candidates' text "carries signal of its own", text-only leak | **annotation artifacts**, measured with a **hypothesis-only (text-only) baseline** [2, 3] | Text-only rules beat chance without the video, so how the candidates were written gives the answer away in part |

## Video and evaluation

| Our term (code name) | Standard name | What it is here |
|---|---|---|
| event-order check (`order_check`, `submission_order`) | **monotonic (order-preserving) alignment** by dynamic programming, as in dynamic time warping | SigLIP scores each event against each frame; the best total when events keep their stated order is compared with the total when they needn't |
| order aligner (`train_order`) | learned **monotonic alignment** model | Low-rank corrections and a position prior trained on frozen SigLIP features |
| action pairing (`action_pairing`) | **reversed-video consistency test** | Does reversing the frames flip the choice between a description and its reversed-action version? |
| hub-corrected word overlap (`lexical`, `_csls`) | **hubness reduction (CSLS)** [4] | Word-overlap similarity corrected for texts that are similar to everything |
| merge bounds, "bounds before uploads" (`merge_bounds`) | **worst-case bound by exhaustive enumeration** | The minimum score of a rank fusion over every placement of the unseen wrong candidates, from per-episode public results |
| public rank shares (`bayes`) | **empirical-Bayes prior** from leaderboard rank frequencies | Weights from how often each rule put the correct answer at each rank publicly |
| negative types (`word_swap`, `similar_episode`, `reorder`, ...) | **hard negatives** of named kinds | Generated wrong descriptions used for training and validation |

## References

1. S. Kumar and W. Byrne. Minimum Bayes-Risk Decoding for Statistical Machine Translation. NAACL 2004.
   [ACL Anthology N04-1022](https://aclanthology.org/N04-1022/)
2. S. Gururangan et al. Annotation Artifacts in Natural Language Inference Data. NAACL 2018.
   [arXiv:1803.02324](https://arxiv.org/abs/1803.02324)
3. A. Poliak et al. Hypothesis Only Baselines in Natural Language Inference. *SEM 2018.
   [arXiv:1805.01042](https://arxiv.org/abs/1805.01042)
4. A. Conneau et al. Word Translation Without Parallel Data. ICLR 2018.
   [arXiv:1710.04087](https://arxiv.org/abs/1710.04087)
