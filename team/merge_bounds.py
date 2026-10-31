"""Score ways of merging two rankings from their public per-episode CSVs, without uploading.

A CSV gives only the correct text's rank under each rule, so for each episode the correct text sits
at rank a under rule A and b under rule B, and the four negatives' places are unknown. For every
pair (a, b) this enumerates all 4! x 4! placements of the negatives and scores the merged ranking:

- worst / best: the lowest and highest score any placement allows,
- independent: the mean when the two rules order the negatives independently,
- overlap rho: with probability rho rule B orders the negatives as rule A does.

`rank_sum` is scored the same way so that rho can be checked against its public score.

    python -m team.merge_bounds cuda/public_csv/central_submission.csv cuda/public_csv/text_style.csv \
        --reverse-second --observed-rank-sum 0.5443
"""

import argparse
import itertools
import statistics
from functools import lru_cache
from pathlib import Path

from team.compare_public import RANK, read_scores

SCORES = (1.0, 0.5, 0.25, 0.125, 0.0)


def take_turns(first, second):
    merged = []
    for pair in zip(first, second):
        for item in pair:
            if item not in merged:
                merged.append(item)
    return merged


def rank_sum(first, second):
    # Ties go to the second rule's order, as the uploaded rank sum's shares suggest.
    return sorted(range(5), key=lambda i: (first.index(i) + second.index(i), second.index(i)))


MERGES = {"take_turns": take_turns, "rank_sum": rank_sum}


def orders_with(correct_rank, negatives_order):
    """A full order of items 0-4, where item 0 is the correct text."""
    order = list(negatives_order)
    order.insert(correct_rank - 1, 0)
    return order


@lru_cache(maxsize=None)
def outcomes(merge, a, b):
    """Scores of the merged ranking for every placement: (same negative order, all placements)."""
    same, every = [], []
    for first_negatives in itertools.permutations((1, 2, 3, 4)):
        first = orders_with(a, first_negatives)
        for second_negatives in itertools.permutations((1, 2, 3, 4)):
            second = orders_with(b, second_negatives)
            score = SCORES[MERGES[merge](first, second).index(0)]
            every.append(score)
            if second_negatives == first_negatives:
                same.append(score)
    return tuple(same), tuple(every)


def expected(merge, pairs, rho):
    total = 0.0
    for a, b in pairs:
        same, every = outcomes(merge, a, b)
        total += rho * statistics.fmean(same) + (1 - rho) * statistics.fmean(every)
    return total / len(pairs)


def bounds(merge, pairs):
    worst = sum(min(outcomes(merge, a, b)[1]) for a, b in pairs) / len(pairs)
    best = sum(max(outcomes(merge, a, b)[1]) for a, b in pairs) / len(pairs)
    return worst, best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", type=Path, help="CSV of the rule that goes first.")
    parser.add_argument("second", type=Path)
    parser.add_argument("--reverse-second", action="store_true",
                        help="Use the second rule reversed (rank r becomes 6 - r).")
    parser.add_argument("--observed-rank-sum", type=float,
                        help="Public score of the 1:1 rank sum, to fit rho.")
    args = parser.parse_args()
    first, _ = read_scores(args.first)
    second, _ = read_scores(args.second)
    shared = sorted(set(first) & set(second))
    pairs = []
    for episode in shared:
        b = RANK[second[episode]]
        pairs.append((RANK[first[episode]], 6 - b if args.reverse_second else b))
    both = sum(a == 1 and b == 1 for a, b in pairs)
    only_first = sum(a == 1 and b != 1 for a, b in pairs)
    only_second = sum(a != 1 and b == 1 for a, b in pairs)
    print(f"{len(pairs)} episodes; correct first under both {both}, only the first rule "
          f"{only_first}, only the second {only_second}")
    print(f"first rule alone {statistics.fmean(SCORES[a - 1] for a, _ in pairs):.4f}, "
          f"second alone {statistics.fmean(SCORES[b - 1] for _, b in pairs):.4f}")
    rho_fit = None
    if args.observed_rank_sum is not None:
        rho_fit = min((abs(expected("rank_sum", pairs, r / 100) - args.observed_rank_sum), r / 100)
                      for r in range(0, 101, 5))[1]
        print(f"rho fitted to the rank sum's public {args.observed_rank_sum}: {rho_fit:.2f}")
    for merge in MERGES:
        worst, best = bounds(merge, pairs)
        line = (f"{merge:10s} worst {worst:.4f}  independent {expected(merge, pairs, 0):.4f}  "
                f"best {best:.4f}")
        if rho_fit is not None:
            line += f"  at rho {rho_fit:.2f}: {expected(merge, pairs, rho_fit):.4f}"
        print(line)


if __name__ == "__main__":
    main()
