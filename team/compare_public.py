"""Compare public per-episode CSVs from the scoring site against a baseline submission.

Each CSV has `episode_id,score,error_log`, and a score tells the rank of the correct text (1, 0.5,
0.25, 0.125 or 0 for ranks 1-5). Reports each file's mean and rank distribution, and against the
baseline the paired mean difference with a 95% interval, how many episodes changed, a two-sided sign
test, and the most common rank moves. Use the CSVs in aggregate only: never tune to single episodes.

    python -m team.compare_public cuda/public_csv/central_submission.csv cuda/public_csv/*.csv
"""

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

RANK = {1.0: 1, 0.5: 2, 0.25: 3, 0.125: 4, 0.0: 5}


def read_scores(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    errors = sum(bool(row.get("error_log")) for row in rows)
    return {row["episode_id"]: float(row["score"]) for row in rows}, errors


def sign_test(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def describe(name, scores, errors):
    ranks = Counter(RANK[value] for value in scores.values())
    shares = " / ".join(f"{100 * ranks[r] / len(scores):.1f}" for r in range(1, 6))
    print(f"{name}: mean {statistics.fmean(scores.values()):.4f}, {len(scores)} episodes, "
          f"{errors} errors, rank 1-5 % {shares}")


def compare(base_name, base, name, other):
    shared = sorted(set(base) & set(other))
    differences = [other[e] - base[e] for e in shared]
    mean = statistics.fmean(differences)
    error = statistics.stdev(differences) / math.sqrt(len(differences))
    better = sum(d > 0 for d in differences)
    worse = sum(d < 0 for d in differences)
    moves = Counter((RANK[base[e]], RANK[other[e]]) for e in shared if base[e] != other[e])
    print(f"  vs {base_name}: {mean:+.4f} [{mean - 1.96 * error:+.4f}, {mean + 1.96 * error:+.4f}], "
          f"changed {better + worse}, better {better}, worse {worse}, "
          f"sign test p = {sign_test(better, worse):.3f}")
    first_won = sum(1 for e in shared if base[e] != 1.0 and other[e] == 1.0)
    first_lost = sum(1 for e in shared if base[e] == 1.0 and other[e] != 1.0)
    print(f"  correct text newly first in {first_won} episodes, no longer first in {first_lost}")
    common = ", ".join(f"{a}->{b}: {count}" for (a, b), count in moves.most_common(6))
    print(f"  most common rank moves: {common}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("others", type=Path, nargs="+")
    args = parser.parse_args()
    base, errors = read_scores(args.baseline)
    describe(args.baseline.stem, base, errors)
    for path in args.others:
        if path.resolve() == args.baseline.resolve():
            continue
        scores, errors = read_scores(path)
        describe(path.stem, scores, errors)
        compare(args.baseline.stem, base, path.stem, scores)


if __name__ == "__main__":
    main()
