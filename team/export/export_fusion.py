"""Create a `submission_fusion.py` submission: text signals and how to combine them.

Counts word bigrams and character n-grams (case and punctuation kept) in every training description
in the data directory; the test descriptions are never seen. `--rank-shares` takes, for each signal
used by `--method bayes`, the share of public episodes with the correct text at ranks 1-5, as read
from that signal's public per-episode CSV (team/evaluation/compare_public.py). package.py requires the kit's
files, so they are copied from an exported kit submission; they aren't loaded.

    python -m team.export.export_fusion --kit-submission cuda/sample_submission \
        --output-dir cuda/fusion_short_submission --signals short --method single
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, load_config, save_json
from dataset import load_episodes
from team.inference.submission_central import words

SIGNALS = ("centrality", "unfamiliar_words", "unfamiliar_chars", "short")
CHAR_ORDER = 5
# Public shares of episodes with the correct text at ranks 1-5 (docs/experiments/validation.md).
PUBLIC_RANK_SHARES = {
    "centrality": [0.408, 0.164, 0.175, 0.140, 0.113],
    "unfamiliar_words": [0.353, 0.239, 0.182, 0.122, 0.105],
}


def char_counts(descriptions, order):
    counts = defaultdict(Counter)
    for text in descriptions:
        padded = "\x02" * (order - 1) + text + "\x03"
        for i in range(order - 1, len(padded)):
            for length in range(order):
                counts[padded[i - length:i]][padded[i]] += 1
    return {history: {"n": sum(c.values()), "c": dict(c)} for history, c in counts.items()}


def export(config_path, kit_submission, output_dir, rule):
    kit_submission, output_dir = Path(kit_submission), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    descriptions = [e.description for e in load_episodes(load_config(config_path)["data_dir"])]
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "inference_config.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models")
    shutil.copyfile(ROOT / "team/inference/submission_central.py", output_dir / "submission_central.py")
    shutil.copyfile(ROOT / "team/inference/submission_fusion.py", output_dir / "submission.py")
    if "unfamiliar_words" in rule["signals"]:
        unigrams, bigrams = Counter(), Counter()
        for text in descriptions:
            tokens = ["<s>", *words(text), "</s>"]
            unigrams.update(tokens)
            bigrams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
        save_json(output_dir / "weights/bigram_counts.json",
                  {"prior": 1.0, "unigrams": dict(unigrams), "bigrams": dict(bigrams)})
    if "unfamiliar_chars" in rule["signals"]:
        save_json(output_dir / "weights/char_counts.json",
                  {"order": CHAR_ORDER, "counts": char_counts(descriptions, CHAR_ORDER)})
    save_json(output_dir / "weights/rule.json", rule)
    save_json(output_dir / "training_info.json", {
        "method": "text-only signals (team/inference/submission_fusion.py)", "rule": rule,
        "uses_video": False, "trained": False, "counted_descriptions": len(descriptions),
        "note": "The kit's DINOv2, MPNet and weights/mlp.pt are present only because package.py "
                "requires them; they aren't loaded."})
    print(f"Exported: {output_dir.resolve()} ({len(descriptions)} descriptions counted), "
          f"rule {json.dumps(rule)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signals", required=True, help=f"Comma-separated, in order: {SIGNALS}")
    parser.add_argument("--method", choices=["single", "interleave", "bayes"], required=True)
    parser.add_argument("--gate", action="store_true",
                        help="Centrality first when the others look like edits of one candidate.")
    parser.add_argument("--rank-shares", type=json.loads, default={},
                        help='JSON, e.g. {"short": [0.3, 0.2, 0.2, 0.15, 0.15]}; overrides the '
                             "public shares built in for centrality and unfamiliar_words.")
    args = parser.parse_args()
    signals = args.signals.split(",")
    if unknown := set(signals) - set(SIGNALS):
        parser.error(f"Unknown signals: {sorted(unknown)}")
    rule = {"signals": signals, "method": args.method, "gate": args.gate}
    if args.method == "bayes":
        shares = {**PUBLIC_RANK_SHARES, **args.rank_shares}
        if missing := [s for s in signals if s not in shares]:
            parser.error(f"--method bayes needs public rank shares for: {missing}")
        rule["rank_shares"] = {s: shares[s] for s in signals}
    export(args.config, args.kit_submission, args.output_dir, rule)


if __name__ == "__main__":
    main()
