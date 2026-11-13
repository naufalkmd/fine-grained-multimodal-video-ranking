"""Create the text-only `submission_regime.py` submission, with bigram counts from the descriptions.

Counts every training description in the data directory (the test descriptions are never seen).
package.py requires the kit's files, so they are copied from an exported kit submission; they
aren't loaded at inference.

    python -m team.export.export_regime --kit-submission cuda/sample_submission \
        --output-dir cuda/regime_v1_submission
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from common import ROOT, load_config, save_json
from dataset import load_episodes
from team.inference.submission_central import words

PRIOR = 1.0


def export(config_path, kit_submission, output_dir, gate=True):
    kit_submission, output_dir = Path(kit_submission), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    descriptions = [e.description for e in load_episodes(load_config(config_path)["data_dir"])]
    unigrams, bigrams = Counter(), Counter()
    for text in descriptions:
        tokens = ["<s>", *words(text), "</s>"]
        unigrams.update(tokens)
        bigrams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "inference_config.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models")
    shutil.copyfile(ROOT / "team/inference/submission_central.py", output_dir / "submission_central.py")
    shutil.copyfile(ROOT / "team/inference/submission_regime.py", output_dir / "submission.py")
    save_json(output_dir / "weights/bigram_counts.json",
              {"prior": PRIOR, "gate": gate, "unigrams": dict(unigrams), "bigrams": dict(bigrams)})
    save_json(output_dir / "training_info.json", {
        "method": ("structure-gated interleave" if gate else "interleave")
                  + " of centrality and unfamiliar wording (team/inference/submission_regime.py)",
        "uses_video": False, "trained": False, "bigram_descriptions": len(descriptions),
        "note": "The kit's DINOv2, MPNet and weights/mlp.pt are present only because package.py "
                "requires them; they aren't loaded."})
    print(f"Exported text-only submission: {output_dir.resolve()} "
          f"({len(descriptions)} descriptions counted)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-gate", action="store_true",
                        help="Always put centrality first (plain interleaving).")
    args = parser.parse_args()
    export(args.config, args.kit_submission, args.output_dir, gate=not args.no_gate)


if __name__ == "__main__":
    main()
