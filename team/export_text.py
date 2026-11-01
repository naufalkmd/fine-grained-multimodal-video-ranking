"""Create a text-rule submission (no trained weights) from an exported kit submission.

package.py requires the encoder files, so they are copied from the base submission even when the rule
never loads them. `--method` picks the submission:

- `blend`: team/submission_text_rules.py, the rank sum of three similarity rules (text_blend).
- `interleave`: team/submission_style.py, centrality and reversed word-pair familiarity in turn
  (text_interleave). Word-pair counts are built here from the training descriptions.
- `blend_interleave`: team/submission_text_rules.py, `blend` and reversed word-pair familiarity in turn
  (text_blend_interleave), with the same counts.
- `blend_interleave_gated`: the same, but reversed familiarity goes first when no candidate looks like the one
  the others were edited from (no candidate has all its words in the others).

    python -m team.export_text --method blend_interleave --base-dir cuda/sample_submission \
        --data-dir cuda/Data/Train --output-dir cuda/text_blend_interleave_submission
"""

import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, save_json
from dataset import load_episodes
from team.submission_style import build_counts


def export(method, base_dir, data_dir, output_dir):
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    shutil.copytree(base_dir, output_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "submission.py", "training_info.json",
                                                  "inference_config.json"))
    if method != "interleave":
        shutil.copyfile(ROOT / "team/submission_style.py", output_dir / "submission_style.py")
    if method == "blend":
        source, config = "submission_text_rules.py", {"rule": "blend"}
        note = "Text rule; ignores the video and has no trained weights."
    else:
        if method == "interleave":
            source, config = "submission_style.py", {"merge": "interleave", "style_sign": -1}
        else:
            source, config = "submission_text_rules.py", {"rule": "blend", "merge": "interleave",
                                                          "gate": method == "blend_interleave_gated"}
        note = "Word-pair counts from the organizers' training descriptions; ignores the video."
        descriptions = [episode.description for episode in load_episodes(data_dir)]
        (output_dir / "models/style").mkdir(parents=True)
        (output_dir / "models/style/counts.json").write_text(
            json.dumps(build_counts(descriptions)), encoding="utf-8")
    shutil.copyfile(ROOT / "team" / source, output_dir / "submission.py")
    save_json(output_dir / "inference_config.json", config)
    save_json(output_dir / "training_info.json", {"method": method, "uses_video": False, "trained": False, "note": note})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["blend", "interleave", "blend_interleave", "blend_interleave_gated"],
                        required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.method, args.base_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
