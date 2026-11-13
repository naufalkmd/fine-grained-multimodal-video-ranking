"""Create a text-rule submission (no trained weights) from an exported kit submission.

package.py requires the encoder files, so they are copied from the base submission even when the rule
never loads them. `--method` picks the submission:

- `blend`: team/inference/submission_text_rules.py, the rank sum of three similarity rules (text_blend).
- `interleave`: team/inference/submission_style.py, centrality and reversed word-pair familiarity in turn
  (text_interleave). Word-pair counts are built here from the training descriptions.
- `blend_interleave`: team/inference/submission_text_rules.py, `blend` and reversed word-pair familiarity in turn
  (text_blend_interleave), with the same counts.
- `blend_interleave_gated`: the same, but reversed familiarity goes first when no candidate looks like the one
  the others were edited from (no candidate has all its words in the others).
- `blend_interleave_chars` / `blend_interleave_gated_chars`: as `blend_interleave` / `blend_interleave_gated`, with
  the least familiar wording judged by word pairs and characters together (team/inference/submission_chars.py).
- `blend_interleave_gated_excess`: also reversed familiarity first when even the best candidate's word edit distances
  to the others don't add up. The threshold, 0.3, was fixed on synthetic candidates, not on public scores.
- `blend_interleave_gated_own`: when the gate fires, the first pick comes from a candidate's own words only.
- `blend_interleave_gated_mean`: when the gate fires, the first list averages whole-text and own-word unfamiliarity.
- `blend_interleave_gated_mean_chars`: the same, with whole-text unfamiliarity judged by word pairs and characters.
- `blend_interleave_gated_base_first`: when it doesn't fire, candidates with no own words come before the others.

    python -m team.export.export_text --method blend_interleave --base-dir cuda/sample_submission \
        --data-dir cuda/Data/Train --output-dir cuda/text_blend_interleave_submission
"""

import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, save_json
from dataset import load_episodes
from team.inference.submission_chars import build_char_counts
from team.inference.submission_style import build_counts


def export(method, base_dir, data_dir, output_dir):
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    shutil.copytree(base_dir, output_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "submission.py", "training_info.json",
                                                  "inference_config.json"))
    if method != "interleave":
        shutil.copyfile(ROOT / "team/inference/submission_style.py", output_dir / "submission_style.py")
    if method == "blend":
        source, config = "submission_text_rules.py", {"rule": "blend"}
        note = "Text rule; ignores the video and has no trained weights."
    else:
        if method == "interleave":
            source, config = "submission_style.py", {"merge": "interleave", "style_sign": -1}
        else:
            source, config = "submission_text_rules.py", {"rule": "blend", "merge": "interleave",
                                                          "gate": "gated" in method}
            if method == "blend_interleave_gated_excess":
                config["gate_excess"] = 0.3
            if method.endswith("_chars"):
                config["chars"] = True
            if method == "blend_interleave_gated_own":
                config["gate_own_words"] = True
            if method in ("blend_interleave_gated_mean", "blend_interleave_gated_mean_chars"):
                config["gate_mean"] = True
            if method == "blend_interleave_gated_base_first":
                config["gate_base_first"] = True
        note = "Word-pair counts from the organizers' training descriptions; ignores the video."
        descriptions = [episode.description for episode in load_episodes(data_dir)]
        (output_dir / "models/style").mkdir(parents=True)
        (output_dir / "models/style/counts.json").write_text(
            json.dumps(build_counts(descriptions)), encoding="utf-8")
        if config.get("chars"):
            shutil.copyfile(ROOT / "team/inference/submission_chars.py", output_dir / "submission_chars.py")
            (output_dir / "models/style/char_counts.json").write_text(
                json.dumps(build_char_counts(descriptions)), encoding="utf-8")
    shutil.copyfile(ROOT / "team" / "inference" / source, output_dir / "submission.py")
    save_json(output_dir / "inference_config.json", config)
    save_json(output_dir / "training_info.json", {"method": method, "uses_video": False, "trained": False, "note": note})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["blend", "interleave", "blend_interleave", "blend_interleave_gated",
                                             "blend_interleave_gated_excess", "blend_interleave_chars",
                                             "blend_interleave_gated_chars", "blend_interleave_gated_own",
                                             "blend_interleave_gated_mean",
                                             "blend_interleave_gated_base_first",
                                             "blend_interleave_gated_mean_chars"], required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.method, args.base_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
