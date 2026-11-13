"""Create the `submission_arbiter.py` submission from a fine-tuned SmolVLM (team/training/train_smolvlm.py).

Word-pair counts are built here from the training descriptions. package.py requires the kit's files, so they are
copied from an exported kit submission; they aren't loaded at inference.

    python -m team.export.export_arbiter --model-dir cuda/outputs/smolvlm/ft_v1 --data-dir cuda/Data/Train \
        --kit-submission cuda/sample_submission --output-dir cuda/arbiter_v1_submission
"""

import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, save_json
from dataset import load_episodes
from team.export.export_smolvlm_gate import MODEL_FILES
from team.inference.submission_style import build_counts


def export(model_dir, data_dir, kit_submission, output_dir, threshold, max_frames, image_size, prior_weight):
    model_dir, kit_submission, output_dir = map(Path, (model_dir, kit_submission, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models",
                    ignore=shutil.ignore_patterns("smolvlm", "style"))
    (output_dir / "models/smolvlm").mkdir()
    for name in MODEL_FILES:
        if (model_dir / "model" / name).exists():
            shutil.copyfile(model_dir / "model" / name, output_dir / "models/smolvlm" / name)
    (output_dir / "models/style").mkdir()
    descriptions = [episode.description for episode in load_episodes(data_dir)]
    (output_dir / "models/style/counts.json").write_text(json.dumps(build_counts(descriptions)), encoding="utf-8")
    for name in ["submission_central.py", "smolvlm_scorer.py", "submission_fill.py", "submission_style.py",
                 "submission_text_rules.py"]:
        shutil.copyfile(ROOT / "team" / "inference" / name, output_dir / name)
    shutil.copyfile(ROOT / "team/inference/submission_arbiter.py", output_dir / "submission.py")
    save_json(output_dir / "inference_config.json", {
        "rule": "blend", "merge": "interleave",
        "arbiter": {"threshold": threshold, "max_frames": max_frames, "image_size": image_size,
                    "prior_weight": prior_weight}})
    training = json.loads((model_dir / "training_info.json").read_text(encoding="utf-8"))
    save_json(output_dir / "training_info.json", {
        "method": "blend and reversed word-pair familiarity in turn, with SmolVLM deciding between their top picks",
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct", "storage_dtype": "float16",
        "uses_video": True, "smolvlm_training": training,
        "note": "The kit's DINOv2 and weights/mlp.pt are present only because package.py requires them; "
                "they aren't loaded. MPNet is used by the blend rule."})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="team.training.train_smolvlm output, with model/ and training_info.json.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    export(args.model_dir, args.data_dir, args.kit_submission, args.output_dir, args.threshold,
           args.max_frames, args.image_size, args.prior_weight)


if __name__ == "__main__":
    main()
