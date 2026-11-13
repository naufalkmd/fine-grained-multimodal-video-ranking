"""Create the `submission_fill.py` submission from a fine-tuned SmolVLM (team/training/train_smolvlm.py).

package.py requires the kit's files, so they are copied from an exported kit submission; they
aren't loaded at inference.

    python -m team.export.export_fill --model-dir cuda/outputs/smolvlm/ft_v1 \
        --kit-submission cuda/sample_submission --output-dir cuda/fill_v1_submission
"""

import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, save_json
from team.export.export_smolvlm_gate import MODEL_FILES


def export(model_dir, kit_submission, output_dir, central_rank_weight, max_frames, image_size, prior_weight):
    model_dir, kit_submission, output_dir = map(Path, (model_dir, kit_submission, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models",
                    ignore=shutil.ignore_patterns("smolvlm"))
    (output_dir / "models/smolvlm").mkdir()
    for name in MODEL_FILES:
        if (model_dir / "model" / name).exists():
            shutil.copyfile(model_dir / "model" / name, output_dir / "models/smolvlm" / name)
    for name in ["submission_central.py", "smolvlm_scorer.py"]:
        shutil.copyfile(ROOT / "team" / "inference" / name, output_dir / name)
    shutil.copyfile(ROOT / "team/inference/submission_fill.py", output_dir / "submission.py")
    save_json(output_dir / "inference_config.json", {"fill": {
        "central_rank_weight": central_rank_weight, "max_frames": max_frames,
        "image_size": image_size, "prior_weight": prior_weight}})
    training = json.loads((model_dir / "training_info.json").read_text(encoding="utf-8"))
    save_json(output_dir / "training_info.json", {
        "method": "SmolVLM image support on the tokens that differ between candidates",
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct", "storage_dtype": "float16",
        "smolvlm_training": training,
        "note": "The kit's DINOv2, MPNet and weights/mlp.pt are present only because package.py "
                "requires them; they aren't loaded."})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="team.training.train_smolvlm output, with model/ and training_info.json.")
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--central-rank-weight", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    export(args.model_dir, args.kit_submission, args.output_dir, args.central_rank_weight,
           args.max_frames, args.image_size, args.prior_weight)


if __name__ == "__main__":
    main()
