"""Create the `submission_unfamiliar.py` submission from a fine-tuned SmolVLM (team/train_smolvlm.py).

package.py requires the kit's files, so they are copied from an exported kit submission; they
aren't loaded at inference.

    python -m team.export_unfamiliar --model-dir cuda/outputs/smolvlm/ft_v1 \
        --kit-submission cuda/sample_submission --output-dir cuda/unfamiliar_v1_submission
"""

import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, save_json
from team.export_smolvlm_gate import MODEL_FILES


def export(model_dir, kit_submission, output_dir, merge):
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
    for name in ["submission_central.py", "smolvlm_scorer.py", "submission_fill.py"]:
        shutil.copyfile(ROOT / "team" / name, output_dir / name)
    shutil.copyfile(ROOT / "team/submission_unfamiliar.py", output_dir / "submission.py")
    save_json(output_dir / "inference_config.json", {"unfamiliar": {"merge": merge}})
    training = json.loads((model_dir / "training_info.json").read_text(encoding="utf-8"))
    save_json(output_dir / "training_info.json", {
        "method": "least expected wording under SmolVLM without frames, on the tokens that differ",
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct", "storage_dtype": "float16",
        "uses_video": False, "smolvlm_training": training,
        "note": "The kit's DINOv2, MPNet and weights/mlp.pt are present only because package.py "
                "requires them; they aren't loaded."})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="team.train_smolvlm output, with model/ and training_info.json.")
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge", choices=["none", "interleave"], default="none")
    args = parser.parse_args()
    export(args.model_dir, args.kit_submission, args.output_dir, args.merge)


if __name__ == "__main__":
    main()
