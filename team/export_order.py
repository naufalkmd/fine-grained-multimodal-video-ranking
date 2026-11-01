"""Create the `submission_order.py` submission, storing SigLIP in float16. No training.

package.py requires the kit's files, so they are copied from an exported kit submission; they
aren't loaded at inference.

    python -m team.export_order --siglip-dir cuda/models/siglip \
        --kit-submission cuda/sample_submission --output-dir cuda/probe_order_submission
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModel

from common import ROOT, save_json


def export(siglip_dir, kit_submission, output_dir):
    siglip_dir, kit_submission, output_dir = map(Path, (siglip_dir, kit_submission, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "inference_config.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models",
                    ignore=shutil.ignore_patterns("siglip"))
    shutil.copyfile(ROOT / "team/submission_central.py", output_dir / "submission_central.py")
    shutil.copyfile(ROOT / "team/submission_order.py", output_dir / "submission.py")
    target = output_dir / "models/siglip"
    model = AutoModel.from_pretrained(siglip_dir, local_files_only=True, dtype=torch.float32)
    state = {key: value.detach().half() if value.is_floating_point() else value.detach()
             for key, value in model.state_dict().items()}
    model.save_pretrained(target, state_dict=state, safe_serialization=True)
    for name in ["preprocessor_config.json", "tokenizer.json"]:
        shutil.copyfile(siglip_dir / name, target / name)
    save_json(output_dir / "training_info.json", {
        "method": "event-order fit of each candidate to the frames with SigLIP "
                  "(team/submission_order.py)",
        "model_id": "google/siglip-base-patch16-224", "trained_layers": "none",
        "storage_dtype": "float16",
        "note": "The kit's DINOv2, MPNet and weights/mlp.pt are present only because package.py "
                "requires them; they aren't loaded."})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--siglip-dir", type=Path, required=True)
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.siglip_dir, args.kit_submission, args.output_dir)


if __name__ == "__main__":
    main()
