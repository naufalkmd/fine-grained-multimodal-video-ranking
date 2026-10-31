"""Create a submission folder that ranks with SigLIP alone, storing it in float16.

package.py requires the kit's files, so the folder also holds them, copied from an exported
submission. They aren't loaded at inference.

    python -m team.export_siglip --siglip-dir cuda/models/siglip \
        --kit-submission cuda/sample_submission --output-dir cuda/siglip_v1_submission
"""

import argparse
import shutil
from pathlib import Path

import torch
from common import ROOT, save_json
from transformers import AutoModel


def export(siglip_dir, kit_submission, output_dir):
    siglip_dir, kit_submission, output_dir = map(Path, (siglip_dir, kit_submission, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    for name in ["baseline_model.py", "submission.json", "inference_config.json", "weights/mlp.pt"]:
        shutil.copyfile(kit_submission / name, output_dir / name)
    shutil.copytree(kit_submission / "models", output_dir / "models")
    shutil.copyfile(ROOT / "team/submission_siglip.py", output_dir / "submission.py")
    target = output_dir / "models/siglip"
    model = AutoModel.from_pretrained(siglip_dir, local_files_only=True, dtype=torch.float32)
    state = {key: value.detach().half() if value.is_floating_point() else value.detach()
             for key, value in model.state_dict().items()}
    model.save_pretrained(target, state_dict=state, safe_serialization=True)
    for name in ["preprocessor_config.json", "tokenizer.json"]:
        shutil.copyfile(siglip_dir / name, target / name)
    save_json(output_dir / "training_info.json", {
        "model_id": "google/siglip-base-patch16-224", "trained_layers": "none",
        "encoder_storage_dtype": "float16",
        "note": "Ranks with SigLIP image-text matching only; the kit's models aren't loaded."})
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--siglip-dir", type=Path, required=True)
    parser.add_argument("--kit-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.siglip_dir, args.kit_submission, args.output_dir)


if __name__ == "__main__":
    main()
