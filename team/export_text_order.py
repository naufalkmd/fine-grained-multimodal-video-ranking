"""Create `submission_text_order.py`: `text_blend_interleave` (team/export_text.py) plus the order check.

Builds `text_blend_interleave` exactly as team/export_text.py does, then adds SigLIP in float16 and the
order check's code, and sets `order_merge` ("third" or "arbiter", see team/submission_text_order.py).

    python -m team.export_text_order --merge third --siglip-dir cuda/models/siglip \
        --base-dir cuda/sample_submission --data-dir cuda/Data/Train --output-dir cuda/text_order_third
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel

from common import ROOT, save_json
from team.export_text import export as export_text


def export(merge, siglip_dir, base_dir, data_dir, output_dir):
    siglip_dir, output_dir = Path(siglip_dir), Path(output_dir)
    export_text("blend_interleave", base_dir, data_dir, output_dir)
    for name in ["submission_text_rules.py", "submission_order.py", "submission_central.py"]:
        shutil.copyfile(ROOT / "team" / name, output_dir / name)
    shutil.copyfile(ROOT / "team/submission_text_order.py", output_dir / "submission.py")
    target = output_dir / "models/siglip"
    model = AutoModel.from_pretrained(siglip_dir, local_files_only=True, dtype=torch.float32)
    state = {key: value.detach().half() if value.is_floating_point() else value.detach()
             for key, value in model.state_dict().items()}
    model.save_pretrained(target, state_dict=state, safe_serialization=True)
    for name in ["preprocessor_config.json", "tokenizer.json"]:
        shutil.copyfile(siglip_dir / name, target / name)
    config = json.loads((output_dir / "inference_config.json").read_text(encoding="utf-8"))
    save_json(output_dir / "inference_config.json", {**config, "order_merge": merge})
    save_json(output_dir / "training_info.json", {
        "method": f"text_blend_interleave with the SigLIP order check ({merge})",
        "uses_video": True, "trained": False, "siglip": "google/siglip-base-patch16-224, float16",
        "note": "Word-pair counts from the organizers' training descriptions; no trained weights."})
    print(f"Exported: {output_dir.resolve()} (order_merge {merge})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", choices=["third", "arbiter"], required=True)
    parser.add_argument("--siglip-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.merge, args.siglip_dir, args.base_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
