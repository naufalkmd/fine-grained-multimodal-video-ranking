"""Add SigLIP to an exported ranker submission, so that it ranks with both models.

    python -m team.export.export_combined --ranker-submission cuda/temporal_word_swap_v1_submission \
        --siglip-submission cuda/siglip_v1_submission --output-dir cuda/combined_v1_submission
"""

import argparse
import shutil
from pathlib import Path

from common import ROOT


def export(ranker_submission, siglip_submission, output_dir):
    ranker_submission, siglip_submission, output_dir = map(Path, (ranker_submission, siglip_submission, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    shutil.copytree(ranker_submission, output_dir, dirs_exist_ok=True)
    shutil.copytree(siglip_submission / "models/siglip", output_dir / "models/siglip")
    # The ranker's own entry point stays available under its source name.
    shutil.copyfile(ROOT / "team/inference/submission_ranker.py", output_dir / "submission_ranker.py")
    shutil.copyfile(ROOT / "team/inference/submission_siglip.py", output_dir / "submission_siglip.py")
    shutil.copyfile(ROOT / "team/inference/submission_combined.py", output_dir / "submission.py")
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranker-submission", type=Path, required=True)
    parser.add_argument("--siglip-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.ranker_submission, args.siglip_submission, args.output_dir)


if __name__ == "__main__":
    main()
