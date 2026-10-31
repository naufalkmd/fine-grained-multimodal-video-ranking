"""Create the text-only `submission_central.py` submission from an existing exported submission.

package.py requires the encoder files, so they are copied from the base submission unchanged even
though this submission never loads them. Only submission.py and training_info.json differ.

    python -m team.export_central --base-dir cuda/sample_submission --output-dir cuda/central_submission
"""

import argparse
import shutil
from pathlib import Path

from common import ROOT, save_json


def export(base_dir, output_dir):
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    shutil.copytree(base_dir, output_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "submission.py", "training_info.json"))
    shutil.copyfile(ROOT / "team/submission_central.py", output_dir / "submission.py")
    save_json(output_dir / "training_info.json", {
        "method": "candidate centrality (team/submission_central.py)",
        "uses_video": False, "trained": False,
        "note": "Encoder files are present only because package.py requires them.",
    })
    print(f"Exported text-only submission: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="An exported submission holding the encoder files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()
