"""Package an exported text-rule submission with only the files it loads, so that it uploads in seconds.

package.py also requires the kit's encoders, which `text_interleave` never loads. The server contract needs only
submission.py at the top of the ZIP (docs/starterkit/SERVER_COMPATIBILITY.md), so this leaves them out. The rankings don't change.

    python -m team.export.package_small --submission-dir cuda/text_interleave_submission \
        --output cuda/text_interleave_small.zip
"""

import argparse
import zipfile
from pathlib import Path

FILES = ("submission.py", "submission.json", "inference_config.json", "training_info.json",
         "models/style/counts.json")


def package(source, output):
    source, output = Path(source), Path(output)
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete export: {missing}")
    code = (source / "submission.py").read_text(encoding="utf-8")
    if "AutoModel" in code or "import torch" in code:
        raise ValueError("This submission loads a model, so it can't leave the encoders out.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(source / name, name)
    print(f"Created: {output}; {output.stat().st_size / 2**20:.2f} MiB", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.submission_dir, args.output)


if __name__ == "__main__":
    main()
