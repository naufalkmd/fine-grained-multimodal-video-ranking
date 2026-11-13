"""Build notebooks/final_text_colab.ipynb from the organizers' notebook, for a final built on the text rules.

Environment, data and model cells are reused. The training, validation and export cells change. The text rules
have no trained weights; `arbiter` adds a SmolVLM fine-tune.

    python -m team.export.make_text_notebook
"""

import json

from common import ROOT
from team.export.make_final_notebook import replace_once

# models/style/counts.json of the team's text_interleave.zip. A match means this run reproduces its rankings.
EXPECTED_COUNTS = "b6264b015a5cbadf7c06f7ca05b400f3c7fb562ee557a838ce8de69f9cee21f9"

INTRO = """# INTHON 2026 · Final text-rule submission on Colab

Produces the team's final ZIP on a T4. Run from the top with **Runtime → Change runtime type → T4 GPU**.
The text rules take about 20 minutes, most of it installing packages and copying data, because they have no
trained weights. `arbiter` also fine-tunes SmolVLM, which is estimated at 2 to 4 hours, so **start it early and
keep the session open**: the fine-tuned model is saved to Drive only when training ends.

Before starting, copy the repository to `MyDrive/starterkit`, without `cuda/`, and the data to
`MyDrive/data/Train.zip`. **Set `METHOD` in section 2** to the rule the team chose.

Everything saved for inspection goes to `starterkit/colab_runs/<experiment name>/`: code, settings, logs,
the word-pair counts and the ZIP."""

KIT_MD = """## 7. Kit files, and the SmolVLM fine-tune for `arbiter`
`package.py` requires the kit's `weights/mlp.pt`, so the kit's MLP is trained on 12 episodes. No rule loads it,
so this doesn't affect any ranking. For `arbiter`, SmolVLM is then fine-tuned on the correct descriptions of the
training split only, with no generated negatives and no teacher model."""

KIT = """OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
run_script("train.py", "--config", CONFIG_PATH, "--device", "cuda:0", "--max-episodes", "12",
           log_name="train.log", extra_env=OFFLINE)
CHECKPOINT = CHECKPOINT_DIR / "best.pt"
if not CHECKPOINT.is_file():
    raise FileNotFoundError("best.pt was not created. Check train.log.")
print("Kit checkpoint:", CHECKPOINT)
if METHOD == "arbiter":
    SMOLVLM_DIR = LOCAL_KIT / "models/smolvlm"
    run_python(r\"\"\"
import sys
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceTB/SmolVLM-256M-Instruct", local_dir=sys.argv[1], ignore_patterns=["onnx/*"])
\"\"\", SMOLVLM_DIR, log_name="models.log")
    SPLITS = DRIVE_RUN / "splits.json"
    run_python(r\"\"\"
import json, sys
from pathlib import Path
from dataset import load_episodes, split_indices
cfg = json.load(open(sys.argv[1]))
episodes = load_episodes(cfg["data_dir"])
train, valid = split_indices(episodes, cfg["validation_fraction"], cfg["seed"])
Path(sys.argv[2]).write_text(json.dumps({"train": [episodes[i].episode_id for i in train],
                                         "valid": [episodes[i].episode_id for i in valid]}, indent=2))
\"\"\", CONFIG_PATH, SPLITS, log_name="splits.log")
    FINE_TUNE_DIR = DRIVE_RUN / "smolvlm"
    run_command([PYTHON, "-u", "-m", "team.training.train_smolvlm", "--config", CONFIG_PATH, "--splits", SPLITS,
                 "--model-dir", SMOLVLM_DIR, "--output-dir", FINE_TUNE_DIR],
                log_name="smolvlm.log", extra_env=OFFLINE)
    print("Fine-tuned SmolVLM:", FINE_TUNE_DIR)
"""

BUILD_MD = """## 8. Build the submission
Exports the kit's files, then adds the rule. For the rules that use them, the word-pair counts are built here
from all 4,170 training descriptions, and their hash is checked against the team's existing ZIPs."""

BUILD = f"""EXPECTED_COUNTS = "{EXPECTED_COUNTS}"
BASE_DIR = LOCAL_ROOT / "kit-submission"
run_script("export.py", "--checkpoint", CHECKPOINT, "--output-dir", BASE_DIR,
           log_name="export.log", extra_env=OFFLINE)
if METHOD == "arbiter":
    run_command([PYTHON, "-u", "-m", "team.export.export_arbiter", "--model-dir", FINE_TUNE_DIR, "--data-dir", DATA_DIR,
                 "--kit-submission", BASE_DIR, "--output-dir", EXPORT_DIR], log_name="export.log", extra_env=OFFLINE)
else:
    run_command([PYTHON, "-u", "-m", "team.export.export_text", "--method", METHOD, "--base-dir", BASE_DIR,
                 "--data-dir", DATA_DIR, "--output-dir", EXPORT_DIR], log_name="export.log", extra_env=OFFLINE)
if METHOD != "blend":
    counts = EXPORT_DIR / "models/style/counts.json"
    digest = hashlib.sha256(counts.read_bytes()).hexdigest()
    shutil.copyfile(counts, DRIVE_RUN / "counts.json")
    print("Word-pair counts:", digest)
    if digest != EXPECTED_COUNTS:
        raise ValueError("The counts differ from the team's ZIPs. Check the data before using this ZIP.")
    print("The counts match the team's ZIPs, so the rankings are the same.")
"""

CHECK_MD = """## 9. Check offline inference
Using Train inputs, checks that every choice ID is returned exactly once and that repeated inference is
consistent. This is an interface check, not a score."""

CHECK = """# Run from a separate CWD so that the exported files are the ones actually used.
validator = LOCAL_ROOT / "validate-export.py"
validation_code = r\"\"\"
import sys
from pathlib import Path
artifact, kit, data = map(Path, sys.argv[1:4])
sys.path.insert(0, str(artifact))
sys.path.insert(1, str(kit))
from validate import check_direct
check_direct(artifact, data, "cuda:0")
\"\"\"
validator.write_text(validation_code, encoding="utf-8")
run_command([PYTHON, "-I", validator, EXPORT_DIR, LOCAL_KIT, DATA_DIR],
            cwd=LOCAL_ROOT, log_name="validate.log",
            extra_env={**OFFLINE, "HF_HOME": str(LOCAL_ROOT / "empty-validation-cache")})
"""


def main():
    notebook = json.loads((ROOT / "notebooks/starterkit_colab.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}

    def get(cell_id):
        return "".join(cells[cell_id]["source"])

    def put(cell_id, text):
        cells[cell_id]["source"] = text.splitlines(keepends=True)
        if cells[cell_id]["cell_type"] == "code":
            cells[cell_id]["outputs"] = []
            cells[cell_id]["execution_count"] = None

    put("17e8f92d", INTRO)
    configure = get("configure")
    configure = replace_once(configure, 'TRAIN_SOURCE = Path("/content/drive/MyDrive/data/Train")  # Or data/Train.zip\n',
                             'TRAIN_SOURCE = Path("/content/drive/MyDrive/data/Train.zip")  # Or a Train folder\n')
    configure = replace_once(configure, "EPOCHS = 20\n",
                             'METHOD = "blend_interleave"  # "blend_interleave", "interleave", "blend" or "arbiter"\n'
                             "EPOCHS = 3\n")
    configure = replace_once(configure, '"download_models.py"]',
                             '"download_models.py", "team/export/export_text.py", "team/inference/submission_style.py",\n'
                             '            "team/inference/submission_text_rules.py", "team/training/train_smolvlm.py",\n'
                             '            "team/inference/smolvlm_scorer.py", "team/export/export_arbiter.py", "team/inference/submission_arbiter.py",\n'
                             '            "team/inference/submission_fill.py", "team/inference/submission_central.py",\n'
                             '            "team/export/export_smolvlm_gate.py"]')
    put("configure", configure)
    put("stage-data", replace_once(get("stage-data"), '".git", "models",', '".git", "cuda", "models",'))
    put("0f532306", KIT_MD)
    put("train", KIT)
    put("3ff2a64a", BUILD_MD)
    put("evaluate", BUILD)
    put("946bb581", CHECK_MD)
    put("export-validate", CHECK)
    put("e4f34cd4", "## Changing code and rerunning\n"
                    "Edit the code on Drive, then start a new experiment from cell 2. A copy of the code used "
                    "for each run stays in its `source/` folder.")
    output = ROOT / "notebooks/final_text_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
