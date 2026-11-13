"""Build notebooks/final_colab.ipynb from the organizers' notebook, so that their cells stay in one place.

Environment, data and model cells are reused. Only the training, validation and export cells change.

    python -m team.export.make_final_notebook
"""

import json

from common import ROOT

INTRO = """# INTHON 2026 · Final model on Colab

Produces the team's final model on a T4. Run from the top with **Runtime → Change runtime type → T4 GPU**.

Before starting, copy the repository to `MyDrive/starterkit`, without `cuda/`, and the data to
`MyDrive/data/Train.zip`. `SMOKE_TEST` starts as `True` to check that everything works. **Set it to
`False` in section 2 for the final run,** and don't submit a smoke-test ZIP.

Everything saved for inspection goes to `starterkit/colab_runs/<experiment name>/`: code, settings,
logs, checkpoint, the negatives used in every epoch, and the ZIP.

Scores here use our own generated negatives, so they aren't leaderboard estimates."""

TRAIN_MD = """## 7. Training
Caches frame features, then trains for a fixed number of epochs. Don't add early stopping: with flipping,
order pairing stays near chance until about epoch 25.

The checkpoint is chosen by the clean rank score averaged with order pairing, and saved to Drive with the
negatives used. **If the session disconnects,** there's no resume, so start a new experiment from cell 2.
The frame cache stays in `colab_runs/frames` on Drive and is reused. The first full run spends extra time
creating it, including a smoke test."""

TRAIN = """TRAINING_DIR = CHECKPOINT_DIR
FRAME_CACHE = DRIVE_STARTERKIT / "colab_runs/frames/features.pt"
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
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
run_python(r\"\"\"
import json, sys
from pathlib import Path
from common import load_config
from dataset import load_episodes
from team.candidates.negatives import load_action_labels
from team.evaluation.rank_eval import load_candidates
config = load_config(sys.argv[1])
splits = json.loads(Path(sys.argv[2]).read_text())
episodes = {e.episode_id: e for e in load_episodes(config["data_dir"])}
valid = splits["valid"]
candidates = load_candidates(Path(config["output_dir"]).parent / "validation/candidates.json", valid,
                             [episodes[i].description for i in valid],
                             [load_action_labels(episodes[i]) for i in valid], config["seed"])
print("Validation episodes:", len(candidates))
\"\"\", CONFIG_PATH, SPLITS, log_name="candidates.log")
args = ["--config", CONFIG_PATH, "--splits", SPLITS, "--output-dir", TRAINING_DIR, "--frame-cache", FRAME_CACHE,
        "--model", MODEL, "--train-negatives", TRAIN_NEGATIVES, "--reverse-weight", REVERSE_WEIGHT,
        "--flip-probability", FLIP_PROBABILITY, "--save-negatives", "--device", "cuda:0"]
args += ["--epochs", 2, "--patience", 2, "--max-train-episodes", 64] if SMOKE_TEST else [
    "--epochs", EPOCHS, "--patience", EPOCHS]
run_command([PYTHON, "-u", "-m", "team.training.train_ranker", *args], log_name="train.log", extra_env=OFFLINE)
CHECKPOINT = TRAINING_DIR / "best.pt"
if not CHECKPOINT.is_file():
    raise FileNotFoundError("best.pt was not created. Check train.log.")
print("Selected checkpoint:", CHECKPOINT)
"""

EVALUATE_MD = """## 8. Scorecard
Scores from `report.json`. Clean rank score should be well above 0.375, order pairing above 0.5, and the
wrong-video scores on `similar_episode` and `lexical` near 0.375."""

EVALUATE = """report = json.loads((TRAINING_DIR / "report.json").read_text())
clean = (report["scores"]["similar_episode"]["score"] + report["scores"]["lexical"]["score"]) / 2
print(f"Best epoch: {report['best_epoch']}")
print(f"Clean rank score: {clean:.3f}")
print(f"Order pairing: {report['order_pairing']['score']:.3f}")
print("set | score | wrong video")
for name, item in report["scores"].items():
    print(f"{name} | {item['score']:.3f} | {report['wrong_video_scores'][name]['score']:.3f}")
"""

EXPORT = """run_command([PYTHON, "-u", "-m", "team.export.export_ranker", "--checkpoint", CHECKPOINT,
             "--output-dir", EXPORT_DIR], log_name="export.log", extra_env=OFFLINE)
# Run from a separate CWD so that the exported modules are the ones actually used.
validator = LOCAL_ROOT / "validate-export.py"
validation_code = r\"\"\"
import sys
from pathlib import Path
artifact, kit, data = map(Path, sys.argv[1:4])
sys.path.insert(0, str(artifact))
sys.path.insert(1, str(kit))
import baseline_model, ranker_model
assert Path(baseline_model.__file__).parent == artifact
assert Path(ranker_model.__file__).parent == artifact
from validate import check_direct
check_direct(artifact, data, "cuda:0")
\"\"\"
validator.write_text(validation_code, encoding="utf-8")
run_command([PYTHON, "-I", validator, EXPORT_DIR, LOCAL_KIT, DATA_DIR],
            cwd=LOCAL_ROOT, log_name="validate.log",
            extra_env={**OFFLINE, "HF_HOME": str(LOCAL_ROOT / "empty-validation-cache")})
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one match in the organizers' notebook: {old}")
    return text.replace(old, new)


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
    configure = replace_once(configure, "SMOKE_TEST = False\n",
                             "SMOKE_TEST = True  # Set to False for the final run\n")
    configure = replace_once(configure, "EPOCHS = 20\n",
                             'EPOCHS = 60\nMODEL = "temporal"\n'
                             'TRAIN_NEGATIVES = "reorder,swapped_action,similar_episode,random,lexical,word_swap"\n'
                             "REVERSE_WEIGHT = 0.1\nFLIP_PROBABILITY = 0.5\n")
    configure = replace_once(configure, '"download_models.py"]',
                             '"download_models.py", "team/candidates/negatives.py", "team/candidates/lexical.py", "team/candidates/word_swap.py", "team/training/train_negatives.py",\n'
                             '            "team/training/train_ranker.py", "team/inference/ranker_model.py", "team/evaluation/rank_eval.py",\n'
                             '            "team/inference/submission_ranker.py", "team/export/export_ranker.py"]')
    put("configure", configure)
    put("stage-data", replace_once(get("stage-data"), '".git", "models",', '".git", "cuda", "models",'))
    put("0f532306", TRAIN_MD)
    put("train", TRAIN)
    put("3ff2a64a", EVALUATE_MD)
    put("evaluate", EVALUATE)
    put("export-validate", EXPORT)
    put("package-save", replace_once(get("package-save"), '"weights/mlp.pt",', '"weights/mlp.pt", "ranker_model.py",'))
    put("e4f34cd4", "## Changing code and rerunning\n"
                    "Edit the code on Drive, then start a new experiment from cell 2. A copy of the code used "
                    "for each run stays in its `source/` folder. If CUDA runs out of memory, lower the batch "
                    "sizes in cell 2.")
    output = ROOT / "notebooks/final_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
