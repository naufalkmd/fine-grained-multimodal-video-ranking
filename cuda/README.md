# Local GPU work

Everything for running on a personal NVIDIA GPU lives here. The team's main
environment is the provided Colab, using `starterkit_colab.ipynb`.

Git tracks only `README.md`, `config.json` and `requirements-eval.txt`. `Data/`,
`models/` and `outputs/` stay local.

## Setup

Run from the repository root, with the training data unzipped to `cuda/Data/Train`.

```bash
conda create -n inthon python=3.12.3 -y
conda activate inthon
python -m pip install -r cuda/requirements-eval.txt
python download_models.py --config cuda/config.json
```

## Train and evaluate

```bash
python train.py --config cuda/config.json --device auto
python evaluate.py --checkpoint cuda/outputs/baseline/best.pt
```

Use a new `--output-dir cuda/outputs/<name>` for each further run.

Checkpoints and feature caches record absolute paths. Moving `Data/` or `models/`
makes existing runs in `outputs/` unusable, so retrain after any move.

## Create a submission

```bash
python export.py --checkpoint cuda/outputs/baseline/best.pt --output-dir cuda/sample_submission
python validate.py --submission-dir cuda/sample_submission --data-dir cuda/Data/Train --device auto
python package.py --submission-dir cuda/sample_submission --output cuda/sample_submission.zip
```

Local timings don't prove the T4 inference limit, so check final submissions on a T4.
