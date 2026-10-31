# Run verification record

> English translation. The organizers' Korean original is in [docs/ko/VERIFICATION.md](docs/ko/VERIFICATION.md).

Performed on 2026-10-29. Only the original `data/Train` was read, and `data/Test` was not used.
Server code, production settings and original data were not changed.

## Training and evaluation

```bash
/home/aikusrv04/anaconda3/bin/python starterkit/train.py \
  --image-model-dir verification/models/dinov2 \
  --text-model-dir verification/models/mpnet \
  --device cpu --max-episodes 12 --epochs 3 \
  --output-dir starterkit/outputs/smoke

/home/aikusrv04/anaconda3/bin/python starterkit/evaluate.py \
  --checkpoint starterkit/outputs/smoke/best.pt --device cpu
```

- 12 selected with seed 2026 from the 4,170 actual Train episodes: 10 train, 2 validation.
- Training pairs per epoch: 10 correct, 10 random negative.
- 3-epoch run completed. The checkpoint selected by validation BCE is epoch 1.
- Validation results after reloading the checkpoint match the record from training:
  BCE `0.6996325254440308`, binary accuracy `0.5`, pair accuracy `0.5`.
- These numbers come from a small sample for functional checking, and are not full-training or
  leaderboard performance.

## Code checks

```bash
/home/aikusrv04/anaconda3/bin/python -m pytest -q starterkit/tests
server/.venv/bin/ruff check starterkit --output-format concise
/home/aikusrv04/anaconda3/bin/python -m compileall -q starterkit
```

**7 tests passed**, and the Ruff and Python compile checks passed.
The checks cover exact 1:1 labels, exclusion of the episode's own and identical descriptions,
negatives within each split, seed reproducibility and per-epoch redrawing, numeric image sorting,
averaging that includes the last image batch, text padding exclusion, scalar sigmoid output, and
MLP updates from BCE.

## Submission and server runner

```bash
/home/aikusrv04/anaconda3/bin/python starterkit/export.py \
  --checkpoint starterkit/outputs/smoke/best.pt
/home/aikusrv04/anaconda3/bin/python starterkit/validate.py --device cpu
/home/aikusrv04/anaconda3/bin/python starterkit/package.py
/home/aikusrv04/anaconda3/bin/python starterkit/validate.py \
  --zip starterkit/sample_submission.zip --server-dir server --device cpu
```

- Direct inference: passed with one 0–1 score per choice, 5 and 2 choices, arbitrary string IDs, and
  repeated calls.
- Actual `server/contracts/archive.py`: API v2 ZIP check and safe extraction passed.
- Actual `server/runner/execute.py`: returned valid rankings for both Train episodes.
- Runner executed with a new temporary CWD, an empty HF cache, an offline environment and
  `python -I`.
- The ZIP includes the models and tokenizers, and does not include Train/Test, training data,
  feature caches or pycache.
- The exported `submission.py` and `baseline_model.py` are byte-identical to the training code's
  originals.

| Item | Value |
| --- | --- |
| ZIP | `sample_submission.zip` |
| Compressed size | 365,781,282 bytes, about 348.8 MiB |
| Extracted size | 394,711,685 bytes, about 376.4 MiB |
| Files in ZIP | 17 |
| SHA256 | `8a3c7222bde9bd99a01a1928087169229188c2c489f05e70fca19d9b49934646` |

## Distinction from the production environment

Verification ran the actual server runner in a local CPU environment with ML packages installed.
It was not run on the current Pillow-only `cpu-reference-v2` Docker image.
The current server's default upload limit of 256 MiB is also smaller than this ZIP.
The common ML runtime and capacity settings needed before deployment are specified in
[SERVER_COMPATIBILITY.md](SERVER_COMPATIBILITY.md).

No GPU driver was available, so T4 run time and memory, and whether the full training 24 hours and
full Test inference 15 minutes are met, were not verified.
