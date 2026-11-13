# Server compatibility

> English translation. The organizers' Korean original is in [docs/starterkit/ko/SERVER_COMPATIBILITY.md](ko/SERVER_COMPATIBILITY.md).

## Confirmed submission contract

- `server/contracts/archive.py`: `submission.py` is required at the top level of the ZIP.
- `submission.json` is optional; if included, use only `{"api_version": 2}`.
- Default extraction limit of 1 GiB, and at most 10,000 files.
- `server/runner/submission.py`: initializes `Submission(artifact_dir, device)` once, then calls
  `predict(images, choices)` for each problem.
- `server/runner/execute.py`: passes RGB images and choices without answer information, and checks
  for a `list[str]` containing every choice ID exactly once.

This starter kit exports according to the contract above. The model, tokenizer, preprocessor and
MLP are all read from inside the ZIP. `validate.py --zip ... --server-dir ../server` can confirm the
server's safe ZIP check and extraction, and an actual runner execution.

## Settings needed in the production environment

The current repository's `cpu-reference-v2` is a **CPU reference runtime with only Pillow
installed**. `server/runtime/requirements.lock.txt` does not include PyTorch or Transformers, so
this neural network submission cannot run as is on the existing immutable image.
Even if requirements are placed in the submission ZIP, the server does not install participants'
packages.

A common scoring image that uses models must provide the PyTorch / Transformers / NumPy / Pillow /
safetensors specified in the starter kit's `requirements.txt`.
For the GPU image, the organizers must build and verify a CUDA/PyTorch combination that supports
NVIDIA T4, and keep the runtime ID and immutable image digest consistent across the server, image
and worker. This work did not change the existing server runtime settings or production images.

Even with the encoders stored as float16, the full submission ZIP is larger than the current server
default upload limit of 256 MiB. The production environment and reverse proxy must allow enough
capacity. For example, the server settings could be set as follows.

```text
MAX_UPLOAD_BYTES=1073741824
MAX_EXTRACTED_BYTES=1073741824
EXECUTION_TIMEOUT_SECONDS=900
```

The extraction limit is the same as the existing 1 GiB. The 900-second execution limit is the 15
minutes in the competition rules. The current server default execution limit is 1800 seconds, so
production settings must be aligned with the rules.
The reverse proxy's request size limit and upload time limit must also fit the actual submission
size.

## Scope of verification

- Training: run on a small subset of the actual Train set.
- Submission code: run with the actual server archive contract and CPU runner.
- Execution environment: local Python 3.11.7, PyTorch 2.5.1+cu121, Transformers 4.57.3,
  NumPy 1.26.4, Pillow 10.2.0. The Pillow version in the install specification and on the server is
  12.3.0.
- No GPU driver was available, so T4 speed and memory, the full Test 15 minutes and the full
  training 24 hours were not measured. Running on the existing Pillow-only Docker image is also not
  marked as passed.

Actual run results are recorded in [VERIFICATION.md](VERIFICATION.md).
