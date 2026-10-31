# Inthon baseline starter kit

> English translation. The organizers' Korean original is in [docs/ko/README.md](docs/ko/README.md).

An example of training, validation and submission using only the original Train data. It uses
Python 3.11 and the permitted `facebook/dinov2-base` and
`sentence-transformers/all-mpnet-base-v2`.

## Method

1. Encode every image in an episode separately with DINOv2. **Mean-pool** the per-frame CLS
   features to get a 768-dimensional image sequence feature.
2. Encode the original description with MPNet and take the token mean pooling, excluding padding,
   to get a 768-dimensional text feature.
3. **Concat** the two features and apply `Linear(1536, 256) → ReLU → Dropout → Linear(256, 1)`.
   The sigmoid output is a scalar in the 0–1 range for each video–description pair.
4. In every epoch, each training episode uses its one correct description (label 1) and **one
   original description picked at random from another training episode** (label 0). Descriptions
   identical to the episode's own are excluded from negatives. It does not generate images or
   text, and does not write new episode files.
5. Both encoders are frozen and only the MLP is trained. For numerical stability, training applies
   `BCEWithLogitsLoss` to the logits before the sigmoid, and inference returns sigmoid scores.

Images are read in numeric order, with only a fixed resize, center crop and normalization applied.
It does not use random image transforms, frame reduction, description changes or teacher
predictions. Mean pooling is order-invariant, so this baseline has no temporal-order reasoning
module of its own. The sigmoid value is a learned matching score and is not guaranteed to be a
calibrated probability.

## Installation and data

Run the commands below from `starterkit/`. On a T4, use a PyTorch environment that supports CUDA.
On Colab, where PyTorch is already installed, first check compatibility with the existing CUDA
environment.

```bash
python -m pip install -r requirements.txt
python download_models.py
```

If you already have **the same permitted models** locally, you can copy them instead of
downloading.

```bash
python download_models.py --source-root /path/to/model_snapshots
```

This path must contain the config, safetensors and processor/tokenizer files under `dinov2/` and
`mpnet/`. The default settings are in `config.json`, and its paths are resolved relative to that
config file's location. Paths passed on the CLI are relative to the current working directory.

The data format is as follows. `description.json` and annotations of wrong answers are not
required.

```text
data/Train/
  episode_1/
    description.txt
    images/1.jpg, 2.jpg, ..., 10.jpg, ...
  episode_2/
    ...
```

## Training and local evaluation

```bash
python train.py --device auto
python evaluate.py --checkpoint outputs/baseline/best.pt
```

- By default, the full Train set is split 80:20 by episode. The seed is 2026.
- Train and validation IDs are recorded in `outputs/baseline/splits.json`. Splitting is never done
  by frame.
- Negatives are drawn only from within each split. Training negatives are redrawn every epoch, and
  validation negatives are fixed. With N training episodes per epoch, there are exactly 2N training
  pairs.
- Frozen encoder features are saved to `features.pt` and the MLP is trained on them. If the data or
  encoder files change, the existing cache is refused.
- The checkpoint with the lowest validation BCE is saved to `best.pt`. Existing experiments are
  not overwritten, so specify `--output-dir outputs/another_name` when rerunning.
- Without a GPU, you can run with `--device cpu`. Full feature extraction takes time.

To check quickly that it works, use the following commands first.

```bash
python train.py --max-episodes 12 --epochs 3 --output-dir outputs/smoke
python evaluate.py --checkpoint outputs/smoke/best.pt
```

The output metrics are `binary_accuracy` over the two pairs, `pair_accuracy` (the share of cases
where the correct description scores higher than the random negative), and BCE. Ties do not count
as correct pairs. These metrics are **not the leaderboard multiple-choice accuracy**. A description
from another video can be partly correct, and random negatives do not represent hard wrong answers
that only change the order of events. Train and test also come from different sources, so this
validation performance alone does not guarantee generalization.

## Creating a submission

```bash
python export.py --checkpoint outputs/baseline/best.pt
python validate.py --submission-dir sample_submission --device auto
python package.py
```

The export target defaults to `sample_submission/`. If results already exist there, use a new
`--output-dir`. Replacing an existing ZIP with package requires an explicit `--force`.

```text
sample_submission/
  submission.py
  submission.json              # {"api_version": 2}
  baseline_model.py
  inference_config.json
  training_info.json
  weights/mlp.pt
  models/dinov2/                # Encoder weights + image preprocessor
  models/mpnet/                 # Encoder weights + tokenizer
```

`submission.py` must be at the top level of the ZIP. Do not add an extra parent folder named
`sample_submission/` inside the ZIP. Training data, feature caches and training scripts do not go
in the inference ZIP. For reproducibility inspection, keep this starter kit and the training
settings and logic separately.

Frozen encoders are stored as float16 to reduce submission size. CUDA inference runs in float16 and
CPU inference in float32, and converting the storage precision can cause small numerical
differences.

The server supports the class interface below.

```python
class Submission:
    def __init__(self, artifact_dir: str, device: str):
        ...  # Load local models and the MLP only once

    def predict(self, images, choices) -> list[str]:
        ...  # All choice IDs in descending score order
```

`images` is a list of RGB PIL images in time order, and `choices` is a list of
`{"id": str, "text": str}`. Do not hard-code the number of choices or fix their IDs to A–E.
`Submission.score()` computes a 0–1 score per choice, and `predict()` returns the IDs sorted by it.
It does not update parameters, statistics or retrieval memory between evaluation episodes, and does
not depend on the internet or external caches.

In the organizers' repository, the ZIP contract and inference can be checked together using the
actual server code.

```bash
python validate.py --zip sample_submission.zip --server-dir ../server --device cpu
```

This check prepares inputs taken from the original Train set in a temporary directory, and runs the
server's `runner/execute.py` with an empty HF cache and a separate CWD. It does not read Test data.
The current server's runtime and upload limits, and the settings needed for model submissions, are
summarized in [SERVER_COMPATIBILITY.md](SERVER_COMPATIBILITY.md).

## Where to make changes

| File | What participants can change |
| --- | --- |
| `config.json` | Batch size, learning rate, MLP size, epochs, etc. |
| `baseline_model.py` | Image aggregation, text encoding, concat MLP structure |
| `dataset.py` | Data loading and negative sampling |
| `train.py` | Training logic and loss |
| `submission.py` | Per-choice scores and returned ranking |

If you change the model or preprocessing, the same change must be reflected in export and
inference. To train unfrozen encoders, the feature cache approach must change as well.
The organizers' `documents/rules.md` takes precedence on rules.

## Included sample weights

The `sample_submission/` generated in this work contains **weights for functional checking only,
from a 3-epoch run on 12 original Train episodes (10 train, 2 validation)**. They do not represent
performance from full Train training. For real use, train on the full Train set, then export to a
new directory and submit that.

Training results and provenance are in `outputs/smoke/`, and the submission's training information
is in `sample_submission/training_info.json`.
