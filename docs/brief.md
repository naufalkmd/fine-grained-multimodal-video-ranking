# InThon 2026 Datathon Brief

The requirements that affect our work, summarized from the
[English translation of the rules](https://app.notion.com/p/40a6f0bfa84d81eba769e29b52984a35)
in Notion. This is not the full rule text.

- The official Korean rules take precedence if wording is unclear.
- Organizers post corrections in the KakaoTalk group. When one changes a rule,
  update Notion and this file together, and record the change in `docs/log.md`.
- The scoring site login and the organizers' contact details are in Notion. Do
  not add them to this repo.

## Deadlines (KST)

| Date and time | Milestone |
|---|---|
| Nov 1, 12:00 | Final submission. The private leaderboard is released after evaluation. |
| Nov 1, 12:30 | Presentation materials. Missing this disqualifies us from awards, whatever our rank. |

- Earlier organizer slides gave 12:10 for final submission. Use 12:00, from the
  later announcement.
- The top 10 teams on the private leaderboard present. Prepare the materials
  before the private leaderboard is released, using the public score.

## Task and scoring

Given a sequence of frames and five candidate descriptions, rank all five from
most to least likely. Candidates can differ only in fine details, such as a
reversed or added action, so the model has to reason over temporal order.

| Rank of correct answer | 1st | 2nd | 3rd | 4th | 5th |
|---|---:|---:|---:|---:|---:|
| Episode score | 1 | 0.5 | 0.25 | 0.125 | 0 |

Assumption: the leaderboard score is the mean over episodes. The slides give
only the per-episode formula.

| | Train | Test |
|---|---|---|
| Source | PerceptionTest video benchmark | A different benchmark, so expect domain shift |
| Size | 4,170 episodes | 1,910 episodes (public 30%, private 70%) |
| Labels | Correct description only, no negatives | 1 correct + 4 incorrect descriptions |

## Limits

- **Submission:** ZIP of inference code and weights, 1 GB at most.
- **Public score checks:** 30 submissions, raised from 20. We haven't confirmed
  whether that is per day or in total, so plan for a total of 30.
- **Inference:** the full test set must finish within 15 minutes, with no
  internet and no external APIs. That is about 0.47 s per episode, including
  model loading.
- **Reproducibility:** all submissions must reproduce within 24 hours on one
  NVIDIA T4 with 16 GB VRAM.
- **Where to run:** experiments may run on personal GPUs. The final model must be
  produced on Colab.

## Permitted models

Weights must be trained from scratch or based on one of these. Fine-tuning and
architecture changes are allowed. Any other pretrained weights make the
submission an error.

| Component | Model |
|---|---|
| Image encoder | `facebook/dinov2-base` |
| Text encoder | `sentence-transformers/all-mpnet-base-v2` |
| Image-text encoder | `google/siglip-base-patch16-224` |
| Vision-language model | `SmolVLM-256M-Instruct` |

## Evaluation environment

These versions override the kit's `requirements.txt` (Python 3.11, torch 2.5.1)
and the versions recorded in `VERIFICATION.md`. Ask the organizers if we need
another package.

| Package | Version |
|---|---|
| Python | `3.12.3` |
| torch | `2.8.0+cu128` |
| torchvision | `0.23.0+cu128` |
| transformers | `4.57.6` |
| numpy | `2.5.4` |
| Pillow | `12.3.0` |
| safetensors | `0.8.0` |
| sentence-transformers | `5.4.0` |
| einops | `0.8.2` |
| timm | `1.0.30` |
| scipy | `1.18.1` |
| scikit-learn | `1.9.1` |
| opencv-python-headless | `4.14.0.94` |
| peft | `0.18.1` |
| accelerate | `1.15.0` |
| bitsandbytes | `0.50.2` |

`sentencepiece` isn't in this list, and SigLIP's usual tokenizer needs it. Load
SigLIP's `tokenizer.json` with `PreTrainedTokenizerFast` instead. Its text input is
limited to 64 tokens, which most full descriptions exceed, so encode sentences.

## Inference rules

- Each prediction may use only that episode's input and artifacts finalized
  before submission.
- During evaluation, do not train parameters, and do not update the model,
  calibration values or retrieval memory from other evaluation episodes.

## Data and augmentation rules

- Train only on the organizers' data and augmentation derived from it, using
  the methods below.
- Rule-based augmentation is allowed.
- Open-source generative models are allowed (the rules give
  `Qwen3-VL-4B-Instruct` as an example) if the augmentation can be reproduced
  on one T4 with 16 GB VRAM. Do not run them on personal hardware with more
  VRAM than that.
- External AI services such as Claude or Codex must not generate or augment
  data. Coding agents may write the augmentation code.
- Plagiarizing other people's code is prohibited.
- Hand-crafted descriptions are not permitted, including manually writing or
  curating them as final training artifacts. All final artifacts must be
  reproducible from code.
- Local generative output does not have to be identical on every run, but the
  process must still be reproducible through code.

Team decisions, stricter than the rules above:

- Don't use word lists written by Claude, Codex or other AI services.
- Don't use outside dictionaries such as WordNet.
- Don't edit or hand-pick descriptions.

## Distillation

Prohibited, whether or not the teacher model is in the submission. Do not use a
teacher's predicted answers, probabilities, logits, scores or rankings as
training labels or objectives, or train a model to imitate them. This also
applies if those outputs are saved to files or transformed into text, code or
new samples first.

## What we must keep for inspection

- All training logic for the final model, including augmentation and model
  training. Award candidates must submit it.
- The generated dataset actually used, and the configuration and process used
  to make it. Organizers will inspect the augmentation code and data.
- An award can be revoked if something looks suspicious, such as a large gap
  between leaderboard performance and reproduced results.

## Open questions for the organizers

- Does a large word list in code, such as button → unbutton, count as rule-based
  augmentation or as hand-crafted descriptions? Our interpretation is that it
  could be either, so do not rely on one until this is answered.
