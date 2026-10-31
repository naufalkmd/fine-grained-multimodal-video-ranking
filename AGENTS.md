# AI Agent Instructions

Read `docs/brief.md` before changing code. If a change might break a rule, stop and ask.

- Write code that generates data. Never write descriptions, negatives or other training data directly.
- Use only the permitted models in `docs/brief.md`, and only the organizers' data plus augmentation derived from it.
- Any generative augmentation must use a local open-source model and be reproducible on one T4 with 16 GB VRAM.
- Experiments may run on personal GPUs, but produce the final model on Colab (`docs/brief.md`).
- Never use a teacher model's answers, probabilities, logits, scores or rankings as training labels or objectives, even if saved or transformed first.
- Keep each episode's prediction independent of other episodes, with no training or updates during inference.
- Inference must work offline, with no external APIs or downloads.
- Target the evaluation package versions in `docs/brief.md`, not `requirements.txt`.
- When adding files the submission needs, update `export.py` and `package.py`, then check the ZIP contains them. `package.py` keeps only its required list and `models/`.
- When changing negatives or validation, follow `docs/validation.md`. Never report a local score without saying which negative types it was trained and validated on.
- Never commit passwords, data, weights or generated ZIPs.
- Ask the user before making a competition submission.
