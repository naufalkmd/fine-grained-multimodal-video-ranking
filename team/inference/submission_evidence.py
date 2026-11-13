"""Server API v2: rank candidates by what this episode's video adds to the ranker's score.

Each candidate scores its logit on this video minus its mean logit on a fixed set of training videos,
saved in the submission before evaluation. A preference that doesn't depend on the video cancels, so
only evidence from this video remains. No state is kept between episodes.
"""

from pathlib import Path

import torch
from submission_ranker import Submission as RankerSubmission


def evidence_scores(model, frames, frame_mask, reference_frames, reference_mask, tokens, token_mask, candidates):
    """Logits on `frames` (one episode) minus the mean logits over the reference videos."""
    own = model.logits(frames, frame_mask, tokens, token_mask, candidates).squeeze(0)
    count = len(reference_frames)
    baseline = model.logits(reference_frames, reference_mask, tokens.repeat(count, 1, 1),
                            token_mask.repeat(count, 1), candidates).mean(dim=0)
    return (own - baseline).float()


class Submission(RankerSubmission):
    def __init__(self, artifact_dir: str, device: str):
        super().__init__(artifact_dir, device)
        saved = torch.load(Path(artifact_dir) / "weights/reference_frames.pt", map_location="cpu", weights_only=True)
        self.reference_frames = saved["frames"].to(self.device).float()
        self.reference_mask = saved["mask"].to(self.device)

    @torch.inference_mode()
    def _logits(self, images, choices):
        if not images:
            raise ValueError("An episode must contain at least one image.")
        if not choices:
            raise ValueError("choices must not be empty.")
        frames = self._encode_frames(images)
        frame_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=self.device)
        inputs = self.encoders.tokenizer(
            [choice["text"] for choice in choices], padding=True, truncation=True,
            max_length=self.encoders.max_text_length, return_tensors="pt",
        ).to(self.device)
        tokens = self.encoders.text_model(**inputs).last_hidden_state.float()
        scores = evidence_scores(self.model, frames, frame_mask, self.reference_frames, self.reference_mask,
                                 tokens, inputs["attention_mask"].bool(), len(choices))
        if scores.ndim != 1 or not torch.isfinite(scores).all():
            raise ValueError("Invalid candidate scores.")
        return scores
