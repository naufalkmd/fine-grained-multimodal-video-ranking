"""Server API v2 for ranking models: exported as submission.py beside ranker_model.py.

With `central_weight` > 0 in inference_config.json, each candidate's standardized model logit is
added to `central_weight` times its standardized centrality: its mean word-level similarity to the
episode's other candidates. Both use only this episode's input. With 0 the model ranks alone.
"""

import difflib
import json
import re
import statistics
from pathlib import Path

import torch
from baseline_model import FrozenEncoders
from ranker_model import MODELS


def centrality(texts):
    words = [re.findall(r"[a-z0-9']+", text.lower()) for text in texts]
    return [statistics.fmean(difflib.SequenceMatcher(a=words[i], b=words[j], autojunk=False).ratio()
                             for j in range(len(words)) if j != i) for i in range(len(words))]


def standardize(values):
    spread = statistics.pstdev(values)
    mean = statistics.fmean(values)
    return [(value - mean) / spread if spread > 0 else 0.0 for value in values]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(config.get("cpu_threads", 4))
        self.device = torch.device(device)
        self.encoders = FrozenEncoders(
            root / "models/dinov2", root / "models/mpnet", device,
            config["image_batch_size"], config["text_batch_size"], config["max_text_length"],
        )
        self.model = MODELS[config["model"]](**config["model_config"]).to(self.device)
        state = torch.load(root / "weights/mlp.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.central_weight = float(config.get("central_weight", 0.0))

    @torch.inference_mode()
    def _encode_frames(self, images):
        encoders = self.encoders
        features = []
        for start in range(0, len(images), encoders.image_batch_size):
            inputs = encoders.image_processor(
                images=[image.convert("RGB") for image in images[start:start + encoders.image_batch_size]],
                return_tensors="pt",
            )
            pixels = inputs["pixel_values"].to(self.device, dtype=encoders.dtype)
            features.append(encoders.image_model(pixel_values=pixels).last_hidden_state[:, 0].float())
        return torch.cat(features).unsqueeze(0)

    @torch.inference_mode()
    def _logits(self, images, choices):
        """One score per candidate; no state is kept between episodes."""
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
        logits = self.model.logits(frames, frame_mask, tokens, inputs["attention_mask"].bool(),
                                   len(choices)).squeeze(0).float()
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Invalid candidate scores.")
        return logits

    def _ranking_scores(self, images, choices):
        # Logits, not sigmoids: sigmoid values can tie at 1.0 when the model is very confident.
        logits = self._logits(images, choices).cpu().tolist()
        if self.central_weight <= 0 or len(choices) < 2:
            return logits
        central = centrality([choice["text"] for choice in choices])
        return [model + self.central_weight * text
                for model, text in zip(standardize(logits), standardize(central))]

    def score(self, images, choices):
        """One value in [0, 1] per candidate, in the same order as predict()."""
        return torch.tensor(self._ranking_scores(images, choices)).sigmoid().tolist()

    def predict(self, images, choices):
        scores = self._ranking_scores(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
