"""Server API v2 for ranking models: exported as submission.py beside ranker_model.py."""

import json
from pathlib import Path

import torch
from baseline_model import FrozenEncoders
from ranker_model import MODELS


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

    def score(self, images, choices):
        return self._logits(images, choices).sigmoid().cpu().tolist()

    def predict(self, images, choices):
        # Sort by logits: sigmoid values can tie at 1.0 when the model is very confident.
        logits = self._logits(images, choices).cpu().tolist()
        order = sorted(range(len(choices)), key=lambda index: -logits[index])
        return [choices[index]["id"] for index in order]
