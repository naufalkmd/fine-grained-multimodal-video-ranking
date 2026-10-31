"""Offline API v2: one frozen SigLIP encoder and three independent E0 heads.

Only the current episode enters score(). No test cache, adaptation, retrieval,
candidate reconstruction, or cross-episode statistics are used.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from sentence_models import FrozenSiglipResidual
from text_tokenization import make_text_plan, sentence_plan


def local_path(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its directory')
    return target


def pack_features(frames, texts, segments, plans):
    """Produce exactly the masks, positions and weights used by training."""
    if (not len(frames) or not len(texts) or len(segments) != len(texts)
            or len(plans) != len(texts)):
        raise ValueError('Empty or mismatched frame/text features')
    c, d = texts.shape
    max_segments = max(len(x) for x in segments)
    sentences = texts.new_zeros((1, c, max_segments, d))
    mask = torch.zeros((1, c, max_segments), dtype=torch.bool, device=texts.device)
    positions = texts.new_zeros((1, c, max_segments))
    weights = texts.new_zeros((1, c, max_segments))
    for i, (vectors, plan) in enumerate(zip(segments, plans)):
        n = len(vectors)
        if n != len(plan['segments']) or not n:
            raise ValueError('Sentence plan coverage differs')
        sentences[0, i, :n] = vectors
        mask[0, i, :n] = True
        positions[0, i, :n] = texts.new_tensor([s['position'] for s in plan['segments']])
        weights[0, i, :n] = texts.new_tensor([s['weight'] for s in plan['segments']])
    return (frames[None], torch.ones((1, len(frames)), dtype=torch.bool, device=frames.device),
            texts[None], sentences, mask, positions, weights)


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        from transformers import AutoImageProcessor, PreTrainedTokenizerFast, SiglipModel

        self.root = Path(artifact_dir).resolve()
        self.config = json.loads((self.root / 'inference_config.json').read_text(encoding='utf-8'))
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        torch.set_num_threads(4)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        model_dir = local_path(self.root, self.config['encoder_path'])
        self.encoder = SiglipModel.from_pretrained(
            str(model_dir), local_files_only=True, dtype=self.dtype,
            attn_implementation='eager', trust_remote_code=False,
        ).to(self.device).eval().requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=False, trust_remote_code=False)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
        if self.tokenizer.eos_token_id != self.tokenizer.pad_token_id:
            raise ValueError('Expected original SigLIP EOS padding')
        self.heads = []
        for item in self.config['heads']:
            path = local_path(self.root, item['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Head file checksum differs')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if (checkpoint['seed'] != item['seed'] or checkpoint['kind'] != 'sentence'
                    or checkpoint['model_config'] != self.config['model_config']):
                raise ValueError('Wrong head identity/configuration')
            head = FrozenSiglipResidual('sentence', **self.config['model_config'])
            head.load_state_dict(checkpoint['state_dict'], strict=True)
            self.heads.append(head.to(self.device).eval().requires_grad_(False))
        if len(self.heads) != 3:
            raise ValueError('This artifact requires exactly three selected heads')

    @staticmethod
    def _unit(values):
        values = values.float()
        norms = values.norm(dim=-1, keepdim=True)
        if not torch.isfinite(values).all() or (norms <= 1e-12).any():
            raise ValueError('Invalid encoder features')
        return values / norms

    def _images(self, images):
        if not images:
            raise ValueError('images must be a nonempty ordered list')
        parts = []
        batch = self.config.get('image_batch_size', 8)
        for start in range(0, len(images), batch):
            # Preserve the server-provided chronology, even when frames repeat.
            rgb = [x.convert('RGB') for x in images[start:start + batch]]
            try:
                pixels = self.processor(images=rgb, return_tensors='pt')['pixel_values']
                features = self.encoder.get_image_features(pixel_values=pixels.to(self.device, self.dtype))
                parts.append(self._unit(features))
            finally:
                for image in rgb:
                    image.close()
        return torch.cat(parts)

    def _texts(self, choices):
        if (not choices or any(not isinstance(x.get('id'), str) or
                not isinstance(x.get('text'), str) or not x['text'].strip() for x in choices)
                or len({x['id'] for x in choices}) != len(choices)):
            raise ValueError('Require unique string IDs and nonempty choice text')
        base_plans = [make_text_plan(self.tokenizer, x['text']) for x in choices]
        plans = [sentence_plan(self.tokenizer, x['text']) for x in choices]
        # Deduplicate only within this episode. Nothing is retained after return.
        rows, row_map = [], {}
        def index(tokens):
            key = tuple(tokens)
            if key not in row_map:
                row_map[key] = len(rows)
                rows.append(tokens)
            return row_map[key]
        bases = [index(p['token_inputs']['truncate64'][0]) for p in base_plans]
        spans = [[index(s['token_ids']) for s in p['segments']] for p in plans]
        batch, values = self.config.get('text_batch_size', 16), []
        for start in range(0, len(rows), batch):
            inputs = torch.tensor(rows[start:start + batch], dtype=torch.long, device=self.device)
            values.append(self._unit(self.encoder.get_text_features(input_ids=inputs)))
        values = torch.cat(values)
        # Training text_features() normalized paragraph features a second time.
        texts = self._unit(values[bases])
        return texts, [values[ids] for ids in spans], plans

    @torch.inference_mode()
    def score_details(self, images, choices):
        frames = self._images(images)
        texts, sentences, plans = self._texts(choices)
        inputs = pack_features(frames, texts, sentences, plans)
        seed_scores = torch.stack([head(*inputs)[0] for head in self.heads])
        scores = seed_scores.mean(0)
        if scores.shape != (len(choices),) or not torch.isfinite(scores).all():
            raise ValueError('Invalid scores')
        return {'scores': scores.cpu().tolist(), 'seed_scores': seed_scores.cpu().tolist()}

    def score(self, images, choices):
        """Raw matching scores, not calibrated probabilities."""
        return self.score_details(images, choices)['scores']

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]['id'] for i in order]
