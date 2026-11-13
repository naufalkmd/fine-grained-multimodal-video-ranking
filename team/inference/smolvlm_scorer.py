"""Score candidate descriptions by how likely SmolVLM is to write them after seeing the frames.

Each candidate's score is the summed log-probability of its tokens given the frames, minus
`prior_weight` times the same with no frames (pointwise mutual information when the weight is 1).
Wording the model likes regardless of the video cancels, and tokens shared by all candidates add
about the same to every score, so the words that differ decide the ranking. The frames are encoded
once per call and their key-value cache is shared by all candidates. No state is kept between calls.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = "Describe what happens in these video frames, in order."


def sample_frames(images, max_frames):
    if len(images) <= max_frames:
        return list(images)
    step = (len(images) - 1) / (max_frames - 1)
    return [images[round(i * step)] for i in range(max_frames)]


class SmolVLMScorer:
    def __init__(self, model_dir, device, max_frames=8, image_size=512, prior_weight=1.0,
                 dtype=None):
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {"longest_edge": image_size}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=self.dtype, local_files_only=True).to(self.device).eval()
        self.tokenizer = self.processor.tokenizer
        self.max_frames = max_frames
        self.prior_weight = prior_weight

    def _prompt(self, frame_count):
        content = [{"type": "image"}] * frame_count + [{"type": "text", "text": PROMPT}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True)

    def _candidate_ids(self, texts):
        end = self.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        return [self.tokenizer(" " + text.strip(), add_special_tokens=False)["input_ids"] + [end]
                for text in texts]

    @torch.inference_mode()
    def _log_likelihoods(self, prefix_inputs, candidate_ids):
        """Summed log-probability of each candidate's tokens after a shared prefix."""
        prefix = self.model(**prefix_inputs, use_cache=True)
        count = len(candidate_ids)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(count)
        width = max(len(ids) for ids in candidate_ids)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.full((count, width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((count, width), dtype=torch.long, device=self.device)
        for row, values in enumerate(candidate_ids):
            ids[row, :len(values)] = torch.tensor(values, device=self.device)
            mask[row, :len(values)] = 1
        prefix_mask = torch.ones((count, prefix_inputs["input_ids"].shape[1]), dtype=torch.long,
                                 device=self.device)
        logits = self.model(input_ids=ids, attention_mask=torch.cat([prefix_mask, mask], dim=1),
                            past_key_values=cache, use_cache=False).logits
        first = prefix.logits[:, -1:].expand(count, 1, -1)
        logits = torch.cat([first, logits[:, :-1]], dim=1).float().log_softmax(dim=-1)
        token_scores = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mask
        return token_scores.sum(dim=1)

    @torch.inference_mode()
    def _text_log_likelihoods(self, prefix_ids, candidate_ids):
        """As _log_likelihoods for a short text-only prefix, in one pass without a cache."""
        rows = [prefix_ids + ids for ids in candidate_ids]
        width = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        targets = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        for row, values in enumerate(rows):
            ids[row, :len(values)] = torch.tensor(values, device=self.device)
            mask[row, :len(values)] = 1
            targets[row, len(prefix_ids):len(values)] = 1
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        logits = logits[:, :-1].float().log_softmax(dim=-1)
        token_scores = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) * targets[:, 1:]
        return token_scores.sum(dim=1)

    def components(self, images, texts):
        """(log p(text | frames), log p(text | no frames)) for each text."""
        frames = [image.convert("RGB") for image in sample_frames(images, self.max_frames)]
        candidate_ids = self._candidate_ids(texts)
        with_frames = self.processor(text=self._prompt(len(frames)), images=[frames],
                                     return_tensors="pt").to(self.device)
        with_frames["pixel_values"] = with_frames["pixel_values"].to(self.dtype)
        conditional = self._log_likelihoods(with_frames, candidate_ids)
        prior = self._text_log_likelihoods(
            self.tokenizer(self._prompt(0))["input_ids"], candidate_ids)
        return conditional.cpu(), prior.cpu()

    def score(self, images, texts):
        conditional, prior = self.components(images, texts)
        return (conditional - self.prior_weight * prior).tolist()
