import re
import io
import difflib
import statistics
from pathlib import Path

# EXP6 Hybrid v1
# 정확히 허용된 SigLIP 체크포인트 사용
# Centrality는 팀원의 기존 계산과 동일하게 유지

MAX_CENTRALITY_GAP = 0.012
MIN_VISUAL_MARGIN = 1.0
MIN_SUPPORT_FRAMES = 2
N_FRAMES = 8

BLOCKED_WORDS = {
    "left", "right", "up", "down",
    "forward", "backward", "toward", "towards",
    "away", "before", "after", "first", "last",
    "next", "then", "earlier", "later",
    "never", "not", "without", "no",
}


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def centrality_scores(choices):
    texts = [words(c["text"]) for c in choices]
    n = len(texts)

    if n == 1:
        return [1.0]

    return [
        statistics.fmean(
            difflib.SequenceMatcher(
                a=texts[i],
                b=texts[j],
                autojunk=False
            ).ratio()
            for j in range(n)
            if j != i
        )
        for i in range(n)
    ]


def changed_sentence_pair(text_a, text_b):
    a = re.split(r"(?<=[.!?])\s+", text_a.strip())
    b = re.split(r"(?<=[.!?])\s+", text_b.strip())

    if len(a) != len(b):
        return None

    changed = [(x, y) for x, y in zip(a, b) if x != y]

    # 한 문장만 다른 경우에 한정
    if len(changed) != 1:
        return None

    x, y = changed[0]
    wx, wy = words(x), words(y)

    matcher = difflib.SequenceMatcher(
        a=wx,
        b=wy,
        autojunk=False
    )

    changed_count = 0
    changed_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        changed_count += max(i2 - i1, j2 - j1)
        changed_words += wx[i1:i2] + wy[j1:j2]

    if not 1 <= changed_count <= 5:
        return None

    # 시간 순서 / 방향 / 부정은
    # 현재 SigLIP 보정 대상으로 쓰지 않음
    if set(changed_words) & BLOCKED_WORDS:
        return None

    if matcher.ratio() < 0.80:
        return None

    if max(len(wx), len(wy)) > 40:
        return None

    return x, y


def to_pil(frame):
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(frame, dict):
        for key in ("image", "frame", "path"):
            if key in frame:
                frame = frame[key]
                break

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, (str, Path)):
        with Image.open(frame) as im:
            return im.convert("RGB").copy()

    if isinstance(frame, bytes):
        with Image.open(io.BytesIO(frame)) as im:
            return im.convert("RGB").copy()

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    arr = np.asarray(frame)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        if arr.shape[-1] not in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr).astype("float32")
        if arr.min() >= 0 and arr.max() <= 1:
            arr *= 255
        arr = np.clip(arr, 0, 255).astype("uint8")

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return Image.fromarray(arr).convert("RGB")


def sample_frames(images):
    import numpy as np
    import torch

    if images is None:
        return []

    if isinstance(images, dict):
        images = images.get("frames", images.get("images", images))

    if isinstance(images, (list, tuple)):
        items = list(images)
    elif isinstance(images, (np.ndarray, torch.Tensor)):
        items = (
            list(images)
            if len(images.shape) == 4
            else [images]
        )
    else:
        items = [images]

    if not items:
        return []

    n = min(N_FRAMES, len(items))
    indices = [
        round(i * (len(items) - 1) / max(n - 1, 1))
        for i in range(n)
    ]

    return [to_pil(items[i]) for i in indices]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        import torch

        requested = str(device)

        self.device = (
            requested
            if requested.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )

        script_dir = Path(__file__).resolve().parent

        locations = [
            Path(artifact_dir) / "siglip_model",
            script_dir / "siglip_model"
        ]

        self.model_dir = next(
            (
                p for p in locations
                if (p / "config.json").exists()
            ),
            locations[0]
        )

        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.last_debug = {"mode": "not_run"}

    def _load_siglip(self):
        if self.model is not None:
            return

        import torch
        from transformers import (
            SiglipModel,
            SiglipImageProcessor,
            PreTrainedTokenizerFast
        )

        model_dir = str(self.model_dir)

        assert (self.model_dir / "config.json").exists()
        assert (self.model_dir / "tokenizer.json").exists()

        self.image_processor = (
            SiglipImageProcessor.from_pretrained(
                model_dir,
                local_files_only=True
            )
        )

        # SentencePiece를 import하지 않음
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(
                self.model_dir / "tokenizer.json"
            ),
            eos_token="</s>",
            pad_token="</s>",
            unk_token="<unk>",
            model_max_length=64
        )

        assert self.tokenizer.eos_token_id == 1
        assert self.tokenizer.pad_token_id == 1

        dtype = (
            torch.float16
            if self.device.startswith("cuda")
            else torch.float32
        )

        self.model = SiglipModel.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=dtype
        ).to(self.device).eval()

    def _tokenize(self, texts):
        import torch

        # SigLIP 원본과 동일한 64토큰 고정 길이.
        # 끝에 EOS를 추가하고 나머지도 EOS로 패딩.
        encoded = self.tokenizer(
            [t.lower() for t in texts],
            add_special_tokens=False,
            truncation=True,
            max_length=63
        )["input_ids"]

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        ids, masks = [], []

        for seq in encoded:
            seq = seq[:63] + [eos]

            ids.append(
                seq + [pad] * (64 - len(seq))
            )
            masks.append(
                [1] * len(seq) + [0] * (64 - len(seq))
            )

        return {
            "input_ids": torch.tensor(
                ids, dtype=torch.long,
                device=self.device
            ),
            "attention_mask": torch.tensor(
                masks, dtype=torch.long,
                device=self.device
            )
        }

    def _visual_evidence(self, images, pair):
        import torch

        frames = sample_frames(images)

        if not frames:
            return 0.0, 0

        self._load_siglip()

        image_inputs = self.image_processor(
            images=frames,
            return_tensors="pt"
        )

        text_inputs = self._tokenize(pair)

        dtype = next(self.model.parameters()).dtype

        with torch.inference_mode():
            output = self.model(
                pixel_values=image_inputs[
                    "pixel_values"
                ].to(self.device, dtype=dtype),
                **text_inputs
            )

        # [후보 2개, 프레임 수]
        logits = output.logits_per_text.float()

        k = min(2, logits.shape[1])

        top_scores = logits.topk(
            k=k, dim=1
        ).values.mean(dim=1)

        # 두 번째 후보의 점수 - 첫 번째 후보의 점수
        margin = (
            top_scores[1] - top_scores[0]
        ).item()

        support = int(
            ((logits[1] - logits[0]) > 0)
            .sum().item()
        )

        return float(margin), support

    def baseline_score(self, choices):
        return centrality_scores(choices)

    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty")

        base = centrality_scores(choices)

        self.last_debug = {
            "mode": "centrality",
            "flipped": False
        }

        if len(choices) != 5:
            return base

        order = sorted(
            range(len(choices)),
            key=lambda i: (-base[i], i)
        )

        first, second = order[:2]
        gap = base[first] - base[second]

        if gap > MAX_CENTRALITY_GAP:
            return base

        pair = changed_sentence_pair(
            choices[first]["text"],
            choices[second]["text"]
        )

        if pair is None:
            return base

        if images is None:
            return base

        try:
            margin, support = self._visual_evidence(
                images, pair
            )

        except Exception as exc:
            self.last_debug = {
                "mode": "fallback",
                "flipped": False,
                "error": repr(exc)
            }
            return base

        flip = (
            margin > MIN_VISUAL_MARGIN
            and support >= MIN_SUPPORT_FRAMES
        )

        self.last_debug = {
            "mode": "hybrid",
            "centrality_gap": float(gap),
            "visual_margin": margin,
            "support_frames": support,
            "flipped": flip
        }

        if not flip:
            return base

        result = list(base)
        result[second] = base[first] + 1e-6

        return result

    def predict(self, images, choices):
        scores = self.score(images, choices)

        order = sorted(
            range(len(choices)),
            key=lambda i: (-scores[i], i)
        )

        return [choices[i]["id"] for i in order]
