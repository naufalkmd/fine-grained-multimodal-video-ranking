"""Exact pure text planners from the verified training source; no tail dropped in sentence path."""
import math
import hashlib
import re
MAX_TOKENS = 64
VERSION = "original-sentence-spans-complete-token-windows-v1"
BOUNDARY_PATTERN = r"(?<=[.!?])\s+(?=\S)"

def token_windows(content_ids, special_token_count, max_length=MAX_TOKENS):
    """Non-overlapping token chunks reserving the actual tokenizer special-token budget."""
    budget = max_length - special_token_count
    if budget < 1:
        raise ValueError("Tokenizer special tokens exhaust the text context")
    return [list(content_ids[start:start + budget])
            for start in range(0, len(content_ids), budget)] or [[]]

def pad_special_tokens(tokenizer, content_ids, max_length=MAX_TOKENS):
    ids = add_siglip_specials(tokenizer, content_ids)
    if len(ids) > max_length:
        raise ValueError("Chunk plus tokenizer special tokens exceeds the model context")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    return ids + [int(tokenizer.pad_token_id)] * (max_length - len(ids))

def add_siglip_specials(tokenizer, content_ids):
    """The generic fast wrapper lacks SiglipTokenizer's Python X </s> override."""
    if getattr(tokenizer, "is_fast", False):
        if tokenizer.num_special_tokens_to_add(pair=False) != 1 or tokenizer.eos_token_id is None:
            raise ValueError("Unexpected SigLIP backend special-token template")
        ids = list(content_ids)
        return ids if ids and ids[-1] == tokenizer.eos_token_id else ids + [tokenizer.eos_token_id]
    return list(tokenizer.build_inputs_with_special_tokens(list(content_ids)))

def make_text_plan(tokenizer, text, max_length=MAX_TOKENS):
    """Pure tokenization plan also used for truncation/collision audits."""
    raw = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    full_length = len(add_siglip_specials(tokenizer, raw))
    truncated = tokenizer(text, padding="max_length", max_length=max_length,
                          truncation=True, return_attention_mask=False)["input_ids"]
    if len(truncated) != max_length:
        raise ValueError("Tokenizer did not produce the fixed 64-token input")
    windows = token_windows(raw, specials, max_length)
    chunks = [pad_special_tokens(tokenizer, ids, max_length) for ids in windows]
    return {"content_token_count": len(raw), "full_token_count": full_length,
            "truncated": full_length > max_length, "chunk_count": len(chunks),
            "chunk_content_weights": [max(1, len(ids)) for ids in windows],
            "token_inputs": {"truncate64": [list(truncated)], "token_chunks_weighted_mean": chunks}}

def sentence_spans(text):
    """Return original nonempty substrings, discarding boundary whitespace only."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Original description must be nonempty text")
    starts, ends = [0], []
    for match in re.finditer(BOUNDARY_PATTERN, text):
        ends.append(match.start())
        starts.append(match.end())
    ends.append(len(text))
    spans = []
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({"start": start, "end": end, "text": text[start:end]})
    # A change to the boundary logic must not silently omit non-whitespace data.
    if "".join("".join(x["text"].split()) for x in spans) != "".join(text.split()):
        raise ValueError("Sentence splitting lost original content")
    return spans

def sentence_plan(tokenizer, text):
    """Plan all ordered model inputs; never truncate a sentence's tail.

    Token counts of separately encoded sentences need not equal the token count
    of the whole paragraph, because tokenizer boundaries may differ. Coverage
    is checked against each source sentence's own complete token sequence.
    """
    original = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    segments, sentences = [], []
    for sentence_index, span in enumerate(sentence_spans(text)):
        ids = list(tokenizer.encode(span["text"], add_special_tokens=False, truncation=False))
        windows = token_windows(ids, special_count, MAX_TOKENS)
        if [token for window in windows for token in window] != ids:
            raise ValueError("Token windows lost or duplicated sentence tokens")
        sentences.append({**span, "content_token_count": len(ids),
                          "content_token_ids": ids, "window_count": len(windows)})
        for window_index, window in enumerate(windows):
            padded = pad_special_tokens(tokenizer, window, MAX_TOKENS)
            segments.append({"text": span["text"], "start": span["start"], "end": span["end"],
                             "sentence_index": sentence_index, "window_index": window_index,
                             "token_ids": padded, "content_token_ids": window,
                             "weight": max(1, len(window))})
    denominator = max(len(segments) - 1, 1)
    for index, segment in enumerate(segments):
        segment["position"] = index / denominator
    full_count = len(add_siglip_specials(tokenizer, original))
    return {"version": VERSION, "original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "original_content_token_count": len(original),
            "original_full_token_count": full_count,
            "original_truncated": full_count > MAX_TOKENS,
            "sentence_count": len(sentences), "segment_count": len(segments),
            "sentence_content_token_count": sum(x["content_token_count"] for x in sentences),
            "encoded_content_token_count": sum(len(x["content_token_ids"]) for x in segments),
            "all_sentence_tokens_encoded": True, "sentences": sentences, "segments": segments}

