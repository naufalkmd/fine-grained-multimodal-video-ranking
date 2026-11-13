"""Test-like validation candidates: full-length descriptions with plausible errors written by an LLM.

The test's negatives seem to be natural-sounding edits of full descriptions, probably written by an
LLM (docs/experiments/validation.md, "Writing style and its reversal"). Our rule-based negatives behave
differently on wording, so this asks a local Qwen3-4B-Instruct to rewrite each validation
description into four versions that no longer match the video, either keeping most of the wording
(`--style edit`) or rewording it (`--style rewrite`). The prompt names kinds of change but no words
or examples. Only code rules filter the output, and these sets are for validation only: they never
train a model.

Runs in float16 on one T4 (16 GB). Greedy decoding, so a rerun gives the same text. The JSONL keeps
the raw output and check results, and a rerun resumes where it stopped.

    python -m team.candidates.qwen_long_negatives --splits cuda/outputs/baseline/splits.json \
        --model-dir cuda/models/qwen3-4b-instruct-2507 --output cuda/outputs/qwen/long_negatives.jsonl
"""

import argparse
import difflib
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

from common import ROOT, load_config
from dataset import load_episodes

KINDS = ("reversed_action", "added_action", "changed_detail", "changed_order")
# "edit" keeps most of the wording; "rewrite" has each version reworded in the model's own words.
STYLES = {
    "edit": (1, "keep most of the original wording"),
    "rewrite": (2, "be reworded in your own words rather than copied"),
}
PROMPT = """Below is a description of a short video. Write four incorrect versions of it for a \
multiple-choice test. Each version must read like a natural description of the same scene and \
{wording}, but must no longer match the video because of one change:
- "reversed_action": an action is replaced by its opposite.
- "added_action": an action that does not happen is added.
- "changed_detail": an object, color, position or other detail is changed.
- "changed_order": two actions happen in a different order.
Return only a JSON object with these four keys, each holding the full incorrect description.

Description: {description}"""


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def check(description, negative, seen):
    """Code rules for one negative; the reason is stored so rejected text can be inspected."""
    if not isinstance(negative, str) or not negative.strip():
        return "missing"
    d, n = words(description), words(negative)
    if " ".join(n) in seen:
        return "duplicate"
    if not 0.6 * len(d) <= len(n) <= 1.6 * len(d):
        return "length"
    similarity = difflib.SequenceMatcher(a=d, b=n, autojunk=False).ratio()
    if similarity < 0.2:
        return "unrelated"
    return "ok"


def parse(raw):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def checked_record(episode, raw, prompt_version):
    parsed = parse(raw) or {}
    record = {"episode_id": episode.episode_id, "prompt_version": prompt_version,
              "description": episode.description, "raw": raw, "negatives": {}, "checks": {}}
    seen = {" ".join(words(episode.description))}
    for kind in KINDS:
        negative = parsed.get(kind)
        result = check(episode.description, negative, seen) if parsed else "unparsed"
        record["checks"][kind] = result
        if result == "ok":
            record["negatives"][kind] = negative.strip()
            seen.add(" ".join(words(negative)))
    return record


@torch.inference_mode()
def generate(model, tokenizer, descriptions, max_new_tokens, wording):
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(description=text, wording=wording)}],
        tokenize=False, add_generation_prompt=True) for text in descriptions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", choices=sorted(STYLES), default="edit")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    episodes = {e.episode_id: e for e in load_episodes(config["data_dir"])}
    ids = json.loads(args.splits.read_text(encoding="utf-8"))[args.split][:args.limit]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["episode_id"]
                for line in args.output.read_text(encoding="utf-8").splitlines()}
    # Similar lengths in a batch waste less padding.
    todo = sorted((episodes[i] for i in ids if i not in done), key=lambda e: len(e.description))
    print(f"Episodes to generate: {len(todo)} (already done: {len(done)})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                              padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                 dtype=torch.float16, device_map="cuda").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, results = time.perf_counter(), Counter()
    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompt_version, wording = STYLES[args.style]
            raws = generate(model, tokenizer, [e.description for e in batch], args.max_new_tokens,
                            wording)
            for episode, raw in zip(batch, raws):
                record = checked_record(episode, raw, prompt_version)
                results.update(f"{key}:{value}" for key, value in record["checks"].items())
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{start + len(batch)}/{len(todo)} episodes, "
                  f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    print(json.dumps(dict(sorted(results.items())), indent=2))


if __name__ == "__main__":
    main()
