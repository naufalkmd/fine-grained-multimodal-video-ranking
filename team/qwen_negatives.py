"""Test-like candidates from the organizers' descriptions, generated with a local open-source LLM.

For each episode, the model rewrites the description as one short sentence (the positive) and makes
single-change variants of that sentence (the negatives), the kind of near-identical candidates the
test set appears to use. The model only generates text. Every filter is a code rule on the text,
never a model judgment or score, so none of this is distillation (docs/brief.md). The prompt has no
example sentences, so no hand-written description reaches the data.

Runs in float16 on one T4 (16 GB), which has no bfloat16. The output JSONL keeps the prompt version,
raw output and check results for the organizers' inspection, and a rerun resumes where it stopped.

    python -m team.qwen_negatives --config cuda/config.json --splits cuda/outputs/baseline/splits.json \
        --model-dir cuda/models/qwen3-4b-instruct-2507 --output cuda/outputs/qwen/candidates.jsonl
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

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
NEGATIVE_KINDS = ("reversed_action", "added_action", "swapped_order", "changed_detail")
PROMPT_VERSION = 3
PROMPT = """You are given a description of a short video. Rewrite it and create altered versions.
Return only a JSON object with these keys:
- "positive": one short sentence of 8 to 15 words describing the main activity in the video. \
Mention at most two actions, in their original order. Leave out positions, colors and minor steps. \
Use plain words.
- "reversed_action": the positive sentence with exactly one action verb replaced by the verb for \
the opposite action. Change nothing else.
- "added_action": the positive sentence with one extra action added at the end, joined with "and \
then". The extra action must fit the same scene but must not appear anywhere in the description.
- "swapped_order": the positive sentence rewritten so that its two actions happen in the opposite \
order, reusing the same words. Fill this whenever the positive sentence has two actions; use null \
only when it has a single action.
- "changed_detail": the positive sentence with exactly one object noun replaced by a different \
object that could plausibly be there. Keep every verb. Change nothing else.
Write nothing outside the JSON object.

Description: {description}"""


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def changed_words(a, b):
    """Words removed from a and added in b, by a word-level diff."""
    removed = added = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += i2 - i1
            added += j2 - j1
    return removed, added


def check(kind, positive, negative):
    """Code rules for one negative; the reason is stored so rejected candidates can be inspected."""
    if not isinstance(negative, str) or not negative.strip():
        return "missing"
    p, n = words(positive), words(negative)
    if p == n:
        return "same_as_positive"
    removed, added = changed_words(p, n)
    if kind in ("reversed_action", "changed_detail"):
        limit = 2 if kind == "reversed_action" else 3
        if not (1 <= removed <= limit and 1 <= added <= limit):
            return f"changed_{removed}_{added}_words"
    elif kind == "added_action":
        if n[:len(p)] != p:
            return "does_not_extend_positive"
        if not 2 <= len(n) - len(p) <= 12:
            return f"added_{len(n) - len(p)}_words"
    elif kind == "swapped_order":
        if sum(((Counter(p) - Counter(n)) + (Counter(n) - Counter(p))).values()) > 4:
            return "not_a_reordering"
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


def checked_record(episode, split, raw):
    parsed = parse(raw)
    record = {"episode_id": episode.episode_id, "split": split, "prompt_version": PROMPT_VERSION,
              "description": episode.description, "raw": raw, "positive": None, "negatives": {},
              "checks": {}}
    positive = parsed.get("positive") if parsed else None
    if not isinstance(positive, str) or not 4 <= len(words(positive)) <= 20:
        record["checks"]["positive"] = "missing" if parsed else "unparsed"
        return record
    record["positive"] = positive.strip()
    record["checks"]["positive"] = "ok"
    seen = {" ".join(words(positive))}
    for kind in NEGATIVE_KINDS:
        negative = parsed.get(kind)
        result = check(kind, positive, negative)
        if result == "ok" and " ".join(words(negative)) in seen:
            result = "duplicate"
        record["checks"][kind] = result
        if result == "ok":
            record["negatives"][kind] = negative.strip()
            seen.add(" ".join(words(negative)))
    return record


@torch.inference_mode()
def generate(model, tokenizer, descriptions, max_new_tokens):
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(description=text)}],
        tokenize=False, add_generation_prompt=True) for text in descriptions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help=f"Local copy of {MODEL_ID}; generation never downloads.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-names", default="valid,train")
    parser.add_argument("--limit", type=int, help="Episodes per split, for a pilot run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    episodes = {e.episode_id: e for e in load_episodes(config["data_dir"])}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    done = set()
    if args.output.exists():
        done = {json.loads(line)["episode_id"] for line in args.output.read_text(encoding="utf-8").splitlines()}
    todo = []
    for split in args.split_names.split(","):
        ids = [i for i in splits[split] if i not in done][:args.limit]
        todo += [(episodes[i], split) for i in ids]
    print(f"Episodes to generate: {len(todo)} (already done: {len(done)})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                 dtype=torch.float16, device_map="cuda").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, results = time.perf_counter(), Counter()
    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            raws = generate(model, tokenizer, [e.description for e, _ in batch], args.max_new_tokens)
            for (episode, split), raw in zip(batch, raws):
                record = checked_record(episode, split, raw)
                results.update(f"{key}:{value}" for key, value in record["checks"].items())
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            elapsed = time.perf_counter() - started
            print(f"{start + len(batch)}/{len(todo)} episodes, {elapsed / 60:.1f} min", flush=True)
    print(json.dumps(dict(sorted(results.items())), indent=2))


if __name__ == "__main__":
    main()
