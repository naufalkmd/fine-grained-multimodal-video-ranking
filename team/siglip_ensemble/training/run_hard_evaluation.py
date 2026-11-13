"""Inference-only challenge evaluation of immutable E0--E3 checkpoints.

All artificial negatives remain rule assumptions, never visually verified truth.
No training, teacher labels, model selection, or competition submission occurs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import gzip
import hashlib
import json
import os
import re
import html
from pathlib import Path
import shutil
import statistics
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from data_contract import load_contracts, read_manifest
from experiment_io import atomic_json, digest, file_record, immutable_json as _immutable_json, read_json, verified_file
from factorial_models import build_model
from hard_negative_data import build_suite
from prepare_features import load_features
from prepare_team_validation import load_team_validation, read_action_labels
from run_sentence_comparison import evaluate
from sentence_features import (MODEL_REVISION, _encode_plans, load_sentence_features,
                               sentence_plan, sentence_spans)
from siglip_model import FrozenSiglip, MODEL_ID, make_text_plan
from temporal_models import collate_frames

VERSION = "frozen-factorial-true-append-eval-v2"
TIE_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-5
SEEDS = [2026, 2027, 2028]
CACHE_DIAGNOSTIC_DIR = None


def json_form(value):
    """Compare JSON content, not Python-only integer keys or tuples.

    Do not change the original request object: its existing digest and all
    saved cache fingerprints/shard checksums must remain unchanged.
    """
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def normalize_encoder_metadata(encoder):
    original=encoder.provenance
    normalized=json_form(original)
    if digest(original)!=digest(normalized):
        raise ValueError("JSON metadata normalization would change the original fingerprint; cache not modified")
    if original!=normalized:
        print("Normalize encoder metadata JSON types; original fingerprint preserved.",flush=True)
    encoder.provenance=normalized


def cache_differences(saved,current,path="$",limit=30):
    differences=[]
    def visit(a,b,p):
        if len(differences)>=limit or a==b: return
        if isinstance(a,dict) and isinstance(b,dict):
            for key in sorted(set(a)|set(b)):
                if key not in a or key not in b:
                    differences.append({"path":p+"."+key,"saved":repr(a.get(key,"<missing>"))[:250],"current":repr(b.get(key,"<missing>"))[:250]})
                else: visit(a[key],b[key],p+"."+key)
                if len(differences)>=limit: break
        elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
            for i,(x,y) in enumerate(zip(a,b)):
                visit(x,y,f"{p}[{i}]")
                if len(differences)>=limit: break
        else: differences.append({"path":p,"saved":repr(a)[:250],"current":repr(b)[:250]})
    visit(saved,current,path)
    return differences


def immutable_json(path,value):
    path=Path(path)
    if not path.exists():
        return _immutable_json(path,value)
    saved=read_json(path)
    normalized=json_form(value)
    if saved==normalized:
        if saved!=value:
            print("Cache JSON representation normalized; all stored values match:",path,flush=True)
        return
    detail={"cache_request":str(path),"differences":cache_differences(saved,normalized),
            "note":"Only JSON representation is normalized; actual setting differences remain blocking."}
    print("CACHE CONFIGURATION DIFFERENCES: "+json.dumps(detail,ensure_ascii=False),flush=True)
    if CACHE_DIAGNOSTIC_DIR is not None:
        atomic_json(Path(CACHE_DIAGNOSTIC_DIR)/("cache_mismatch_"+path.parent.name+".json"),detail)
    raise ValueError("Existing artifact has different content after JSON normalization: "+str(path))


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("run_hard_evaluation.py", "hard_negative_data.py")}


def compare_predictions(actual, saved, tolerance=SCORE_TOLERANCE, diagnostic_path=None):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Score tolerance must be finite and positive")
    a = {r["episode_id"]: r for r in actual["predictions"]}
    b = {r["episode_id"]: r for r in saved["predictions"]}
    if len(a) != len(actual["predictions"]) or len(b) != len(saved["predictions"]) or not a or set(a) != set(b):
        raise ValueError("Original prediction episode coverage differs")
    maximum, rank_changes = 0.0, 0
    differences, per_episode = [], []
    for ep, row in a.items():
        old = b[ep]
        if row["correct_choice_id"] != old["correct_choice_id"] or set(row["scores"]) != set(old["scores"]):
            raise ValueError("Original prediction choices differ")
        values = [abs(float(v) - float(old["scores"][k])) for k, v in row["scores"].items()]
        if not all(np.isfinite(values)):
            raise ValueError("Nonfinite original prediction difference")
        maximum = max(maximum, *values)
        rank_changes += row["ranking"] != old["ranking"]
        differences.extend(values)
        for prediction in (row, old):
            if len(prediction["ranking"]) != len(prediction["scores"]) or set(prediction["ranking"]) != set(prediction["scores"]):
                raise ValueError("Invalid original prediction ranking")
        correct = row["correct_choice_id"]
        per_episode.append({"episode_id":ep,"maximum_score_difference":max(values),
                            "old_ranking":old["ranking"],"new_ranking":row["ranking"],
                            "old_correct_rank":old["ranking"].index(correct)+1,
                            "new_correct_rank":row["ranking"].index(correct)+1})
    report = {"episodes": len(a), "maximum_score_difference": maximum,
              "mean_absolute_score_difference":float(np.mean(differences)),
              "rms_score_difference":float(np.sqrt(np.mean(np.square(differences)))),
              "p99_absolute_score_difference":float(np.quantile(differences,.99)),
              "scores_above_0_001":sum(x>1e-3 for x in differences),
              "rank_changes":rank_changes,
              "top1_choice_changes":sum(r["old_ranking"][0]!=r["new_ranking"][0] for r in per_episode),
              "correct_rank_changes":sum(r["old_correct_rank"]!=r["new_correct_rank"] for r in per_episode),
              "top1_accuracy_delta":float(np.mean([(r["new_correct_rank"]==1)-(r["old_correct_rank"]==1) for r in per_episode])),
              "tolerance":tolerance,"passed":maximum<=tolerance,
              "exceeds_original_fresh_tolerance":maximum>1e-3,
              "worst_episodes":sorted(per_episode,key=lambda r:r["maximum_score_difference"],reverse=True)[:20],
              "changed_rankings":[r for r in per_episode if r["old_ranking"]!=r["new_ranking"]],
              "note":"Numerical compatibility only, not proof of identical features or the cause of drift. Current scores are recomputed for every checkpoint."}
    if diagnostic_path is not None:
        Path(diagnostic_path).parent.mkdir(parents=True,exist_ok=True)
        atomic_json(diagnostic_path,report)
    if maximum > tolerance:
        raise ValueError(f"Reused checkpoint score drift {maximum:g} exceeds {tolerance:g}; stop rather than mix runs")
    return report


def text_cache_request(directory, request):
    """Reuse only the shipped v2 cache whose encoding algorithm is unchanged.

    The sole permitted mismatch is the runner source hash, changed by this
    recovery's diagnostics/CLI. Every data, tokenizer, encoder, input and other
    source field still matches exactly; shard hashes retain the old request.
    """
    path=Path(directory)/"request.json"
    if path.exists():
        previous=read_json(path)
        allowed="098a767d99ad43687460777d91b2cea77615568c52bac6cf57f8dcbb25815361"
        if previous.get("sources",{}).get("run_hard_evaluation.py")==allowed:
            compatible={**request,"sources":{**request["sources"],"run_hard_evaluation.py":allowed}}
            if previous==json_form(compatible):
                print("Reuse verified v2 text cache; encoding algorithm unchanged.",flush=True)
                return previous
    immutable_json(path,request)
    return request


def verify_source_run(folder, split, random_data, team_data, bank, sentence_bank):
    folder = Path(folder)
    status, config = read_json(folder / "status.json"), read_json(folder / "run_config.json")
    if status.get("status") != "complete" or status.get("run_hash") != digest(config):
        raise ValueError("Original factorial run is incomplete or has changed")
    for name, record in status["artifacts"].items():
        verified_file(folder / name, record)
    if config.get("version") != "siglip-frame-reorder-factorial-v1" or config.get("smoke"):
        raise ValueError("Need the full completed E0--E3 factorial run")
    required = {"split_hash": digest(split), "random_candidates_hash": digest(random_data),
                "team_candidates_hash": digest(team_data), "features": bank["index"]["fingerprint"],
                "sentence_features": sentence_bank["index"]["fingerprint"], "seeds": SEEDS}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError("Original factorial configuration mismatch: " + key)
    for label, data in (("random", random_data), ("team", team_data)):
        if config["evaluated_candidates_hash"][label] != digest(data["episodes"]):
            raise ValueError("Original evaluated candidate rows differ: " + label)
    root = Path(__file__).resolve().parent
    for relative, expected in config["sources"].items():
        path = root / relative
        if not path.is_file():
            raise ValueError("Missing frozen model source: " + relative)
        actual = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        if actual != expected and file_record(path)["sha256"] != expected:
            raise ValueError("Frozen model source changed: " + relative)
    return config, status


def checkpoint_inventory(folder, config):
    inventory = []
    for experiment in ("E0", "E1", "E2", "E3"):
        for seed in SEEDS:
            head = Path(folder) / "heads" / experiment / f"seed{seed}"
            if not (head / "best.pt").is_file():
                raise FileNotFoundError(f"기존 Drive 가중치가 필요합니다: {head / 'best.pt'} (결과 ZIP에는 없습니다)")
            marker, result = read_json(head / "complete.json"), read_json(head / "result.json")
            identity = {"run_hash": digest(config), "experiment": experiment, "seed": seed}
            for record in (marker, result):
                if any(record.get(k) != v for k, v in identity.items()):
                    raise ValueError("Checkpoint completion identity differs: " + str(head))
            for name in ("best.pt", "result.json"):
                verified_file(head / name, marker["artifacts"][name])
            checkpoint = torch.load(head / "best.pt", map_location="cpu", weights_only=True)
            if (any(checkpoint.get(k) != v for k, v in identity.items()) or
                checkpoint.get("model_config") != config["model"] or
                checkpoint.get("epoch") != result["best_epoch"] or
                checkpoint.get("frame_transformer") != config["experiments"][experiment]["frame_transformer"]):
                raise ValueError("Checkpoint configuration differs: " + str(head))
            inventory.append({**identity, "head": head, "result": result,
                              "checkpoint_record": marker["artifacts"]["best.pt"],
                              "epoch": result["best_epoch"]})
    return inventory


def load_portable_models(path,weights_dir,split,random_data):
    """Verify uploaded weights against the previously shared immutable results.

    No old Drive caches or completion files are needed; the portable metadata
    contains their trusted checkpoint hashes and saved original predictions.
    """
    portable=read_json(path)
    config=portable["run_config"]
    if (portable.get("version")!="factorial-portable-v1" or portable.get("run_hash")!=digest(config) or
        config.get("version")!="siglip-frame-reorder-factorial-v1" or config.get("smoke") or config.get("seeds")!=SEEDS):
        raise ValueError("Invalid portable factorial run identity")
    if config.get("split_hash")!=digest(split) or config.get("random_candidates_hash")!=digest(random_data):
        raise ValueError("Portable source split or original random choices differ")
    if portable.get("random_candidates")!=random_data:
        raise ValueError("Embedded random candidates differ from recreated contracts")
    team=portable.get("team_candidates")
    if not team or config.get("team_candidates_hash")!=digest(team):
        raise ValueError("Embedded original team candidates differ")
    valid=set(split["val_ids"])
    for label,data in (("random",random_data),("team",team)):
        if config["evaluated_candidates_hash"][label]!=digest(data["episodes"]):
            raise ValueError("Portable evaluated candidate rows differ: "+label)
        if {r["episode_id"] for r in data["episodes"]}!=valid:
            raise ValueError("Portable choices do not cover the fixed validation split")
        if any(c["source_episode_id"] not in valid for r in data["episodes"] for c in r["choices"]):
            raise ValueError("Fresh validation cache cannot use non-validation original captions")
    root=Path(__file__).resolve().parent
    for name,expected in config["sources"].items():
        source=root/name
        if not source.is_file() or expected not in (hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest(),file_record(source)["sha256"]):
            raise ValueError("Frozen portable model source differs: "+name)
    records={(r["experiment"],r["seed"]):r for r in portable["heads"]}
    required={(e,s) for e in ("E0","E1","E2","E3") for s in SEEDS}
    if len(records)!=len(portable["heads"]) or set(records)!=required:
        raise ValueError("All twelve portable trained checkpoints are mandatory")
    inventory=[]
    for experiment,seed in sorted(required):
        row=records[(experiment,seed)]; head=Path(weights_dir)/experiment/f"seed{seed}"
        path=head/"best.pt"
        if not path.is_file():raise FileNotFoundError(f"Upload all twelve best.pt files; missing {path}")
        verified_file(path,row["checkpoint_record"])
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        identity={"run_hash":digest(config),"experiment":experiment,"seed":seed}
        if (any(checkpoint.get(k)!=v for k,v in identity.items()) or checkpoint.get("epoch")!=row["best_epoch"] or
            checkpoint.get("model_config")!=config["model"] or
            checkpoint.get("frame_transformer")!=config["experiments"][experiment]["frame_transformer"]):
            raise ValueError("Uploaded checkpoint configuration differs: "+str(path))
        if any(row["result"].get(k)!=v for k,v in identity.items()):
            raise ValueError("Portable saved result identity differs")
        inventory.append({**identity,"head":head,"result":row["result"],
                          "checkpoint_record":row["checkpoint_record"],"epoch":row["best_epoch"]})
    return portable,config,team,inventory


def load_extra_candidates(path,manifest,split,action_labels=None):
    """Accept only explicit local-generator records, never teacher scores/labels."""
    if not path: return None
    value=read_json(path)
    request=value.get("request",{})
    if request.get("split_hash") != digest(split) or request.get("manifest_hash") != digest(manifest):
        raise ValueError("Extra candidate generation used a different manifest or split")
    sources=request.get("sources",{})
    for name in ("qwen_generate_candidates.py","hard_negative_data.py"):
        source=Path(__file__).resolve().parent/name
        if (not source.is_file() or sources.get(name) != hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()):
            raise ValueError("Extra generation source differs from the submitted reproducible code: "+name)
    original={r["episode_id"]:r["description"] for r in manifest["episodes"]}
    metadata={r["episode_id"]:r.get("metadata_sha256") for r in manifest["episodes"]}
    valid=set(split["val_ids"])
    seen=set()
    records=value.get("records")
    if not isinstance(records,list): raise ValueError("Extra candidate records must be a list")
    for row in records:
        if (not isinstance(row,dict) or not isinstance(row.get("id"),str) or not row["id"] or
            row["id"] in seen or row.get("episode_id") not in valid):
            raise ValueError("Invalid/duplicate extra candidate ID or validation episode")
        base=original[row["episode_id"]]
        if row.get("original_text") != base:
            raise ValueError("Append source must be the untouched official caption")
        if row.get("family") != "qwen_true_append":
            raise ValueError("Only genuine appended events are accepted; replacement records are obsolete")
        for polarity in ("positive","negative"):
            suffix=row.get(polarity+"_suffix")
            if (not isinstance(suffix,str) or not suffix.strip() or suffix != suffix.strip() or
                row.get(polarity+"_text") != base+" "+suffix or len(sentence_spans(suffix)) != 1):
                raise ValueError("Both candidates must preserve the complete original and append one sentence")
        if row["positive_suffix"] == row["negative_suffix"] or not isinstance(row.get("type"),str):
            raise ValueError("Invalid or identical appended events")
        if row.get("semantic_status") != "unverified_event_absence":
            raise ValueError("Generated negatives must not claim verified teacher truth labels")
        if not all(isinstance(row.get(p+"_action_family"),str) and row[p+"_action_family"] for p in ("positive","negative")):
            raise ValueError("Append action families are required")
        if row["positive_action_family"] == row["negative_action_family"]:
            raise ValueError("A different concrete action family is required")
        support=row.get("evidence",{}).get("positive_action_label")
        if not support or support not in (action_labels or {}).get(row["episode_id"],[]):
            raise ValueError("Supported positive event must have an exact official episode action label")
        evidence=row.get("evidence",{})
        if metadata[row["episode_id"]] and evidence.get("metadata_sha256")!=metadata[row["episode_id"]]:
            raise ValueError("Appended event annotation source differs from the source ZIP")
        seen.add(row["id"])
    return value


def prepare_append_groups(records,tokenizer,action_labels=None):
    """Length-exact pairs; reciprocal families selected without model scores.

    This controls action-family frequency, not every language artifact and not
    visual label noise. Unpaired/novel actions never enter the balanced group.
    """
    edges=defaultdict(list)
    for row in records:
        if action_labels is not None:
            from qwen_generate_candidates import validate_append_record
            validate_append_record(row,tokenizer,action_labels.get(row["episode_id"],[]))
        p,n=(row[k] for k in ("positive_text","negative_text"))
        if (len(tokenizer.encode(p,add_special_tokens=False,truncation=False)) !=
            len(tokenizer.encode(n,add_special_tokens=False,truncation=False)) or
            len(re.findall(r"\b\w+\b",p)) != len(re.findall(r"\b\w+\b",n)) or
            len(sentence_spans(p)) != len(sentence_spans(n))):
            raise ValueError("Appended pair must have exact SigLIP token, word and sentence counts")
        edges[(row["positive_action_family"],row["negative_action_family"])].append(row)
    balanced=set()
    for (a,b),rows in sorted(edges.items()):
        if a>=b: continue
        reverse=edges.get((b,a),[])
        forward_deltas=defaultdict(list);reverse_deltas=defaultdict(list)
        for row in rows:forward_deltas[len(row["negative_text"])-len(row["positive_text"])].append(row)
        for row in reverse:reverse_deltas[len(row["negative_text"])-len(row["positive_text"])].append(row)
        for delta,forward_rows in sorted(forward_deltas.items()):
            reverse_rows=reverse_deltas.get(-delta,[])
            count=min(len(forward_rows),len(reverse_rows))
            balanced.update(r["id"] for r in sorted(forward_rows,key=lambda r:r["id"])[:count])
            balanced.update(r["id"] for r in sorted(reverse_rows,key=lambda r:r["id"])[:count])
    output=[]
    frequencies={"positive":defaultdict(int),"negative":defaultdict(int)}
    for row in records:
        family=("qwen_append_balanced" if row["id"] in balanced else
                "qwen_append_novel_auxiliary" if row["negative_action_family"] in {"sit_down","stand_up"}
                else "qwen_append_unbalanced_auxiliary")
        output.append({**row,"family":family,"source_record_id":row["id"]})
        if row["id"] in balanced:
            for polarity in frequencies: frequencies[polarity][row[polarity+"_action_family"]]+=1
        # Natural append is intentionally unmatched. Never pool its score into
        # the exact-length balanced result.
        output.append({**row,"id":row["id"]+"_natural","source_record_id":row["id"],
            "positive_text":row["original_text"],"family":"append_unmatched_auxiliary",
            "type":"original_vs_literal_extra_event",
            "diagnostic_warning":"Positive is the shorter original. Length-confounded auxiliary only."})
    if dict(frequencies["positive"]) != dict(frequencies["negative"]):
        raise AssertionError("Reciprocal family balancing failed")
    return output,{"source_pairs":len(records),"balanced_pairs":len(balanced),
        "unbalanced_pairs":len(records)-len(balanced),
        "action_family_counts":{k:dict(v) for k,v in frequencies.items()},
        "method":"Reciprocal action families AND mirrored character-length deltas; stable record IDs, never scores",
        "warning":"Exact token/word/sentence counts do not remove all language bias; absent actions remain unverified."}


def write_review(directory,records):
    """Reviewable candidates only, no automatic visual truth decisions."""
    rows=[r for r in records if r.get("original_text") and r["family"]!="append_unmatched_auxiliary"]
    fields=["id","episode_id","family","original_text","positive_text","negative_text",
            "positive_action_family","negative_action_family","semantic_status"]
    with (Path(directory)/"append_review.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    cards=[]
    for r in rows:
        cards.append('<article><h3>'+html.escape(r['episode_id'])+'</h3>'+''.join(
            '<p><b>'+html.escape(k)+':</b> '+html.escape(str(r.get(k,'')))+'</p>'
            for k in fields if k not in ('id','episode_id'))+'</article>')
    page='<meta charset="utf-8"><title>Append candidate review</title><style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}article{border-top:1px solid #aaa;padding:16px}</style><h1>검토용 사건 추가 후보</h1><p>참 사건 근거는 공식 행동 라벨입니다. 추가 오답 사건의 실제 부재는 미검증입니다. 이 페이지는 후보 문장을 편집하거나 정답을 판정하지 않습니다.</p>'+''.join(cards)
    (Path(directory)/"append_review.html").write_text(page,encoding="utf-8")


def encode_text_bank(records, original_bank, original_sentences, manifest, encoder, directory):
    """Reuse original vectors; new text shards are immutable and resumable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    texts = sorted({r[k] for r in records for k in ("positive_text", "negative_text")})
    ids = {text: "text_" + digest(text) for text in texts}
    originals = {}
    for row in manifest["episodes"]:
        if row["episode_id"] in original_bank["texts"]:
            originals.setdefault(row["description"], row["episode_id"])
    bank = {"frames": original_bank["frames"], "texts": {}}
    sentences = {"features": {}, "positions": {}, "weights": {}}
    original_base_inputs = {}
    for ep,key in original_bank["token_keys"].items():
        original_base_inputs.setdefault(key,original_bank["texts"][ep])
    plans, audits = {}, {}
    for text in texts:
        key = ids[text]
        plan = sentence_plan(encoder.tokenizer, text)
        base = make_text_plan(encoder.tokenizer, text)["token_inputs"]["truncate64"][0]
        plans[key] = plan
        audits[key] = {"base_ids": base, "sentence_ids": [s["token_ids"] for s in plan["segments"]],
                       "positions": [s["position"] for s in plan["segments"]],
                       "weights": [s["weight"] for s in plan["segments"]],
                       "full_token_count": plan["original_full_token_count"],
                       "sentence_count": plan["sentence_count"]}
        if text in originals:
            ep = originals[text]
            bank["texts"][key] = original_bank["texts"][ep]
            for field in sentences:
                sentences[field][key] = original_sentences[field][ep]
    request = {"version": VERSION, "texts_hash": digest(texts),
               "original_features": original_bank["index"]["fingerprint"],
               "original_sentences": original_sentences["index"]["fingerprint"],
               "encoder": encoder.provenance, "text_batch_size": encoder.text_batch_size,
               "token_inputs_hash": digest(audits), "sources": source_hashes()}
    request = text_cache_request(directory, request)
    missing = [t for t in texts if ids[t] not in bank["texts"]]
    for start in range(0, len(missing), 64):
        current = missing[start:start + 64]
        keys = [ids[t] for t in current]
        target = directory / f"shard-{start:05d}.npz"
        marker = target.with_suffix(".json")
        expected = {"request_hash": digest(request), "keys": keys}
        if marker.exists():
            metadata = read_json(marker)
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("New text shard request changed")
            verified_file(target, metadata["file"])
        else:
            current_plans = [plans[k] for k in keys]
            sentence_vectors = _encode_plans(encoder, current_plans)
            base_vectors = []
            with torch.inference_mode():
                for b in range(0, len(keys), encoder.text_batch_size):
                    tensor = torch.tensor([audits[k]["base_ids"] for k in keys[b:b+encoder.text_batch_size]],
                                          dtype=torch.long, device=encoder.device)
                    base_vectors.append(encoder._unit(encoder.model.get_text_features(input_ids=tensor)).cpu().numpy())
            arrays = {"base": np.concatenate(base_vectors).astype(np.float32),
                      "sentences": np.concatenate(sentence_vectors).astype(np.float32),
                      "offsets": np.array([0] + list(np.cumsum([len(x) for x in sentence_vectors])), dtype=np.int64)}
            temporary = target.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(target)
            atomic_json(marker, {**expected, "file": file_record(target)})
        with np.load(target, allow_pickle=False) as values:
            bases, all_sentences, offsets = values["base"], values["sentences"], values["offsets"]
            if (bases.shape != (len(keys), 768) or offsets.shape != (len(keys)+1,) or offsets[0] != 0 or
                np.any(np.diff(offsets) < 1) or all_sentences.shape != (int(offsets[-1]), 768)):
                raise ValueError("Invalid new feature shard shape")
            for vectors in (bases, all_sentences):
                if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
                    raise ValueError("Invalid new text vectors")
            for i, key in enumerate(keys):
                a, b = map(int, offsets[i:i+2])
                if b-a != len(audits[key]["positions"]):
                    raise ValueError("Sentence feature coverage changed")
                bank["texts"][key] = torch.from_numpy(bases[i].copy())
                existing=original_base_inputs.get(digest([audits[key]["base_ids"]]))
                if existing is not None:
                    # An identical token input must have exactly the cached base
                    # score, not batch-dependent FP16 re-encoding noise.
                    bank["texts"][key] = existing
                sentences["features"][key] = torch.from_numpy(all_sentences[a:b].copy())
                sentences["positions"][key] = torch.tensor(audits[key]["positions"], dtype=torch.float32)
                sentences["weights"][key] = torch.tensor(audits[key]["weights"], dtype=torch.int64)
        print(f"New candidate text features {min(start+len(current),len(missing))}/{len(missing)}", flush=True)
    return ids, bank, sentences, audits


def verify_encoder_anchors(encoder,manifest,bank,count=12):
    """Sanity-check numerical compatibility; exact token collisions reuse vectors."""
    rows=[r for r in manifest["episodes"] if r["episode_id"] in bank["texts"]][:count]
    differences=[]
    with torch.inference_mode():
        token_ids=[make_text_plan(encoder.tokenizer,r["description"])["token_inputs"]["truncate64"][0] for r in rows]
        for start in range(0,len(rows),encoder.text_batch_size):
            x=torch.tensor(token_ids[start:start+encoder.text_batch_size],dtype=torch.long,device=encoder.device)
            features=encoder._unit(encoder.model.get_text_features(input_ids=x)).cpu()
            for row,feature in zip(rows[start:start+encoder.text_batch_size],features):
                differences.append(float((feature-bank["texts"][row["episode_id"]]).abs().max()))
    maximum=max(differences,default=0.)
    if maximum>1e-3: raise ValueError(f"Current SigLIP differs numerically from original text cache: {maximum:g}")
    return {"anchors":len(rows),"maximum_vector_element_difference":maximum,"tolerance":1e-3,
            "note":"Numerical sanity check only; original-token base collisions reuse cached vectors exactly."}


def collate_choices(episodes, choices, bank, sentence_bank, device):
    """Variable candidate count, preserving exactly the trained feature layout."""
    if not choices or len({len(c) for c in choices}) != 1:
        raise ValueError("A batch needs a fixed nonzero candidate count")
    frames, frame_mask = collate_frames([bank["frames"][ep] for ep in episodes])
    texts = torch.stack([torch.stack([bank["texts"][k] for k in row]) for row in choices])
    maximum = max(len(sentence_bank["features"][k]) for row in choices for k in row)
    shape = (len(episodes), len(choices[0]), maximum)
    sentences = texts.new_zeros((*shape, texts.shape[-1]))
    mask = torch.zeros(shape, dtype=torch.bool)
    positions, weights = texts.new_zeros(shape), texts.new_zeros(shape)
    for i, row in enumerate(choices):
        for j, key in enumerate(row):
            size = len(sentence_bank["features"][key])
            sentences[i,j,:size] = sentence_bank["features"][key]
            mask[i,j,:size] = True
            positions[i,j,:size] = sentence_bank["positions"][key]
            weights[i,j,:size] = sentence_bank["weights"][key]
    return tuple(x.to(device) for x in (frames, frame_mask, texts, sentences, mask, positions, weights))


def score_pairs(model, records, ids, bank, sentences, device, batch_size):
    if model is not None:
        model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records[start:start+batch_size]
            tensors = collate_choices([r["episode_id"] for r in rows],
                       [[ids[r["positive_text"]], ids[r["negative_text"]]] for r in rows], bank, sentences, device)
            if model is None:
                frames, mask, text = tensors[:3]
                video = F.normalize((frames * mask.unsqueeze(-1)).sum(1) / mask.sum(1,keepdim=True), dim=-1)
                base = (video[:,None,:] * text).sum(-1)
                scores = base
            else:
                components = model.components(*tensors)
                scores, base = components["scores"], components["base"]
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite challenge scores")
            for row, values, original in zip(rows, scores.cpu().tolist(), base.cpu().tolist()):
                margin = values[0]-values[1]
                predictions.append({"id":row["id"], "episode_id":row["episode_id"],
                                    "positive_score":values[0], "negative_score":values[1],
                                    "base_margin":original[0]-original[1], "margin":margin,
                                    "win":tie_win(margin), "tie":abs(margin)<=TIE_TOLERANCE})
    return predictions


def tie_win(margin, tolerance=TIE_TOLERANCE):
    return 1.0 if margin > tolerance else 0.0 if margin < -tolerance else 0.5


def grouped_metrics(records, predictions):
    if len(records) != len(predictions) or any(r["id"] != p["id"] for r,p in zip(records,predictions)):
        raise ValueError("Prediction record identities differ")
    groups = defaultdict(list)
    for row,pred in zip(records,predictions):
        for group in (row["family"], row["family"]+"/"+row["type"]):
            groups[group].append(pred)
    result = {}
    for group, rows in sorted(groups.items()):
        per_episode = defaultdict(list)
        for row in rows:
            per_episode[row["episode_id"]].append(row["win"])
        result[group] = {"pairs":len(rows), "episodes":len(per_episode),
            "pair_win_rate":statistics.mean(r["win"] for r in rows),
            "episode_macro_win_rate":statistics.mean(statistics.mean(v) for v in per_episode.values()),
            "mean_margin":statistics.mean(r["margin"] for r in rows), "ties":sum(r["tie"] for r in rows)}
    return result


def five_choice_metrics(records, predictions):
    """Only four distinct negatives sharing the exact positive; never easy filler."""
    grouped = defaultdict(list)
    for row, pred in zip(records,predictions):
        grouped[(row["family"], row["episode_id"],row["positive_text"])].append((row,pred))
    output = defaultdict(list)
    for (family, ep, _), rows in grouped.items():
        unique = {}
        for row,pred in sorted(rows,key=lambda x:x[0]["id"]):
            unique.setdefault(row["negative_text"],(row,pred))
        if len(unique)<4:
            continue
        values = list(unique.values())[:4]
        # The scorer is candidate-independent; tolerate numerical batch noise only.
        positives = [p["positive_score"] for _,p in values]
        if max(positives)-min(positives)>SCORE_TOLERANCE:
            raise ValueError("Identical candidate scores vary across pair batches")
        pos = statistics.mean(positives)
        neg = [p["negative_score"] for _,p in values]
        above = sum(x > pos+TIE_TOLERANCE for x in neg)
        tied = sum(abs(x-pos)<=TIE_TOLERANCE for x in neg)
        rewards = [1.0,.5,.25,.125,0.0]
        output[family].append({"episode_id":ep, "pair_ids":[r["id"] for r,_ in values],
            "expected_reward":statistics.mean(rewards[above:above+tied+1]),
            "expected_top1":1/(tied+1) if above==0 else 0.0})
    return {family:{"episodes":len(rows),"mean_rank_reward":statistics.mean(r["expected_reward"] for r in rows),
                    "top1_accuracy":statistics.mean(r["expected_top1"] for r in rows),"predictions":rows,
                    "tie_policy":"Expected metric over uniform ordering within a score tie", "easy_fillers":0}
            for family,rows in output.items()}


def length_audit(records, ids, audits):
    groups = defaultdict(list)
    for row in records:
        p,n = (audits[ids[row[k]]] for k in ("positive_text","negative_text"))
        positive,negative=row["positive_text"],row["negative_text"]
        word_counts=[len(re.findall(r"\b\w+\b",t)) for t in (positive,negative)]
        def overlap(text,base):
            a=set(re.findall(r"\b\w+\b",text.lower())); b=set(re.findall(r"\b\w+\b",base.lower()))
            return len(a&b)/max(1,len(a|b))
        def repeats(text,base):
            def triples(t):
                words=re.findall(r"\b\w+\b",t.lower()); return set(zip(words,words[1:],words[2:]))
            a,b=triples(text),triples(base)
            return len(a&b)/max(1,len(a))
        base=row.get("original_text","")
        ps=row.get("positive_suffix",positive); ns=row.get("negative_suffix",negative)
        if row["family"]=="append_unmatched_auxiliary": ps=""
        groups[row["family"]].append({"episode_id":row["episode_id"],
            "shortest":tie_win(n["full_token_count"]-p["full_token_count"],0),
            "longest":tie_win(p["full_token_count"]-n["full_token_count"],0),
            "fewest_sentences":tie_win(n["sentence_count"]-p["sentence_count"],0),
            "same_base64":p["base_ids"]==n["base_ids"],
            "same_full_input":all(p[k]==n[k] for k in ("base_ids","sentence_ids","positions","weights")),
            "delta_tokens":n["full_token_count"]-p["full_token_count"],
            "fewest_words":tie_win(word_counts[1]-word_counts[0],0),
            "fewest_characters":tie_win(len(negative)-len(positive),0),
            "delta_words":word_counts[1]-word_counts[0],"delta_characters":len(negative)-len(positive),
            "higher_suffix_source_word_overlap":tie_win(overlap(ps,base)-overlap(ns,base),0),
            "higher_suffix_source_trigram_repetition":tie_win(repeats(ps,base)-repeats(ns,base),0)})
    return {name:{"pairs":len(rows), **{key:statistics.mean(float(r[key]) for r in rows)
             for key in ("shortest","longest","fewest_sentences","same_base64","same_full_input","delta_tokens",
                         "fewest_words","fewest_characters","delta_words","delta_characters",
                         "higher_suffix_source_word_overlap","higher_suffix_source_trigram_repetition")},
             "pair_chance":.5,
             "note":"Within-pair blind diagnostics. Overlap metrics are most relevant to append families; exact token equality does not remove other text cues."}
            for name,rows in groups.items()}


def write_gzip(path, value):
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)


def run(args):
    global CACHE_DIAGNOSTIC_DIR
    started = time.perf_counter()
    if args.model_cache:
        os.environ["HF_HOME"] = args.model_cache
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Colab의 무료 T4 GPU 런타임을 선택해주세요.")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights_dir=getattr(args,"weights_dir",None)
    portable_mode=bool(weights_dir)
    score_tolerance=getattr(args,"original_score_tolerance",None)
    if score_tolerance is None: score_tolerance=1e-3 if portable_mode else SCORE_TOLERANCE
    if not np.isfinite(score_tolerance) or not 0<score_tolerance<=2e-3:
        raise ValueError("Original score tolerance must be positive and at most 0.002")
    if not portable_mode and score_tolerance!=SCORE_TOLERANCE:
        raise ValueError("Legacy original-cache verification retains its strict tolerance")
    source = None if portable_mode else Path(args.factorial_run)/"comparison"
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    CACHE_DIAGNOSTIC_DIR=output
    manifest,split,random_data = load_contracts(args.contracts)
    if len(split["val_ids"])!=834 or len(split["train_ids"])!=3336:
        raise ValueError("Full fixed 3336/834 episode split required")
    if read_manifest(args.train_zip) != manifest:
        raise ValueError("Provided training ZIP differs from fixed contracts")
    encoder=None;portable=None;portable_path=None
    if portable_mode:
        portable_path=Path(getattr(args,"portable_models",None) or Path(__file__).with_name("portable_models.json"))
        portable,config,team_data,inventory=load_portable_models(portable_path,weights_dir,split,random_data)
        import prepare_fresh_eval
        # Preserve the frozen source files and fingerprint algorithm. Fix only
        # JSON representation equality in the feature-cache entry point.
        prepare_fresh_eval.immutable_json=immutable_json
        load_or_prepare_eval=prepare_fresh_eval.load_or_prepare_eval
        encoder=FrozenSiglip(device,text_batch_size=args.text_batch_size,revision=MODEL_REVISION)
        normalize_encoder_metadata(encoder)
        fresh_dir=Path(getattr(args,"fresh_feature_dir",None) or output/"fresh_features")
        bank,sentences=load_or_prepare_eval(manifest,split,encoder,fresh_dir,train_zip=args.train_zip)
        print("Fresh original validation features ready; all twelve uploaded model weights verified.",flush=True)
    else:
        bank = load_features(args.feature_dir)
        sentences = load_sentence_features(args.sentence_feature_dir,bank,manifest,split)
        team_data, team_provenance = load_team_validation(args.team_validation_dir,bank,manifest,split)
        config,status = verify_source_run(source,split,random_data,team_data,bank,sentences)
        inventory = checkpoint_inventory(source,config)
    actions = read_action_labels(args.train_zip,manifest,[r["episode_id"] for r in manifest["episodes"]])
    extra_path=getattr(args,"extra_candidates",None)
    extra=load_extra_candidates(extra_path,manifest,split,actions)
    request = {"version":VERSION,"sources":source_hashes(),"factorial_run_hash":digest(config),
               "split_hash":digest(split),"generation_seed":args.seed,"validation_ids":split["val_ids"],
               "checkpoints":[{k:v for k,v in r.items() if k in ("experiment","seed","checkpoint_record","epoch")} for r in inventory],
               "device":device,"batch_size":args.batch_size,"text_batch_size":args.text_batch_size,
               "max_minimal_per_episode":4,"max_additions_per_episode":0,
               "extra_candidates":file_record(extra_path) if extra_path else None,
               "portable_models":file_record(portable_path) if portable_path else None,
               "original_feature_mode":"fresh_validation_only" if portable_mode else "reuse_original_cache",
               "original_score_tolerance":score_tolerance,
               "text_feature_cache_request":file_record(Path(args.text_feature_dir)/"request.json") if getattr(args,"text_feature_dir",None) else None,
               "fresh_feature_fingerprint":bank["index"]["fingerprint"] if portable_mode else None,
               "selection":"none; every fixed checkpoint evaluated","training_steps":0,
               "semantic_warning":"Rule assumptions; event absence is unverified; all scores are development diagnostics."}
    immutable_json(output/"request.json",request)
    source_root=Path(__file__).resolve().parent
    source_names=set(config["sources"]) | set(source_hashes())
    if portable_mode: source_names.add("prepare_fresh_eval.py")
    if extra_path and (source_root/"qwen_generate_candidates.py").is_file(): source_names.add("qwen_generate_candidates.py")
    for name in sorted(source_names):
        original=source_root/name
        snapshot=output/"source"/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        text=original.read_text(encoding="utf-8")
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
            raise ValueError("Source snapshot changed; choose a new output directory: "+name)
        if not snapshot.exists(): snapshot.write_text(text,encoding="utf-8",newline="\n")
    if extra_path:
        # Include the exact used generated data and generation logs, not model weights.
        extra_root=Path(extra_path).resolve().parent
        generation_files=[Path(extra_path).resolve()]
        generation_files += [extra_root/name for name in ("request.json","provenance.json") if (extra_root/name).is_file()]
        generation_files += sorted((extra_root/"jobs").glob("*.json"))
        for original in generation_files:
            relative=original.relative_to(extra_root)
            snapshot=output/"extra_generation"/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if snapshot.exists() and file_record(snapshot)!=file_record(original):
                raise ValueError("Extra generation provenance changed; choose a new output directory")
            if not snapshot.exists(): shutil.copyfile(original,snapshot)
    request_hash = digest(request)
    atomic_json(output/"status.json",{"status":"running","request_hash":request_hash})
    try:
        if encoder is None:encoder = FrozenSiglip(device,text_batch_size=args.text_batch_size,revision=MODEL_REVISION)
        normalize_encoder_metadata(encoder)
        if (encoder.provenance.get("snapshot_commit")!=MODEL_REVISION or
            encoder.provenance["tokenizer"]["backend_sha256"]!=bank["index"]["model"]["tokenizer"]["backend_sha256"]):
            raise ValueError("Candidate encoder differs from original SigLIP")
        anchor_check=verify_encoder_anchors(encoder,manifest,bank)
        atomic_json(output/"encoder_anchor_check.json",anchor_check)
        suite = build_suite(manifest,split,encoder.tokenizer,action_labels=actions,seed=args.seed,max_additions_per_episode=0)
        if extra is not None:
            append_records,balance_audit=prepare_append_groups(extra["records"],encoder.tokenizer,actions)
            suite={"records":suite["records"]+append_records,
                   "audit":{"rule_based":suite["audit"],"extra_generation":extra.get("audit",{}),
                            "append_balance":balance_audit,
                            "extra_request":extra["request"],"extra_file":file_record(extra_path),
                            "note":"Qwen diagnostics remain separate; generated counterfactuals are unverified assumptions."}}
        records = suite["records"]
        if not records:
            raise ValueError("No eligible challenge candidates; examine generator coverage")
        if not {r["episode_id"] for r in records} <= set(split["val_ids"]):
            raise ValueError("Challenge candidates escaped fixed validation split")
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate challenge record IDs")
        immutable_json(output/"generation.json",suite)
        write_review(output,records)
        text_feature_dir=Path(getattr(args,"text_feature_dir",None) or output/"text_features")
        ids,new_bank,new_sentences,token_audits = encode_text_bank(records,bank,sentences,manifest,encoder,text_feature_dir)
        immutable_json(output/"encoder_provenance.json",encoder.provenance)
        # Keep base-only collisions: the full-sentence path can still distinguish them.
        eligible = [r for r in records if not all(token_audits[ids[r["positive_text"]]][k] ==
                    token_audits[ids[r["negative_text"]]][k] for k in ("base_ids","sentence_ids","positions","weights"))]
        excluded = [r["id"] for r in records if r not in eligible]
        if extra is not None:
            # If a complete-input collision removes one orientation, rebalance
            # the surviving source pairs before reporting balanced metrics.
            surviving={r["id"] for r in eligible}
            append_records,balance_audit=prepare_append_groups(
                [r for r in extra["records"] if r["id"] in surviving],encoder.tokenizer,actions)
            eligible=[r for r in eligible if not r.get("original_text")]+append_records
            suite["audit"]["append_balance_after_input_filter"]=balance_audit
        records = eligible
        if not records:
            raise ValueError("All candidate pairs collapse to identical complete model inputs")
        blind = length_audit(records,ids,token_audits)
        atomic_json(output/"evaluated_candidates.json",{"records":records,"excluded_complete_input_collisions":excluded,
                    "audit":suite["audit"]})
        write_review(output,records)
        atomic_json(output/"length_audit.json",blind)
        atomic_json(output/"token_audit.json",{"excluded_identical_full_input":excluded,"input_hash":digest(token_audits),
                    "rows":[{"id":r["id"],"base64_identical":token_audits[ids[r["positive_text"]]]["base_ids"]==token_audits[ids[r["negative_text"]]]["base_ids"]} for r in records]})
        del encoder
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        sets = {"random":random_data["episodes"],"team":team_data["episodes"]}
        zero_reference = portable["zero_shot_reference"] if portable_mode else read_json(source/"zero_shot_reference.json")
        all_results = []
        specifications = [{"experiment":"SigLIP_zero_shot","seed":None,"result":zero_reference}] + inventory
        (output/"predictions").mkdir(exist_ok=True)
        for item in specifications:
            name,seed = item["experiment"],item["seed"]
            key = name if seed is None else f"{name}-seed{seed}"
            result_path = output/"predictions"/(key+".json.gz")
            marker_path = output/"predictions"/(key+".complete.json")
            if marker_path.exists():
                marker = read_json(marker_path)
                if marker["request_hash"] != request_hash or marker["candidate_hash"] != digest(records):
                    raise ValueError("Completed prediction request changed")
                verified_file(result_path,marker["file"])
                with gzip.open(result_path,"rt",encoding="utf-8") as stream: result=json.load(stream)
                all_results.append(result)
                print("Reuse hard evaluation:",key,flush=True)
                continue
            model = None
            if seed is not None:
                model = build_model(config["experiments"][name]["frame_transformer"],config["model"]).to(device)
                checkpoint = torch.load(item["head"]/"best.pt",map_location="cpu",weights_only=True)
                model.load_state_dict(checkpoint["state_dict"],strict=True)
                model.requires_grad_(False)
                model.eval()
            print("Verify original scores and evaluate:",key,flush=True)
            checks = {}
            for label,rows in sets.items():
                original = evaluate(model,rows,bank,sentences,device,config["batch_size"],config["loss_temperature"])
                checks[label] = compare_predictions(original,item["result"][label],request["original_score_tolerance"],
                    diagnostic_path=output/"score_drift"/(key+"-"+label+".json"))
                check=checks[label]
                print(f"  {label}: max drift={check['maximum_score_difference']:.8f}, "
                      f"rank changes={check['rank_changes']}, top1 changes={check['top1_choice_changes']}",flush=True)
            predictions = score_pairs(model,records,ids,new_bank,new_sentences,device,args.batch_size)
            result = {"model":name,"seed":seed,"request_hash":request_hash,"candidate_hash":digest(records),
                      "reuse_checks":checks,"metrics":grouped_metrics(records,predictions),
                      "five_choice":five_choice_metrics(records,predictions),"predictions":predictions}
            write_gzip(result_path,result)
            atomic_json(marker_path,{"request_hash":request_hash,"candidate_hash":digest(records),"file":file_record(result_path)})
            all_results.append(result)
            del model
        summary = summarize(all_results)
        report = {"version":VERSION,"request_hash":request_hash,"result_status":"complete",
                  "full_validation_episodes":834,"eligible_episodes":len({r["episode_id"] for r in records}),
                  "evaluated_pairs":len(records),"models":summary,"blind_diagnostics":blind,
                  "paired_effects":paired_effects(records,all_results),
                  "five_choice_by_model_seed":[{"model":r["model"],"seed":r["seed"],"metrics":r["five_choice"]} for r in all_results],
                  "generation_audit":suite["audit"],"no_training":True,"no_checkpoint_selection":True,
                  "semantic_warning":request["semantic_warning"],"elapsed_seconds":time.perf_counter()-started,
                  "five_choice_note":"Only groups with four eligible distinct hard negatives; no easy fillers; pair and 5-choice scores are separate."}
        atomic_json(output/"comparison.json",report)
        csv_rows = []
        for model, groups in summary.items():
            for group,metrics in groups.items():
                csv_rows.append({"model":model,"challenge":group,**metrics})
        with (output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        artifacts = {p.relative_to(output).as_posix():file_record(p) for p in output.rglob("*") if p.is_file()
                     and p.name != "status.json" and p.suffix not in (".zip",".log",".part")
                     and "text_features" not in p.parts and "fresh_features" not in p.parts and "visual_review" not in p.parts
                     and (not portable_mode or not p.resolve().is_relative_to(fresh_dir.resolve()))}
        atomic_json(output/"status.json",{"status":"complete","request_hash":request_hash,"artifacts":artifacts})
        package=output/"hard_eval_results.zip"
        with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
            for name in list(artifacts)+["status.json"]: archive.write(output/name,name)
            for name in source_hashes():
                if "source/"+name not in artifacts: archive.write(Path(__file__).parent/name,"source/"+name)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print("Share:",package,flush=True)
        return report
    except BaseException as exc:
        atomic_json(output/"status.json",{"status":"failed","request_hash":request_hash,"error":str(exc)})
        raise


def summarize(results):
    grouped=defaultdict(list)
    for result in results: grouped[result["model"]].append(result)
    summary={}
    for model,runs in grouped.items():
        groups=set(runs[0]["metrics"])
        if any(set(r["metrics"])!=groups for r in runs):
            raise ValueError("Seeds have different challenge coverage")
        summary[model]={}
        for group in sorted(groups):
            metrics=[r["metrics"][group] for r in runs]
            if len({(m["pairs"],m["episodes"]) for m in metrics})!=1:
                raise ValueError("Seeds have different candidate coverage")
            values=[m["episode_macro_win_rate"] for m in metrics]
            summary[model][group]={"episodes":metrics[0]["episodes"],"pairs":metrics[0]["pairs"],
                "seeds":len(runs),"episode_macro_win_mean":statistics.mean(values),
                "episode_macro_win_std":statistics.stdev(values) if len(values)>1 else 0.,
                "pair_win_mean":statistics.mean(m["pair_win_rate"] for m in metrics),
                "mean_margin":statistics.mean(m["mean_margin"] for m in metrics),
                "mean_ties":statistics.mean(m["ties"] for m in metrics)}
    return summary


def paired_effects(records,results,bootstrap_samples=2000):
    """Episode-cluster intervals conditional on the fitted checkpoints, not tests."""
    types={r["id"]:(r["family"],r["family"]+"/"+r["type"]) for r in records}
    values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for result in results:
        for row in result["predictions"]:
            for group in types[row["id"]]:
                values[result["model"]][group][row["episode_id"]].append(row["win"])
    comparisons=[("E1","E0"),("E2","E0"),("E3","E1")]
    comparisons += [(name,"SigLIP_zero_shot") for name in ("E0","E1","E2","E3")]
    output={}
    for candidate,baseline in comparisons:
        if candidate not in values or baseline not in values: continue
        entry={}
        for group,by_episode in values[candidate].items():
            before=values[baseline][group]
            if set(by_episode)!=set(before): raise ValueError("Paired models have different episode coverage")
            ids=sorted(by_episode)
            delta=np.array([statistics.mean(by_episode[ep])-statistics.mean(before[ep]) for ep in ids])
            rng=np.random.default_rng(2032)
            boots=np.array([delta[rng.integers(0,len(delta),size=len(delta))].mean() for _ in range(bootstrap_samples)])
            entry[group]={"episodes":len(ids),"mean_delta":float(delta.mean()),
                          "ci95":[float(x) for x in np.quantile(boots,[.025,.975])],
                          "bootstrap_samples":bootstrap_samples,
                          "method":"paired episode bootstrap of seed-and-pair mean wins",
                          "limitations":"Conditional on fitted checkpoints; excludes seed uncertainty, selection bias, label noise and hidden-test shift."}
        output[candidate+"_minus_"+baseline]=entry
    return output


def main():
    parser=argparse.ArgumentParser()
    for name in ("contracts","train-zip","output"):
        parser.add_argument("--"+name,required=True)
    for name in ("feature-dir","sentence-feature-dir","team-validation-dir","factorial-run","weights-dir","portable-models","fresh-feature-dir","text-feature-dir"):
        parser.add_argument("--"+name)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--model-cache")
    parser.add_argument("--extra-candidates",help="Explicit optional qwen_candidates.json; never enabled implicitly")
    parser.add_argument("--seed",type=int,default=2031)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--text-batch-size",type=int,default=16)
    parser.add_argument("--original-score-tolerance",type=float,default=None,
                        help="Explicit fresh-feature diagnostic tolerance, capped at 0.002; default 0.001")
    args=parser.parse_args()
    if min(args.batch_size,args.text_batch_size)<1: parser.error("Batch sizes must be positive")
    if not args.weights_dir and not all((args.feature_dir,args.sentence_feature_dir,args.team_validation_dir,args.factorial_run)):
        parser.error("Use --weights-dir for a new Drive account, or provide all four legacy cache/run paths")
    run(args)


if __name__=="__main__": main()
