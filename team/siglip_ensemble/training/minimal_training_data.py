"""Training-only rule negatives and a lexical-slot replacement schedule.

Only provided training descriptions are transformed by the frozen v2 rules.
Rule-assumed negatives are not visually verified contradictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from hard_negative_data import _attribute_candidates, _minimal, _sentences, _tokens, _words
from sentence_features import sentence_plan
from siglip_model import make_text_plan

VERSION = "train-only-minimal-lexical-slot-v1"
BASE_MIX = {"random":2,"lexical":1,"similar_episode":1}


def _normal(text):
    return " ".join(text.split()).casefold()


def _sources():
    root=Path(__file__).resolve().parent
    return {name:hashlib.sha256((root/name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("minimal_training_data.py","hard_negative_data.py","sentence_features.py","siglip_model.py")}


def full_input_key(tokenizer,text):
    paragraph=make_text_plan(tokenizer,text)
    sentence=sentence_plan(tokenizer,text)
    return digest({"paragraph":paragraph["token_inputs"]["truncate64"],
                   "segments":[{k:s[k] for k in ("token_ids","position","weight")} for s in sentence["segments"]]})


def semantic_group(kind):
    if kind.startswith("action_"): return "action"
    if kind.startswith("state_"): return "state"
    if kind.startswith("spatial_"): return "spatial"
    return {"attribute_color":"color","attribute_shape":"shape","object_noun":"noun"}.get(kind,kind)


def prepare_train_minimal(manifest,split,tokenizer,actions,output,*,seed=2031,max_per_episode=4):
    """Return a content-addressed suite; validation content never enters rules.

    Unlike calling build_suite with swapped split names, both vocabulary and
    target descriptions here come exclusively from the original TRAIN IDs.
    The returned records keep the evaluation-suite field names so the existing
    frozen candidate text encoder can encode them without changes.
    """
    train=list(split["train_ids"]);val=list(split["val_ids"])
    rows={r["episode_id"]:r for r in manifest["episodes"]}
    if (not train or len(train)!=len(set(train)) or len(val)!=len(set(val)) or
        set(train)&set(val) or set(train)|set(val)!=set(rows) or len(rows)!=len(manifest["episodes"])):
        raise ValueError("Need complete disjoint original train/validation split")
    if set(actions)!=set(train):
        raise ValueError("Action labels must contain exactly training IDs")
    if type(max_per_episode) is not int or not 1<=max_per_episode<=32:
        raise ValueError("Minimal candidate cap must be between 1 and 32")
    descriptions={ep:rows[ep]["description"] for ep in train}
    vocabulary=Counter(w for ep in train for w in _words(descriptions[ep]))
    request={"version":VERSION,"train_ids":train,"train_descriptions_hash":digest(descriptions),
             "split_hash":digest(split),"action_labels_hash":digest(actions),"seed":seed,
             "max_per_episode":max_per_episode,"vocabulary_source":"original TRAIN positives only",
             "sources":_sources(),"teacher_filter":False}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    immutable_json(output/"request.json",request)
    keys={ep:full_input_key(tokenizer,text) for ep,text in descriptions.items()}
    rejects=Counter();records=[]
    for ep in train:
        text=descriptions[ep];rng=random.Random(int(digest([seed,ep])[:16],16))
        candidates=list(_minimal(text,actions[ep],rejects))+list(_attribute_candidates(text,vocabulary,rejects))
        buckets=defaultdict(list)
        for candidate in candidates: buckets[semantic_group(candidate["type"])].append(candidate)
        groups=sorted(buckets);rng.shuffle(groups)
        for bucket in buckets.values(): rng.shuffle(bucket)
        ordered=[]
        while any(buckets.values()):
            for group in groups:
                if buckets[group]: ordered.append(buckets[group].pop())
        selected=[];seen={_normal(text)};positive=_tokens(tokenizer,text)
        for candidate in ordered:
            negative=candidate["negative"]
            if _normal(negative) in seen: continue
            seen.add(_normal(negative))
            negative_key=full_input_key(tokenizer,negative)
            if negative_key==keys[ep]:
                rejects["identical_complete_model_input"]+=1;continue
            tokens=_tokens(tokenizer,negative)
            evidence={k:v for k,v in candidate.items() if k!="negative"}
            evidence.update(source_episode_id=ep,visual_truth_verified=False,
                            replacement_word_train_positive_count=vocabulary[candidate["new"].lower()])
            row={"id":"train_min_"+digest([VERSION,seed,ep,candidate["type"],negative])[:24],
                 "aug_id":"text_"+digest(negative),"episode_id":ep,"source_split":"train",
                 "family":"minimal","type":candidate["type"],"semantic_group":semantic_group(candidate["type"]),
                 "positive_text":text,"negative_text":negative,"semantic_status":"rule_assumed",
                 "evidence":evidence,"full_input_key":negative_key,"positive_full_input_key":keys[ep],
                 "positive_token_count":len(positive),"negative_token_count":len(tokens),
                 "positive_sentence_count":len(_sentences(text)),"negative_sentence_count":len(_sentences(negative)),
                 "base64_identical":positive[:63]==tokens[:63]}
            selected.append(row)
            if len(selected)>=max_per_episode:break
        records.extend(selected)
    by_episode={ep:[] for ep in train}
    for row in records: by_episode[row["episode_id"]].append(row)
    suite={"request":request,"records":records,"by_episode":by_episode,"full_input_keys":keys,
           "audit":{"train_episodes":len(train),"eligible_episodes":sum(bool(v) for v in by_episode.values()),
                    "pairs":len(records),"type_counts":dict(Counter(r["type"] for r in records)),
                    "semantic_group_counts":dict(Counter(r["semantic_group"] for r in records)),
                    "rejections":dict(rejects),"base64_identical_pairs":sum(r["base64_identical"] for r in records),
                    "validation_sources":0,"warning":"Code-generated rule assumptions; visual falsity is not verified."}}
    suite["fingerprint"]=digest(suite)
    immutable_json(output/"training_candidates.json",suite)
    return suite


class MinimalTrainingSchedules:
    """Use epoch('team',seed,epoch) wherever TrainingSchedules is accepted.

    The base epoch is the unchanged team sampler. Exactly its actual lexical
    slot is replaced, preserving order, positive slot and every retained choice.
    Unavailable/colliding replacements keep the entire base row unchanged.
    """
    def __init__(self,base_store,suite,output_dir):
        self.base_store=base_store;self.train_ids=list(base_store.train_ids)
        if dict(base_store.mix)!=BASE_MIX or list(base_store.mix)!=list(BASE_MIX):
            raise ValueError("Need ordered random=2,lexical=1,similar_episode=1 base mix")
        if suite.get("fingerprint")!=digest({k:v for k,v in suite.items() if k!="fingerprint"}):
            raise ValueError("Training augmentation fingerprint differs")
        if set(suite["by_episode"])!=set(self.train_ids) or set(suite["full_input_keys"])!=set(self.train_ids):
            raise ValueError("Augmentation coverage must be exactly training IDs")
        self.suite=suite;self.by_episode=suite["by_episode"];self.keys=suite["full_input_keys"]
        for ep,rows in self.by_episode.items():
            for row in rows:
                if (row["episode_id"]!=ep or row["source_split"]!="train" or
                    row["positive_text"]!=base_store.descriptions[ep] or
                    row["positive_full_input_key"]!=self.keys[ep] or row["full_input_key"]==self.keys[ep] or
                    row["aug_id"]!="text_"+digest(row["negative_text"])):
                    raise ValueError("Candidate is not a distinct transformation of its own training episode")
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True)
        self.metadata={"version":VERSION,"base_fingerprint":base_store.fingerprint,
                       "augmentation_fingerprint":suite["fingerprint"],"pool":"training_only",
                       "replace":"exactly the actual lexical slot; all other choices/order unchanged",
                       "sampling":"uniform available semantic group, then type, then candidate per video and epoch",
                       "fallback":"unchanged baseline row","sources":_sources()}
        self.fingerprint=digest(self.metadata)
        immutable_json(self.output_dir/"request.json",{**self.metadata,"fingerprint":self.fingerprint})

    def epoch(self,mode,seed,epoch):
        if mode!="team":raise ValueError("Minimal schedules require the mixed/team training mode")
        base,audit=self.base_store.epoch("team",seed,epoch)
        base_file=self.base_store.output_dir/audit["file"]["path"]
        verified_file(base_file,audit["file"])
        raw=json.loads(gzip.decompress(base_file.read_bytes()))
        if [r["episode_id"] for r in raw["records"]]!=[r[0] for r in base]:
            raise ValueError("Base schedule record order differs")
        result=[];details=[];reasons=Counter();groups_used=Counter();types_used=Counter()
        for (ep,choices),record in zip(base,raw["records"]):
            if ep not in self.keys or choices!=[ep]+[r["source"] for r in record["negative_sources"]]:
                raise ValueError("Base schedule choices differ from original training sources")
            slots=[i+1 for i,r in enumerate(record["negative_sources"]) if r["type"]=="lexical"]
            selected=None;slot=slots[0] if len(slots)==1 else None
            if slot is None:reasons["no_single_lexical_slot"]+=1
            else:
                retained=[source for i,source in enumerate(choices) if i!=slot]
                retained_keys={self.keys[source] for source in retained}
                retained_texts={_normal(self.base_store.descriptions[source]) for source in retained}
                candidates=[r for r in self.by_episode[ep] if r["full_input_key"] not in retained_keys
                            and _normal(r["negative_text"]) not in retained_texts]
                buckets=defaultdict(lambda:defaultdict(list))
                for row in candidates:buckets[row["semantic_group"]][row["type"]].append(row)
                if buckets:
                    rng=random.Random(int(digest([VERSION,seed,epoch,ep])[:16],16))
                    group=rng.choice(sorted(buckets));kind=rng.choice(sorted(buckets[group]))
                    selected=rng.choice(sorted(buckets[group][kind],key=lambda r:r["id"]))
                    groups_used[group]+=1;types_used[kind]+=1
                else:reasons["no_noncolliding_minimal_candidate"]+=1
            changed=list(choices)
            if selected:changed[slot]=selected["aug_id"]
            result.append((ep,changed))
            details.append({"episode_id":ep,"choices_before":choices,"choices_after":changed,
                            "replaced_choice_index":slot if selected else None,
                            "minimal_record_id":selected["id"] if selected else None,
                            "type":selected["type"] if selected else None})
        replaced=sum(groups_used.values())
        out_audit={"mode":"minimal","seed":seed,"epoch":epoch,"episodes":len(base),
                   "schedule_hash":digest(result),"base_schedule_hash":digest(base),
                   "replaced_episodes":replaced,"fallback_episodes":len(base)-replaced,
                   "fallback_reasons":dict(reasons),"semantic_groups_used":dict(groups_used),
                   "types_used":dict(types_used),"pool":"training_only","config_fingerprint":self.fingerprint}
        payload={"version":VERSION,"config_fingerprint":self.fingerprint,"records":details,
                 "audit":out_audit,"base_file":audit["file"]}
        path=self.output_dir/f"seed{seed}"/f"epoch-{epoch:03d}.json.gz";path.parent.mkdir(parents=True,exist_ok=True)
        marker=path.with_name(path.name+".json")
        if marker.exists():verified_file(path,read_json(marker)["file"])
        if path.exists():
            if json.loads(gzip.decompress(path.read_bytes()))!=payload:
                raise ValueError("Paired minimal schedule changed")
        else:
            tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
            tmp.write_bytes(gzip.compress(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
            os.replace(tmp,path)
        output_record={"path":path.relative_to(self.output_dir).as_posix(),**file_record(path)}
        immutable_json(marker,{"audit":out_audit,"file":output_record})
        return result,{**out_audit,"file":output_record}
