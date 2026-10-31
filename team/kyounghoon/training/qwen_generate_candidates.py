"""True append diagnostics: official omitted action vs provisional absent action.

Qwen verbalizes predetermined annotation labels; it never judges truth or scores
choices. The original caption is byte-for-byte preserved in both candidates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import zipfile
from data_contract import load_contracts
from experiment_io import atomic_json, digest, immutable_json, read_json, file_record, verified_file

VERSION = 'qwen-official-action-true-append-v2'
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
SIGLIP_ID = 'google/siglip-base-patch16-224'
SIGLIP_REVISION = '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'
# Transformation predicates and official action labels, never authored captions.
ACTION_FAMILIES = {
 'clap': {'label':'Clapping hands','aliases':r'\bclap\w*\b','verbs':['claps','clapping'],'nouns':['hands']},
 'fold': {'label':'Folding something','aliases':r'\bfold\w*\b','verbs':['folds','folding'],'nouns':['something']},
 'unfold': {'label':'Unfolding something','aliases':r'\bunfold\w*\b','verbs':['unfolds','unfolding'],'nouns':['something']},
 'cover': {'label':'Covering something with something','aliases':r'\bcover\w*\b|\bconceal\w*\b|\bhidden\b|\bhid\w*\b','verbs':['covers','covering'],'nouns':['something']},
 'tie': {'label':'Tying something','aliases':r'\b(?:tie|ties|tied|tying|knot\w*)\b','verbs':['ties','tying'],'nouns':['something']},
 'tear': {'label':'Tearing something into pieces','aliases':r'\b(?:tear\w*|tore|torn|rip\w*)\b','verbs':['tears','tearing'],'nouns':['something','pieces']},
 'crumple': {'label':'Crumpling something','aliases':r'\bcrumpl\w*\b|\bcrush\w*\b|\bscrunch\w*\b','verbs':['crumples','crumpling'],'nouns':['something']},
 'pour': {'label':'Pouring something from something into something','aliases':r'\bpour\w*\b|\bstream\w*\b|\btransfer\w*\b|\btilt\w*\b','verbs':['pours','pouring'],'nouns':['something']},
 'drink': {'label':'Drinking something from something','aliases':r'\b(?:drink\w*|drank|drunk|sip\w*|mouth|lip\w*)\b','verbs':['drinks','drinking'],'nouns':['something']},
 'button': {'label':'Buttoning up a button','aliases':r'\bbutton\w*\b|\bfasten\w*\b','verbs':['buttons','buttoning'],'nouns':['button']},
 'iron': {'label':'Ironing something','aliases':r'\biron\w*\b|\bpress\w*\b','verbs':['irons','ironing'],'nouns':['something']},
 'cut': {'label':'Cutting something with something','aliases':r'\b(?:cut\w*|slice\w*|chop\w*|scissors)\b','verbs':['cuts','cutting'],'nouns':['something']},
 'snap': {'label':'Snapping fingers','aliases':r'\bsnap\w*\b','verbs':['snaps','snapping'],'nouns':['fingers']},
 'juggle': {'label':'Juggling something','aliases':r'\bjuggl\w*\b|\btoss\w*\b','verbs':['juggles','juggling'],'nouns':['something']},
 'sit_down': {'label':None,'aliases':r'\b(?:sit\w*|sat|seated|seat\w*)\b','verbs':['sits','sitting'],'nouns':[],'operator':'sit down'},
 'stand_up': {'label':None,'aliases':r'\b(?:stand\w*|stood|standing|rise|rises|rose)\b','verbs':['stands','standing'],'nouns':[],'operator':'stand up'},
}
CORE = tuple(k for k,v in ACTION_FAMILIES.items() if v['label'])
NOVEL = ('sit_down','stand_up')
FUNCTION = set('the person is a an something with from into on of up down in this clip shown here visibly can be seen performing action'.split())
# Do not infer temporal order, repeated occurrence, negation, counts or modifiers.
FORBIDDEN = re.compile(r'\b(?:then|after|before|again|another|once|twice|never|not|no|without|pretend\w*|try|tries|trying|attempt\w*|finally|next|subsequently|later|earlier)\b',re.I)
WORD = re.compile(r'[A-Za-z]+')

def words(text): return [w.lower() for w in WORD.findall(text)]
def token_ids(tokenizer,text): return list(tokenizer.encode(text,add_special_tokens=False,truncation=False))
def mentions(text,family): return bool(re.search(ACTION_FAMILIES[family]['aliases'],text,re.I))
def clear_negative(family,original,labels):
    return not mentions(original,family) and not any(mentions(label,family) for label in labels)

def load_labels(train_zip,manifest):
    result={}
    with zipfile.ZipFile(train_zip) as archive:
        for row in manifest['episodes']:
            member=str(Path(row['image_members'][0]).parent.parent / 'episode.json').replace('\\','/')
            raw=archive.read(member)
            if hashlib.sha256(raw).hexdigest()!=row['metadata_sha256']:
                raise ValueError('Action metadata differs from manifest: '+row['episode_id'])
            result[row['episode_id']]=[a['label'] for a in json.loads(raw)['actions']]
    return result

def omitted_families(original,labels):
    return [family for family in CORE if ACTION_FAMILIES[family]['label'] in labels and not mentions(original,family)]

def build_plans(manifest,split,labels,seed=2031,max_episodes=100,novel_limit=10):
    if not 2<=max_episodes<=len(split['val_ids']): raise ValueError('max-episodes must be between 2 and validation size')
    originals={r['episode_id']:r['description'] for r in manifest['episodes']}
    order=sorted(split['val_ids'],key=lambda ep:digest([seed,ep]))
    eligible={ep:omitted_families(originals[ep],labels[ep]) for ep in order}
    used=set();plans=[];group=0
    for ep in order:
        if ep in used or len(used)+2>max_episodes:continue
        found=None
        for pos in sorted(eligible[ep],key=lambda f:digest([seed,ep,f])):
            for peer in order:
                if peer==ep or peer in used or not clear_negative(pos,originals[peer],labels[peer]):continue
                for neg in sorted(eligible[peer],key=lambda f:digest([seed,peer,f])):
                    if neg!=pos and clear_negative(neg,originals[ep],labels[ep]):
                        found=(pos,peer,neg);break
                if found:break
            if found:break
        if not found:continue
        pos,peer,neg=found;group+=1;used.update([ep,peer])
        for target,p,n in [(ep,pos,neg),(peer,neg,pos)]:
            plans.append({'episode_id':target,'positive_action_family':p,'negative_action_family':n,
                          'reciprocal_group':group,'novel_action':False})
    # Explicit novel-action auxiliary; never pretend it has balanced positives.
    aux=0
    for plan in list(plans):
        if aux>=novel_limit:break
        ep=plan['episode_id']
        for family in NOVEL:
            if clear_negative(family,originals[ep],labels[ep]):
                plans.append({**plan,'negative_action_family':family,'reciprocal_group':None,'novel_action':True})
                aux+=1;break
    return plans,{'parent_validation_episodes':len(order),'eligible_omitted_action_episodes':sum(bool(x) for x in eligible.values()),
                  'planned_unique_episodes':len(used),'planned_core_pairs':len(plans)-aux,'planned_novel_pairs':aux,
                  'reciprocal_groups':group,'candidate_label_counts':dict(Counter(f for fs in eligible.values() for f in fs)),
                  'unpaired_or_ineligible_episodes':len(order)-len(used)}

def suffix_check(suffix,family):
    spec=ACTION_FAMILIES[family]
    if not isinstance(suffix,str) or suffix!=suffix.strip() or not suffix.startswith('The person '):
        raise ValueError('Suffix must use the fixed actor and have no surrounding whitespace')
    if not suffix.endswith('.') or len(re.findall(r'[.!?]',suffix))!=1 or any(c in suffix for c in '\n\r;:'):
        raise ValueError('Suffix must be exactly one sentence')
    if FORBIDDEN.search(suffix):raise ValueError('Unsupported temporal/negative/attempt modifier')
    tokens=words(suffix)
    if not 4<=len(tokens)<=24:raise ValueError('Suffix word count outside limits')
    allowed=FUNCTION|set(spec['verbs'])|set(spec['nouns'])
    if set(tokens)-allowed:raise ValueError('Unsupported object, attribute, or predicate words')
    if sum(tokens.count(v) for v in spec['verbs'])!=1:raise ValueError('Exactly one requested action predicate is required')
    if any(n not in tokens for n in spec['nouns']):raise ValueError('Required annotation object omitted')
    if family=='sit_down' and (not re.search(r'\b(?:sits|sitting) down\b',suffix) or 'something' in tokens):raise ValueError('sit down predicate/object mismatch')
    if family=='stand_up' and (not re.search(r'\b(?:stands|standing) up\b',suffix) or 'something' in tokens):raise ValueError('stand up predicate/object mismatch')
    for other in ACTION_FAMILIES:
        if other!=family and any(v in tokens for v in ACTION_FAMILIES[other]['verbs']):raise ValueError('Unexpected additional action')
    return len(tokens)

def validate_append_record(row,tokenizer,official_labels):
    if row.get('family')!='qwen_true_append' or row.get('semantic_status')!='unverified_event_absence':
        raise ValueError('Unexpected family or unverified status')
    p,n=row['positive_action_family'],row['negative_action_family']
    if p not in CORE or n not in ACTION_FAMILIES or p==n:raise ValueError('Invalid action family pair')
    original=row['original_text'];ev=row['evidence']
    if (ev.get('positive_action_label')!=ACTION_FAMILIES[p]['label'] or
        ev.get('positive_action_source')!='official_annotation' or ACTION_FAMILIES[p]['label'] not in official_labels):
        raise ValueError('Positive append lacks the specified official annotation')
    if mentions(original,p):raise ValueError('Positive append duplicates an explicitly described action family')
    if not clear_negative(n,original,official_labels):raise ValueError('Negative action may already be annotated/described, including pretending')
    for side in ('positive','negative'):
        if row[side+'_text']!=original+' '+row[side+'_suffix']:raise ValueError('Original caption was modified or replaced')
    wp=suffix_check(row['positive_suffix'],p);wn=suffix_check(row['negative_suffix'],n)
    if wp!=wn:raise ValueError('Suffix word counts differ')
    pi,ni=token_ids(tokenizer,row['positive_text']),token_ids(tokenizer,row['negative_text'])
    if len(pi)!=len(ni):raise ValueError('Full SigLIP token counts differ')
    if pi==ni:raise ValueError('Identical token inputs')
    if bool(ev.get('novel_action'))!=(n in NOVEL):raise ValueError('Novel action flag mismatch')
    return {'positive_token_count':len(pi),'negative_token_count':len(ni),'suffix_word_count':wp,
            'positive_sentence_count':len(re.findall(r'[.!?](?:\s|$)',original))+1,
            'negative_sentence_count':len(re.findall(r'[.!?](?:\s|$)',original))+1,
            'positive_character_count':len(row['positive_text']),'negative_character_count':len(row['negative_text']),
            'base64_identical':pi[:63]==ni[:63], 'original_prefix_unchanged':True,
            'word_and_token_and_sentence_counts_equal':True}

def prompt_for(original,plan,attempt=0,rejection=None):
    p,n=plan['positive_action_family'],plan['negative_action_family']
    system=('You verbalize requested action predicates for a text augmentation program. '
      'Do not judge a video, answer questions, output truth labels, explanations, scores, or rankings. '
      'Return JSON only with exactly two string keys positive_suffix and negative_suffix. '
      'Each suffix is one natural standalone sentence beginning with the exact actor prefix "The person ". '
      'The suffixes will be appended to the complete unchanged original caption by code. '
      'Describe only the requested action, in present tense. Use generic something where the official label is generic; '
      'never invent object names, attributes, locations, counts, or purpose. Do not assert then, again, earlier, or later. '
      'Use exactly the same number of words in both suffixes and aim for the same token length. '
      'Use one action verb per suffix. Only use the permitted vocabulary below. '
      'Permitted shared words: '+', '.join(sorted(FUNCTION))+'.')
    data={'original_caption':original,'requested_positive_label':ACTION_FAMILIES[p]['label'],
          'requested_negative_predicate':ACTION_FAMILIES[n]['label'] or ACTION_FAMILIES[n]['operator'],
          'positive_action_words':ACTION_FAMILIES[p]['verbs'],'positive_object_words':ACTION_FAMILIES[p]['nouns'],
          'negative_action_words':ACTION_FAMILIES[n]['verbs'],'negative_object_words':ACTION_FAMILIES[n]['nouns'],
          'attempt':attempt,'previous_mechanical_rejection':rejection}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(data,ensure_ascii=False)}]

def candidate_record(raw,original,plan,labels,tokenizer,request_hash,job_path,metadata_sha):
    try:
        data=json.loads(raw)
        if not isinstance(data,dict) or set(data)!= {'positive_suffix','negative_suffix'}:raise ValueError('JSON suffix schema required')
        if not all(isinstance(v,str) for v in data.values()):raise ValueError('Suffix values must be strings')
        p,n=plan['positive_action_family'],plan['negative_action_family']
        row={'id':'append_'+digest([request_hash,plan,data])[:24],'episode_id':plan['episode_id'],
             'family':'qwen_true_append','type':p+'__'+n,'semantic_status':'unverified_event_absence',
             'original_text':original,'positive_suffix':data['positive_suffix'],'negative_suffix':data['negative_suffix'],
             'positive_text':original+' '+data['positive_suffix'],'negative_text':original+' '+data['negative_suffix'],
             'positive_action_family':p,'negative_action_family':n,
             'evidence':{'positive_action_label':ACTION_FAMILIES[p]['label'],'positive_action_source':'official_annotation',
                         'source_episode_id':plan['episode_id'],'metadata_sha256':metadata_sha,'official_labels':labels,
                         'negative_absence_screen':'no family in original caption or any action annotation; NOT proof of absence',
                         'novel_action':plan['novel_action'],'reciprocal_group':plan['reciprocal_group'],
                         'job_path':job_path,'visual_truth_verified':False,
                         'limitations':'Omission screen is lexical; original caption may imply the positive action. Negative visual absence is unverified.'}}
        row.update(validate_append_record(row,tokenizer,labels))
        return row,None
    except (ValueError,KeyError,TypeError) as exc:return None,str(exc)

def run(args):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,PreTrainedTokenizerFast
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory>17*2**30:
        raise RuntimeError('Use one free Colab/Kaggle T4 GPU; no personal GPU or fallback.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest,split,_=load_contracts(args.contracts);labels=load_labels(args.train_zip,manifest)
    plans,planning=build_plans(manifest,split,labels,args.seed,args.max_episodes,args.novel_limit)
    if not plans:raise ValueError('No reciprocal omitted-action plans; inspect official action annotations.')
    originals={r['episode_id']:r for r in manifest['episodes']}
    old=read_json(output/'request.json') if (output/'request.json').is_file() else None
    revision=old['resolved_revision'] if old else HfApi().model_info(MODEL_ID,revision=args.revision or 'main').sha
    if not re.fullmatch(r'[0-9a-f]{40}',revision or ''):raise ValueError('Need an immutable model revision')
    if old and args.revision and args.revision not in {old['requested_revision'],revision}:raise ValueError('Generator revision changed')
    request={'version':VERSION,'model_id':MODEL_ID,'resolved_revision':revision,
       'requested_revision':old['requested_revision'] if old else (args.revision or 'main'),
       'manifest_hash':digest(manifest),'split_hash':digest(split),'labels_hash':digest(labels),'plans':plans,
       'seed':args.seed,'max_episodes':args.max_episodes,'novel_limit':args.novel_limit,'max_attempts':3,
       'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes')},
       'sources':{n:hashlib.sha256((Path(__file__).parent/n).read_text(encoding='utf8').encode()).hexdigest() for n in ('qwen_generate_candidates.py','hard_negative_data.py')},
       'quantization':'NF4 4bit double quant, FP16, batch1','max_input_tokens':2048,'max_new_tokens':160,
       'generation':'do_sample=False','purpose':'No teacher truth classification; predetermined annotation-grounded text generation.'}
    immutable_json(output/'request.json',request);request_hash=digest(request)
    if (output/'status.json').is_file():
        status=read_json(output/'status.json')
        if status.get('status')=='complete' and status.get('request_hash')==request_hash:
            verified_file(output/'qwen_candidates.json',status['candidate_file'])
            saved=read_json(output/'qwen_candidates.json')
            for j in saved['audit']['jobs']:verified_file(output/j['path'],j['file'])
            print('Reuse completed true-append candidates',flush=True);return
    sigtok=PreTrainedTokenizerFast.from_pretrained(SIGLIP_ID,revision=SIGLIP_REVISION)
    model=tokenizer=None;records=[];jobs=[];rejections=Counter();started=time.perf_counter()
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    atomic_json(output/'status.json',{'status':'running','request_hash':request_hash})
    try:
        for i,plan in enumerate(plans):
            ep=plan['episode_id'];original=originals[ep]['description'];error=None
            for attempt in range(3):
                messages=prompt_for(original,plan,attempt,error)
                relative=f"jobs/{ep}-{plan['positive_action_family']}-{plan['negative_action_family']}-{attempt}.json"
                path=output/relative;identity={'request_hash':request_hash,'plan':plan,'attempt':attempt,'prompt_hash':digest(messages)}
                if path.is_file():
                    job=read_json(path)
                    if any(job.get(k)!=v for k,v in identity.items()) or job.get('prompt')!=messages or job.get('raw_output_hash')!=digest(job.get('raw_output')):raise ValueError('Generation cache differs')
                else:
                    if model is None:
                        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision)
                        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
                        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=revision,quantization_config=quant,device_map={'':0},dtype=torch.float16,attn_implementation='eager').eval()
                        model.requires_grad_(False)
                        if getattr(model.config,'_commit_hash',None)!=revision:raise ValueError('Model revision mismatch')
                    begin=time.perf_counter();text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    batch=tokenizer(text,return_tensors='pt',add_special_tokens=False);size=batch['input_ids'].shape[1]
                    if size>2048:raw='';generation_error='input exceeds 2048 tokens'
                    else:
                        with torch.inference_mode():generated=model.generate(**{k:v.to('cuda') for k,v in batch.items()},max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,output_scores=False,return_dict_in_generate=False)
                        raw=tokenizer.decode(generated[0,size:],skip_special_tokens=True);generation_error=None
                    job={**identity,'prompt':messages,'raw_output':raw,'raw_output_hash':digest(raw),'seconds':time.perf_counter()-begin,'input_tokens':size,'generation_error':generation_error}
                    atomic_json(path,job)
                row,error=candidate_record(job['raw_output'],original,plan,labels[ep],sigtok,request_hash,relative,originals[ep]['metadata_sha256'])
                error=job.get('generation_error') or error
                jobs.append({'path':relative,'file':file_record(path),'accepted':not bool(error),'rejection':error})
                if not error:records.append(row);break
                rejections[error]+=1
            print(f'True append {i+1}/{len(plans)}; accepted={len(records)}',flush=True)
        audit={**planning,'accepted_pairs':len(records),'accepted_unique_episodes':len({r['episode_id'] for r in records}),
          'accepted_core_pairs':sum(not r['evidence']['novel_action'] for r in records),'accepted_novel_pairs':sum(r['evidence']['novel_action'] for r in records),
          'jobs':jobs,'rejections':dict(rejections),'elapsed_seconds_this_run':time.perf_counter()-started,
          'generation_seconds_all_jobs':sum(read_json(output/j['path'])['seconds'] for j in jobs),
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
          'gpu':torch.cuda.get_device_name(0),'semantic_warning':'Positive append supported by official action annotation; negative absence NOT visually verified. Lexical omission may miss paraphrases.',
          'balance_warning':'Generation failures can break planned reciprocal balance; evaluation must rebalance successful pairs. Novel action pairs are auxiliary only.',
          'time_warning':'Pilot generation before scaling; total reproduction including training must fit 24h on one T4.'}
        atomic_json(output/'qwen_candidates.json',{'request':request,'records':records,'audit':audit})
        atomic_json(output/'provenance.json',{'request_hash':request_hash,'request':request,'audit':audit})
        atomic_json(output/'status.json',{'status':'complete','request_hash':request_hash,'candidate_file':file_record(output/'qwen_candidates.json')})
    except BaseException as exc:
        atomic_json(output/'status.json',{'status':'failed','request_hash':request_hash,'error':str(exc)});raise

def main():
    p=argparse.ArgumentParser()
    for name in ('contracts','train-zip','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-episodes',type=int,default=100);p.add_argument('--seed',type=int,default=2031)
    p.add_argument('--novel-limit',type=int,default=10);p.add_argument('--revision',default='')
    args=p.parse_args()
    if not 0<=args.novel_limit<=10:p.error('novel-limit must be 0..10')
    run(args)
if __name__=='__main__':main()
