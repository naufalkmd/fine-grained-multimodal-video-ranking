"""Frozen original/train-negative feature caches for a fixed disjoint experiment.

All provided original episodes are encoded without gradients. The real split
is retained in each fingerprint; downstream schedules alone select training IDs.
No validation text is generated here or used as a training source.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from data_contract import load_contracts, read_manifest
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from prepare_features import FrameEncoder, check_arrays
from sentence_features import MODEL_REVISION, _encode_plans, _check_arrays, sentence_plan
from siglip_model import MODEL_ID
from siglip_model import make_text_plan

VERSION = 'all-original-fixed-split-features-v1'


def _json_form(value):
    # Normalize integer-key config maps and tuples BEFORE both hashing and IO.
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def _sources():
    root = Path(__file__).resolve().parent
    return {n:hashlib.sha256((root/n).read_text(encoding='utf-8').encode()).hexdigest()
            for n in ('training_features.py','prepare_features.py','siglip_model.py','sentence_features.py')}


def _check_shard(arrays, ids, expected_frame_counts, dimension):
    check_arrays(arrays,ids,dimension)
    if arrays['offsets'].dtype.kind not in 'iu' or not np.array_equal(np.diff(arrays['offsets']),expected_frame_counts):
        raise ValueError('Original frame count/order coverage changed')
    _check_arrays({'episode_ids':arrays['episode_ids'],'offsets':arrays['sentence_offsets'],
                   'vectors':arrays['sentence_vectors'],'positions':arrays['sentence_positions'],
                   'weights':arrays['sentence_weights']},ids,dimension)
    if arrays['full_token_counts'].dtype.kind not in 'iu' or np.any(arrays['full_token_counts']<1):
        raise ValueError('Invalid original full token counts')


def prepare_training_features(manifest, split, encoder, output_dir, *, train_zip, shard_size=32):
    """Encode every original supplied episode, retaining the actual split.

    No trainable backbone is instantiated. Resuming validates original ZIP,
    encoder revision, source hashes, complete frame counts and shard checksums.
    """
    if shard_size<1: raise ValueError('shard_size must be positive')
    ids=[r['episode_id'] for r in manifest['episodes']]
    train=set(split['train_ids']);valid=set(split['val_ids'])
    records={r['episode_id']:r for r in manifest['episodes']}
    if (not ids or len(ids)!=len(set(ids)) or train & valid or
        len(train)!=len(split['train_ids']) or len(valid)!=len(split['val_ids']) or
        train | valid != set(ids)):
        raise ValueError('Invalid disjoint complete train/validation split')
    if read_manifest(train_zip)!=manifest:
        raise ValueError('Source ZIP differs from the fixed original contract')
    model=_json_form(encoder.provenance)
    if model.get('model_id')!=MODEL_ID or model.get('snapshot_commit')!=MODEL_REVISION:
        raise ValueError('Fresh cache needs the pinned allowed SigLIP revision')
    model.update(image_aggregation='none; one unit vector per original frame, numeric order retained',
                 cached_image_representation='all individual unit frame features in numeric order',
                 cached_text_representation='truncate64 and full sentence/window unit features',
                 position_information='sequence order retained; no learned temporal encoder in cache')
    request={'version':VERSION,'manifest_hash':digest(manifest),'split_hash':digest(split),
             'episode_ids':ids,'source_fingerprint':manifest['source']['fingerprint'],
             'encoder':model,'feature_dim':encoder.feature_dim,'shard_size':shard_size,
             'sources':_sources(),'training_steps':0,'selection':'all provided original episodes; schedules enforce true train/validation split',
             'train_ids':list(split['train_ids']),'validation_ids':list(split['val_ids'])}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    request=_json_form(request)
    immutable_json(folder/'request.json',request)
    fingerprint=digest(request)
    started=time.perf_counter();reused=0;shards=[]
    previous=read_json(folder/'status.json') if (folder/'status.json').exists() else None
    if previous and previous.get('status')=='complete':
        if previous.get('request_hash')!=fingerprint: raise ValueError('Fresh cache request differs')
        verified_file(folder/'index.json',previous['index'])
    else:
        atomic_json(folder/'status.json',{'status':'running','request_hash':fingerprint})
    try:
        with zipfile.ZipFile(train_zip) as archive:
            for start in range(0,len(ids),shard_size):
                batch_ids=ids[start:start+shard_size];rows=[records[ep] for ep in batch_ids]
                frame_counts=[len(r['image_members']) for r in rows]
                filename=f'shard-{start:05d}.npz';path=folder/filename;marker=folder/(filename+'.json')
                expected={'request_hash':fingerprint,'episode_ids':batch_ids,'frame_counts':frame_counts}
                if marker.exists():
                    meta=read_json(marker)
                    if any(meta.get(k)!=v for k,v in expected.items()): raise ValueError('Fresh cache shard identity differs')
                    verified_file(path,meta['file'])
                    with np.load(path,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
                    _check_shard(arrays,batch_ids,frame_counts,encoder.feature_dim)
                    reused+=len(batch_ids)
                else:
                    frames=[FrameEncoder.frames(encoder,archive,r['image_members']) for r in rows]
                    if any(len(values)!=count for values,count in zip(frames,frame_counts)):
                        raise ValueError('Encoder omitted original frames')
                    base,base_plans=encoder.text_features([{'id':ep,'text':r['description']} for ep,r in zip(batch_ids,rows)])
                    plans=[sentence_plan(encoder.tokenizer,r['description']) for r in rows]
                    vectors=_encode_plans(encoder,plans)
                    arrays={'episode_ids':np.asarray(batch_ids),'frames':np.concatenate(frames).astype(np.float32),
                        'offsets':np.array([0]+list(np.cumsum(frame_counts)),dtype=np.int64),
                        'texts':np.asarray(base['truncate64'],dtype=np.float32),
                        'token_keys':np.asarray([digest(p['token_inputs']['truncate64']) for p in base_plans]),
                        'full_token_counts':np.asarray([p['full_token_count'] for p in base_plans],dtype=np.int64),
                        'sentence_vectors':np.concatenate(vectors).astype(np.float32),
                        'sentence_offsets':np.array([0]+list(np.cumsum([len(v) for v in vectors])),dtype=np.int64),
                        'sentence_positions':np.array([s['position'] for p in plans for s in p['segments']],dtype=np.float32),
                        'sentence_weights':np.array([s['weight'] for p in plans for s in p['segments']],dtype=np.int64)}
                    _check_shard(arrays,batch_ids,frame_counts,encoder.feature_dim)
                    temporary=path.with_suffix('.npz.part')
                    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
                    temporary.replace(path)
                    meta={**expected,'file':file_record(path)};atomic_json(marker,meta)
                shards.append({'name':filename,**meta})
                print(f'Original frame/text cache {start+len(batch_ids)}/{len(ids)}; reused={reused}',flush=True)
        index={'version':VERSION,'request_hash':fingerprint,'fingerprint':fingerprint,
               'source_fingerprint':manifest['source']['fingerprint'],'episode_ids':ids,
               'feature_dim':encoder.feature_dim,'model':model,'text_mode':'truncate64','shards':shards,
               'validation_only':False,'training_steps':0,'all_original_frames':True,
               'train_ids':list(split['train_ids']),'validation_ids':list(split['val_ids']),
               'split_hash':digest(split),
               'timing':{'elapsed_seconds':time.perf_counter()-started,'reused_episodes':reused}}
        if previous and previous.get('status')=='complete':
            old=read_json(folder/'index.json')
            if {k:v for k,v in old.items() if k!='timing'}!={k:v for k,v in index.items() if k!='timing'}:
                raise ValueError('Completed fresh-cache index differs')
            index=old  # Stable artifact hashes across completed reruns.
        else:
            atomic_json(folder/'index.json',index)
            atomic_json(folder/'status.json',{'status':'complete','request_hash':fingerprint,'index':file_record(folder/'index.json')})
        bank,sentence_bank=_load_banks(folder,index)
        bank['original_text_ids']={r['description']:r['episode_id'] for r in manifest['episodes']}
        return bank,sentence_bank
    except BaseException as exc:
        atomic_json(folder/'status.json',{'status':'failed','request_hash':fingerprint,'error':str(exc)})
        raise


def _load_banks(folder,index):
    bank={'frames':{},'texts':{},'token_keys':{},'full_token_counts':{},'index':index}
    sentence={'features':{},'positions':{},'weights':{},
              'index':{'fingerprint':digest({'features':index['fingerprint'],'representation':'all ordered sentence windows'}),
                       'episode_ids':index['episode_ids'],'model':index['model'],'validation_only':False,
                       'train_ids':index['train_ids'],'validation_ids':index['validation_ids'],
                       'split_hash':index['split_hash']}}
    for meta in index['shards']:
        path=verified_file(Path(folder)/meta['name'],meta['file'])
        with np.load(path,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
        _check_shard(arrays,meta['episode_ids'],meta['frame_counts'],index['feature_dim'])
        for i,ep in enumerate(meta['episode_ids']):
            a,b=map(int,arrays['offsets'][i:i+2]);c,d=map(int,arrays['sentence_offsets'][i:i+2])
            bank['frames'][ep]=torch.from_numpy(arrays['frames'][a:b].copy())
            bank['texts'][ep]=torch.from_numpy(arrays['texts'][i].copy())
            bank['token_keys'][ep]=str(arrays['token_keys'][i]);bank['full_token_counts'][ep]=int(arrays['full_token_counts'][i])
            sentence['features'][ep]=torch.from_numpy(arrays['sentence_vectors'][c:d].copy())
            sentence['positions'][ep]=torch.from_numpy(arrays['sentence_positions'][c:d].copy())
            sentence['weights'][ep]=torch.from_numpy(arrays['sentence_weights'][c:d].copy())
    if list(bank['frames'])!=index['episode_ids']:raise ValueError('Fresh cache coverage/order changed')
    return bank,sentence


def _generated_arrays(arrays, keys, counts, dimension):
    if arrays['keys'].tolist()!=keys:
        raise ValueError('Generated text shard IDs differ')
    offsets=arrays['offsets']
    if offsets.dtype.kind not in 'iu' or offsets.shape!=(len(keys)+1,) or offsets[0]!=0 or not np.array_equal(np.diff(offsets),counts):
        raise ValueError('Generated sentence coverage/order differs')
    if arrays['base'].shape!=(len(keys),dimension):
        raise ValueError('Generated base feature dimensions differ')
    for values in (arrays['base'],arrays['sentences']):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=2e-4):
            raise ValueError('Generated features must be finite unit vectors')
    _check_arrays({'episode_ids':arrays['keys'],'offsets':offsets,'vectors':arrays['sentences'],
                   'positions':arrays['positions'],'weights':arrays['weights']},keys,dimension)


def prepare_generated_texts(records, bank, sentence_bank, encoder, output_dir, *, shard_size=64):
    """Encode supplied candidate text, extending mappings without mutating originals.

    ``records`` may be strings or dicts with positive_text/negative_text.
    Caller generates train-only augmentation and supplies only that data here.
    All candidate IDs are ``text_`` + digest(text), also available in metadata.
    The returned banks keep their original index (the parent dataset identity);
    metadata fingerprints separately bind generated text and cache provenance.
    """
    if shard_size<1: raise ValueError('shard_size must be positive')
    texts=set()
    for row in records:
        if isinstance(row,str):texts.add(row)
        else:
            for key in ('positive_text','negative_text'):
                if key in row:texts.add(row[key])
    if any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError('Generated texts must be nonempty strings')
    texts=sorted(texts);keys={text:'text_'+digest(text) for text in texts}
    plans={text:sentence_plan(encoder.tokenizer,text) for text in texts}
    bases={text:make_text_plan(encoder.tokenizer,text) for text in texts}
    request=_json_form({'version':'minimal-generated-text-features-v1','texts_hash':digest(texts),
             'original_features':bank['index']['fingerprint'],
             'original_sentence_features':sentence_bank['index']['fingerprint'],
             'encoder':encoder.provenance,'dimension':encoder.feature_dim,'shard_size':shard_size,
             'sources':_sources(),'token_input_hash':digest({text:{'base':bases[text]['token_inputs']['truncate64'],
                 'segments':[{k:s[k] for k in ('token_ids','position','weight')} for s in plans[text]['segments']]}
                 for text in texts})})
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    immutable_json(folder/'request.json',request);fingerprint=digest(request)
    # Original text may be represented by several episode IDs. Model-token
    # equality, rather than string heuristics, decides whether features can alias.
    original_lookup={value:ep for ep,value in bank['token_keys'].items()}
    new_bank={**bank,**{name:dict(bank[name]) for name in ('texts','token_keys','full_token_counts')}}
    new_sentences={**sentence_bank,**{name:dict(sentence_bank[name]) for name in ('features','positions','weights')}}
    original_text_ids=bank.get('original_text_ids',{})
    for text in texts:
        original=original_text_ids.get(text)
        if original is not None:
            key=keys[text]
            for name in ('texts','token_keys','full_token_counts'):
                new_bank[name][key]=bank[name][original]
            for name in ('features','positions','weights'):
                new_sentences[name][key]=sentence_bank[name][original]
    to_encode=[text for text in texts if text not in original_text_ids]
    shards=[];completed=0
    try:
        for start in range(0,len(to_encode),shard_size):
            current=to_encode[start:start+shard_size];current_keys=[keys[t] for t in current]
            current_plans=[plans[t] for t in current]
            counts=[len(p['segments']) for p in current_plans]
            name=f'shard-{start:05d}.npz';target=folder/name;marker=folder/(name+'.json')
            expected={'request_hash':fingerprint,'keys':current_keys,'segment_counts':counts}
            if marker.exists():
                meta=read_json(marker)
                if any(meta.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Generated cache shard configuration differs')
                verified_file(target,meta['file'])
                with np.load(target,allow_pickle=False) as values:arrays={k:values[k] for k in values.files}
            else:
                encoded,_=encoder.text_features([{'id':keys[t],'text':t} for t in current])
                sentences=_encode_plans(encoder,current_plans)
                arrays={'keys':np.asarray(current_keys),'base':np.asarray(encoded['truncate64'],dtype=np.float32),
                        'sentences':np.concatenate(sentences).astype(np.float32),
                        'offsets':np.array([0]+list(np.cumsum(counts)),dtype=np.int64),
                        'positions':np.array([s['position'] for p in current_plans for s in p['segments']],dtype=np.float32),
                        'weights':np.array([s['weight'] for p in current_plans for s in p['segments']],dtype=np.int64)}
                _generated_arrays(arrays,current_keys,counts,encoder.feature_dim)
                temp=target.with_suffix('.npz.part')
                with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
                temp.replace(target);meta={**expected,'file':file_record(target)}
                atomic_json(marker,meta)
            _generated_arrays(arrays,current_keys,counts,encoder.feature_dim)
            for i,text in enumerate(current):
                key=keys[text];base_key=digest(bases[text]['token_inputs']['truncate64'])
                original_ep=original_lookup.get(base_key)
                # Reuse original base vectors exactly for truncated-token
                # collisions; never let FP16 batch differences invent a signal.
                new_bank['texts'][key]=(bank['texts'][original_ep] if original_ep is not None
                                        else torch.from_numpy(arrays['base'][i].copy()))
                new_bank['token_keys'][key]=base_key
                new_bank['full_token_counts'][key]=bases[text]['full_token_count']
                a,b=map(int,arrays['offsets'][i:i+2])
                new_sentences['features'][key]=torch.from_numpy(arrays['sentences'][a:b].copy())
                new_sentences['positions'][key]=torch.from_numpy(arrays['positions'][a:b].copy())
                new_sentences['weights'][key]=torch.from_numpy(arrays['weights'][a:b].copy())
            shards.append({'name':name,**meta});completed+=len(current)
            print(f'New candidate text features {completed}/{len(to_encode)}; original texts reused={len(texts)-len(to_encode)}',flush=True)
        metadata={'request':request,'fingerprint':fingerprint,'text_ids':keys,'texts':len(texts),'shards':shards,
                  'original_texts_reused':len(texts)-len(to_encode),'new_texts_encoded':len(to_encode),
                  'training_steps':0,'all_sentence_tokens_retained':True,'parent_split_hash':bank['index'].get('split_hash')}
        immutable_json(folder/'index.json',metadata)
        atomic_json(folder/'status.json',{'status':'complete','request_hash':fingerprint,'index':file_record(folder/'index.json')})
        return new_bank,new_sentences,metadata
    except BaseException as exc:
        atomic_json(folder/'status.json',{'status':'failed','request_hash':fingerprint,'error':str(exc)})
        raise

