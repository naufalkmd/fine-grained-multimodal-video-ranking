"""Fresh, validation-only frozen SigLIP caches; no previous account artifacts.

Every supplied frame is preserved. Only validation episodes are encoded. New
candidate text is handled by the evaluation runner, after this original bank.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

VERSION = 'fresh-validation-frozen-features-v1'


def _sources():
    root = Path(__file__).resolve().parent
    return {n:hashlib.sha256((root/n).read_text(encoding='utf-8').encode()).hexdigest()
            for n in ('prepare_fresh_eval.py','prepare_features.py','siglip_model.py','sentence_features.py')}


def _check_shard(arrays, ids, expected_frame_counts, dimension):
    check_arrays(arrays,ids,dimension)
    if arrays['offsets'].dtype.kind not in 'iu' or not np.array_equal(np.diff(arrays['offsets']),expected_frame_counts):
        raise ValueError('Original frame count/order coverage changed')
    _check_arrays({'episode_ids':arrays['episode_ids'],'offsets':arrays['sentence_offsets'],
                   'vectors':arrays['sentence_vectors'],'positions':arrays['sentence_positions'],
                   'weights':arrays['sentence_weights']},ids,dimension)
    if arrays['full_token_counts'].dtype.kind not in 'iu' or np.any(arrays['full_token_counts']<1):
        raise ValueError('Invalid original full token counts')


def load_or_prepare_eval(manifest, split, encoder, output_dir, *, train_zip, shard_size=32):
    """Return original val-only (frame/text bank, sentence bank), resumably.

    ``encoder`` may be FrameEncoder or FrozenSiglip; its immutable parameters are
    used by FrameEncoder.frames without creating a second GPU model instance.
    Callers enforce full 834 coverage; small fixtures can test this routine.
    """
    if shard_size<1: raise ValueError('shard_size must be positive')
    ids=list(split['val_ids']);train=set(split['train_ids'])
    records={r['episode_id']:r for r in manifest['episodes']}
    if (not ids or len(ids)!=len(set(ids)) or train & set(ids) or
        len(records)!=len(manifest['episodes']) or not set(ids)<=records.keys()):
        raise ValueError('Invalid fixed validation episode selection')
    if read_manifest(train_zip)!=manifest:
        raise ValueError('Source ZIP differs from the fixed original contract')
    model=copy.deepcopy(encoder.provenance)
    if model.get('model_id')!=MODEL_ID or model.get('snapshot_commit')!=MODEL_REVISION:
        raise ValueError('Fresh cache needs the pinned allowed SigLIP revision')
    model.update(image_aggregation='none; one unit vector per original frame, numeric order retained',
                 cached_image_representation='all individual unit frame features in numeric order',
                 cached_text_representation='truncate64 and full sentence/window unit features',
                 position_information='sequence order retained; no learned temporal encoder in cache')
    request={'version':VERSION,'manifest_hash':digest(manifest),'split_hash':digest(split),
             'episode_ids':ids,'source_fingerprint':manifest['source']['fingerprint'],
             'encoder':model,'feature_dim':encoder.feature_dim,'shard_size':shard_size,
             'sources':_sources(),'training_steps':0,'selection':'validation only; every original frame'}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
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
                print(f'Fresh validation cache {start+len(batch_ids)}/{len(ids)}; reused={reused}',flush=True)
        index={'version':VERSION,'request_hash':fingerprint,'fingerprint':fingerprint,
               'source_fingerprint':manifest['source']['fingerprint'],'episode_ids':ids,
               'feature_dim':encoder.feature_dim,'model':model,'text_mode':'truncate64','shards':shards,
               'validation_only':True,'training_steps':0,'all_original_frames':True,
               'timing':{'elapsed_seconds':time.perf_counter()-started,'reused_episodes':reused}}
        if previous and previous.get('status')=='complete':
            old=read_json(folder/'index.json')
            if {k:v for k,v in old.items() if k!='timing'}!={k:v for k,v in index.items() if k!='timing'}:
                raise ValueError('Completed fresh-cache index differs')
            index=old  # Stable artifact hashes across completed reruns.
        else:
            atomic_json(folder/'index.json',index)
            atomic_json(folder/'status.json',{'status':'complete','request_hash':fingerprint,'index':file_record(folder/'index.json')})
        return _load_banks(folder,index)
    except BaseException as exc:
        atomic_json(folder/'status.json',{'status':'failed','request_hash':fingerprint,'error':str(exc)})
        raise


def _load_banks(folder,index):
    bank={'frames':{},'texts':{},'token_keys':{},'full_token_counts':{},'index':index}
    sentence={'features':{},'positions':{},'weights':{},
              'index':{'fingerprint':digest({'features':index['fingerprint'],'representation':'all ordered sentence windows'}),
                       'episode_ids':index['episode_ids'],'model':index['model'],'validation_only':True}}
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


def main():
    p=argparse.ArgumentParser()
    for name in ('train-zip','contracts','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--image-batch-size',type=int,default=8)
    p.add_argument('--text-batch-size',type=int,default=16);args=p.parse_args()
    manifest,split,_=load_contracts(args.contracts)
    if len(split['val_ids'])!=834:raise ValueError('CLI requires the fixed 834 validation episodes')
    encoder=FrameEncoder(args.device,args.image_batch_size,args.text_batch_size,revision=MODEL_REVISION)
    load_or_prepare_eval(manifest,split,encoder,args.output,train_zip=args.train_zip)


if __name__=='__main__':main()
