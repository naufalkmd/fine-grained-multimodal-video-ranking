"""Train E0 with TRAIN-only minimal edits; evaluate on the frozen development sets.

The baseline and new arm are always scored anew on the same frozen features.
Saved old predictions are numerical diagnostics only, never training targets.
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
from pathlib import Path
import statistics
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from data_contract import load_contracts
from experiment_io import atomic_json, digest, file_record, read_json, verified_file
from hard_negative_data import build_suite
from minimal_training_data import prepare_train_minimal, MinimalTrainingSchedules
from prepare_team_validation import read_action_labels
from run_comparison import seed_all, training_schedule
from run_hard_evaluation import (compare_predictions, grouped_metrics, five_choice_metrics,
                                 score_pairs, immutable_json, normalize_encoder_metadata)
from run_sentence_comparison import evaluate, train_head, SELECTION
from sentence_models import FrozenSiglipResidual
from sentence_features import MODEL_REVISION
from siglip_model import FrozenSiglip
from training_features import prepare_training_features, prepare_generated_texts
from training_schedules import TrainingSchedules

VERSION='siglip-train-minimal-e0-v1'
SEEDS=[2026,2027,2028]
MINIMAL_HASH='0de4aec95b9f99ee2545c7827d391598035f9395b474e81f3256d98e7b2262d6'


def sources():
    root=Path(__file__).resolve().parent
    paths=list(root.glob('*.py'))+list((root/'team').glob('*.py'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('resume_')}


def portable_config(path,split,random_candidates):
    data=read_json(path);config=data['run_config']
    if (data.get('version')!='factorial-portable-v1' or data['run_hash']!=digest(config)
        or config.get('split_hash')!=digest(split) or data['random_candidates']!=random_candidates
        or config.get('random_candidates_hash')!=digest(random_candidates)
        or config.get('team_candidates_hash')!=digest(data['team_candidates'])):
        raise ValueError('Embedded model metadata or original validation contracts changed')
    root=Path(__file__).resolve().parent
    for name,expected in config['sources'].items():
        if hashlib.sha256((root/name).read_text(encoding='utf-8').encode()).hexdigest()!=expected:
            raise ValueError('Frozen original model/training source changed: '+name)
    return data,config


def load_baseline(data,folder,seed,device):
    entries=[r for r in data['heads'] if r['experiment']=='E0' and r['seed']==seed]
    if len(entries)!=1:raise ValueError('Missing/duplicate E0 metadata')
    item=entries[0];path=Path(folder)/'E0'/f'seed{seed}'/'best.pt'
    verified_file(path,item['checkpoint_record'])
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    expected={'run_hash':data['run_hash'],'experiment':'E0','seed':seed,'epoch':item['best_epoch'],
              'frame_transformer':False,'model_config':data['run_config']['model']}
    if any(checkpoint.get(k)!=v for k,v in expected.items()):
        raise ValueError('Original E0 checkpoint identity differs')
    model=FrozenSiglipResidual('sentence',**data['run_config']['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.requires_grad_(False).eval()
    return model,item,path


def saved_drift(current,saved,path):
    """Nonblocking history audit: current comparisons never mix old/new scores."""
    ids={r['episode_id'] for r in current['predictions']}
    saved={**saved,'predictions':[r for r in saved['predictions'] if r['episode_id'] in ids]}
    try:return compare_predictions(current,saved,.002,diagnostic_path=path)
    except ValueError as exc:
        if 'Reused checkpoint score drift' not in str(exc) or not path.exists():raise
        report=read_json(path)
        print('Historical score drift exceeds .002; current baseline/new scores both recomputed:',path,flush=True)
        return report


def score_run(name,seed,model,sets,records,text_ids,bank,sentences,device,config,output,extra=None):
    original={label:evaluate(model,rows,bank,sentences,device,config['batch_size'],config['loss_temperature'])
              for label,rows in sets.items()}
    pairs=score_pairs(model,records,text_ids,bank,sentences,device,config['batch_size'])
    result={'model':name,'seed':seed,'original':original,'minimal':grouped_metrics(records,pairs),
            'five_choice':five_choice_metrics(records,pairs),'predictions':pairs,**(extra or {})}
    output.mkdir(parents=True,exist_ok=True)
    name_key=name if seed is None else f'{name}-seed{seed}'
    with gzip.open(output/(name_key+'.json.gz'),'wt',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False)
    return result


def summarize(runs):
    grouped=defaultdict(list)
    for row in runs:grouped[row['model']].append(row)
    report={}
    for name,rows in grouped.items():
        values={}
        for label in ('random','team'):
            for key in ('mean_rank_reward','top1_accuracy'):
                values[label+'_'+key]=[r['original'][label]['metrics'][key] for r in rows]
        for key in rows[0]['minimal']:
            values[key+'_episode_win']=[r['minimal'][key]['episode_macro_win_rate'] for r in rows]
        if 'minimal' in rows[0]['five_choice']:
            values['minimal_5choice_reward']=[r['five_choice']['minimal']['mean_rank_reward'] for r in rows]
            values['minimal_5choice_top1']=[r['five_choice']['minimal']['top1_accuracy'] for r in rows]
        report[name]={'seeds':len(rows),'metrics':{key:{'mean':statistics.mean(v),
            'std':statistics.stdev(v) if len(v)>1 else 0.0} for key,v in values.items()},
            'best_epochs':[r.get('best_epoch') for r in rows]}
    return report


def paired_intervals(runs,bootstrap=2000):
    values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in runs:
        for label,data in r['original'].items():
            for p in data['predictions']:
                values[r['model']][label][p['episode_id']].append(p['rank_reward'])
        for p in r['predictions']:
            values[r['model']]['minimal_pair'][p['episode_id']].append(p['win'])
    result={}
    for old in ('E0_original','SigLIP_zero_shot'):
        if old not in values:continue
        result["E0_minimal_minus_"+old]={}
        for label,current in values['E0_minimal'].items():
            previous=values[old][label]
            if set(previous)!=set(current):raise ValueError('Paired coverage differs')
            ids=sorted(current)
            delta=np.asarray([statistics.mean(current[ep])-statistics.mean(previous[ep]) for ep in ids])
            rng=np.random.default_rng(2032)
            boots=[float(delta[rng.integers(len(delta),size=len(delta))].mean()) for _ in range(bootstrap)]
            result['E0_minimal_minus_'+old][label]={'episodes':len(ids),'mean_delta':float(delta.mean()),
                'ci95':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_samples':bootstrap,
                'limitations':'Episode bootstrap conditional on fitted checkpoints; excludes label noise, selection and hidden-test shift.'}
    return result


def package_results(output):
    excluded={'original_features','generated_features','__pycache__'}
    files={p.relative_to(output).as_posix():file_record(p) for p in output.rglob('*')
           if p.is_file() and p.name!='status.json' and p.suffix not in ('.pt','.npz','.zip','.log','.tmp')
           and not excluded.intersection(p.relative_to(output).parts)}
    atomic_json(output/'status.json',{'status':'complete','artifacts':files})
    with zipfile.ZipFile(output/'minimal_training_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'status.json']:z.write(output/name,name)


def run(args):
    started=time.perf_counter();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda':
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Use one free Colab/Kaggle T4 GPU')
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory>17*2**30:
            raise RuntimeError('One T4 only')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest,split,randoms=load_contracts(args.contracts)
    if len(split['train_ids'])!=3336 or len(split['val_ids'])!=834:raise ValueError('Use fixed original 3336/834 split')
    data,old_config=portable_config(args.portable_models,split,randoms)
    config={**old_config,'epochs':1 if args.smoke else args.epochs,'min_epochs':1 if args.smoke else old_config['min_epochs'],
            'patience':1 if args.smoke else old_config['patience'],'batch_size':args.batch_size}
    seeds=SEEDS[:1] if args.smoke else SEEDS
    if not args.train_control:
        if not args.weights_dir:raise ValueError('Provide E0 weights, or use --train-control to retrain both arms')
        for seed in seeds:
            model,_,_=load_baseline(data,args.weights_dir,seed,'cpu');del model
    encoder=FrozenSiglip(args.device,text_batch_size=args.text_batch_size,revision=MODEL_REVISION)
    normalize_encoder_metadata(encoder)
    bank,sentences=prepare_training_features(manifest,split,encoder,output/'original_features',train_zip=args.train_zip)
    originals={r['episode_id']:r for r in manifest['episodes']}
    train_ids=split['train_ids'];train_set=set(train_ids)
    train_actions=read_action_labels(args.train_zip,manifest,train_ids)
    train_suite=prepare_train_minimal(manifest,split,encoder.tokenizer,train_actions,output/'augmentation')
    if not train_suite['records']:raise ValueError('No training minimum-edit candidates')
    if not {r['episode_id'] for r in train_suite['records']}<=train_set:raise ValueError('Training augmentation leakage')
    actions={**train_actions,**read_action_labels(args.train_zip,manifest,split['val_ids'])}
    val_suite=build_suite(manifest,split,encoder.tokenizer,action_labels=actions,seed=2031,max_additions_per_episode=0)
    val_records=val_suite['records']
    if len(val_records)!=1918 or digest(val_records)!=MINIMAL_HASH:
        raise ValueError('Validation minimum-edit candidates changed from completed v2 evaluation')
    immutable_json(output/'minimal_validation.json',{'records':val_records,'hash':MINIMAL_HASH,'checkpoint_selection':False})
    bank,sentences,text_metadata=prepare_generated_texts(train_suite['records']+val_records,bank,sentences,encoder,output/'generated_features')
    # Keep the unchanged old candidates; new hard validation is final-only.
    sets={'random':randoms['episodes'],'team':data['team_candidates']['episodes']}
    if args.smoke:sets={k:v[:8] for k,v in sets.items()}
    base=TrainingSchedules(train_ids,{ep:originals[ep]['description'] for ep in train_ids},train_actions,
                           bank['token_keys'],output/'negative_schedules/base',
                           mix_text='random=2,lexical=1,similar_episode=1',fill='random',random_schedule_fn=training_schedule)
    minimal=MinimalTrainingSchedules(base,train_suite,output/'negative_schedules/minimal')
    request={'version':VERSION,'seeds':seeds,'smoke':args.smoke,'train_control':args.train_control,
             'source_run_hash':data['run_hash'],'split_hash':digest(split),'train_ids_hash':digest(train_ids),
             'random_validation_hash':digest(randoms),'team_validation_hash':digest(data['team_candidates']),
             'minimal_validation_hash':MINIMAL_HASH,'training_augmentation':train_suite['fingerprint'],
             'model':config['model'],'epochs':config['epochs'],'min_epochs':config['min_epochs'],'patience':config['patience'],
             'learning_rate':config['learning_rate'],'batch_size':config['batch_size'],'loss_temperature':config['loss_temperature'],
             'text_batch_size':args.text_batch_size,'device':args.device,'sources':sources(),
             'original_feature_fingerprint':bank['index']['fingerprint'],'generated_features':text_metadata['fingerprint'],
             'selection':SELECTION,'new_hard_validation_selection':False,'base_schedule':base.fingerprint,
             'minimal_schedule':minimal.fingerprint,'initialization':'identical seed; zero residual; trained from scratch',
             'training':'official positive indices and code-produced weak negatives; no teacher targets',
             'weights':[{'seed':seed,'record':next(r['checkpoint_record'] for r in data['heads'] if r['experiment']=='E0' and r['seed']==seed)} for seed in seeds] if not args.train_control else [],
             'semantic_warning':'Edits remain rule assumptions; final-only development score is not a leaderboard estimate.'}
    immutable_json(output/'request.json',request);run_hash=digest(request)
    source_root=Path(__file__).resolve().parent
    for name in request['sources']:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        text=(source_root/name).read_text(encoding='utf-8')
        if target.exists() and target.read_text(encoding='utf-8')!=text:raise ValueError('Source snapshot changed')
        if not target.exists():target.write_text(text,encoding='utf-8',newline='\n')
    immutable_json(output/'random_validation.json',randoms);immutable_json(output/'team_validation.json',data['team_candidates'])
    del encoder;gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
    atomic_json(output/'status.json',{'status':'running','run_hash':run_hash})
    try:
        references={k:evaluate(None,v,bank,sentences,args.device,args.batch_size,config['loss_temperature']) for k,v in sets.items()}
        runs=[score_run('SigLIP_zero_shot',None,None,sets,val_records,text_metadata['text_ids'],bank,sentences,
                        args.device,config,output/'predictions')]
        for seed in seeds:
            if args.train_control:
                trained=train_head('sentence',seed,config,bank,sentences,sets,references,output/'heads/control'/f'seed{seed}',
                                   args.device,digest([run_hash,'control']),training_mode='mixed',schedule_store=base)
                cp=output/'heads/control'/f'seed{seed}'/'best.pt'
                model=FrozenSiglipResidual('sentence',**config['model']).to(args.device)
                model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']);model.eval()
                info={'best_epoch':trained['best_epoch'],'checkpoint_source':'new paired control training'}
            else:
                model,item,cp=load_baseline(data,args.weights_dir,seed,args.device)
                current={k:evaluate(model,v,bank,sentences,args.device,args.batch_size,config['loss_temperature']) for k,v in sets.items()}
                drift={k:saved_drift(v,item['result'][k],output/'historical_drift'/f'E0-{seed}-{k}.json') for k,v in current.items()}
                info={'best_epoch':item['best_epoch'],'checkpoint_source':'verified original E0 weights; current predictions recomputed',
                      'historical_drift':drift,'checkpoint_record':file_record(cp)}
            runs.append(score_run('E0_original',seed,model,sets,val_records,text_metadata['text_ids'],bank,sentences,args.device,config,output/'predictions',info))
            del model
            folder=output/'heads/minimal'/f'seed{seed}'
            trained=train_head('sentence',seed,config,bank,sentences,sets,references,folder,args.device,
                               digest([run_hash,'minimal']),training_mode='mixed',schedule_store=minimal)
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            model=FrozenSiglipResidual('sentence',**config['model']).to(args.device)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            runs.append(score_run('E0_minimal',seed,model,sets,val_records,text_metadata['text_ids'],bank,sentences,args.device,config,output/'predictions',
                        {'best_epoch':trained['best_epoch'],'baseline_fallback':trained['baseline_fallback'],
                         'initial_parameters_hash':trained['initial_parameters_hash'],'checkpoint_record':file_record(folder/'best.pt')}))
            del model;gc.collect()
        report={'version':VERSION,'result_status':'complete','run_hash':run_hash,'smoke':args.smoke,
                'full_train_episodes':3336,'full_validation_episodes':834,'evaluated_original_episodes':len(sets['team']),
                'minimal_validation_episodes':len({r['episode_id'] for r in val_records}),'minimal_validation_pairs':len(val_records),
                'models':summarize(runs),'paired_effects':paired_intervals(runs),'training_augmentation':train_suite['audit'],
                'selection':SELECTION,'new_hard_validation_selection':False,'elapsed_seconds':time.perf_counter()-started,
                'control_mode':'paired_retrain' if args.train_control else 'reused_weights_recomputed_scores',
                'semantic_warning':request['semantic_warning']}
        atomic_json(output/'comparison.json',report)
        rows=[{'model':model,'metric':key,**value} for model,data in report['models'].items() for key,value in data['metrics'].items()]
        with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['model','metric','mean','std']);writer.writeheader();writer.writerows(rows)
        package_results(output)
        print(json.dumps(report['models'],ensure_ascii=False,indent=2));print('Share:',output/'minimal_training_results.zip',flush=True)
        return report
    except BaseException as exc:
        atomic_json(output/'status.json',{'status':'failed','run_hash':run_hash,'error':str(exc)});raise


def main():
    parser=argparse.ArgumentParser()
    for name in ('contracts','train-zip','portable-models','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--weights-dir');parser.add_argument('--train-control',action='store_true')
    parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--text-batch-size',type=int,default=16)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.text_batch_size<1:parser.error('Positive epochs/batch sizes required')
    if not args.smoke and args.epochs!=30:parser.error('Use 30 epochs for the preregistered comparison')
    run(args)


if __name__=='__main__':main()
