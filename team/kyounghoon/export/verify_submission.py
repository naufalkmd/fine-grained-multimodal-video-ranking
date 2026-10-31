"""Fresh-process offline raw-image inference parity and timing; never submitted."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact_hashes(folder):
    files = {}
    for path in sorted(Path(folder).rglob('*')):
        if '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('No artifact symlinks')
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = file_hash(path)
    return files


def run(args):
    root, run_dir = Path(args.artifact_dir).resolve(), Path(args.run_dir)
    # All imports used to load the model see an empty cache and offline flags.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    def denied(*unused_args, **unused_kwargs):
        raise RuntimeError('Network access attempted during offline verification')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    sys.path.insert(0, str(root))
    from submission import Submission

    request, comparison = read_json(run_dir/'request.json'), read_json(run_dir/'comparison.json')
    if comparison.get('result_status') != 'complete' or comparison.get('smoke'):
        raise ValueError('Need completed full training results')
    config = read_json(root/'inference_config.json')
    seeds = [x['seed'] for x in config['heads']]
    if seeds != [2026, 2027, 2028]:
        raise ValueError('Unexpected ensemble seeds')
    manifest = read_json(Path(args.contracts)/'manifest.json')
    candidates = read_json(run_dir/'team_validation.json')
    if manifest['source']['fingerprint'] != candidates['source_fingerprint']:
        raise ValueError('Wrong source data manifest')
    records = {x['episode_id']: x for x in manifest['episodes']}
    rows = candidates['episodes']
    if len(rows) != 834:
        raise ValueError('Expected fixed 834 validation episodes')
    if args.episodes < 1 or args.episodes > len(rows):
        raise ValueError('episodes must be between 1 and 834')
    if args.episodes < len(rows):
        # Include longest inputs, then fixed evenly spaced episodes. No performance-based selection.
        worst = sorted(rows, key=lambda x: (len(records[x['episode_id']]['image_members']),
                                           sum(len(c['text']) for c in x['choices'])), reverse=True)
        chosen = {x['episode_id'] for x in worst[:min(8, args.episodes)]}
        for i in np.linspace(0, len(rows)-1, args.episodes, dtype=int):
            if len(chosen) >= args.episodes:
                break
            chosen.add(rows[int(i)]['episode_id'])
        for row in rows:
            if len(chosen) >= args.episodes:
                break
            chosen.add(row['episode_id'])
        rows = [x for x in rows if x['episode_id'] in chosen]
    expected = []
    for seed in seeds:
        with gzip.open(run_dir/f'predictions/E0_minimal-seed{seed}.json.gz', 'rt', encoding='utf-8') as f:
            result = json.load(f)
        expected.append({x['episode_id']: x for x in result['original']['team']['predictions']})
    before_hashes = artifact_hashes(root)
    gpu = torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None
    if args.device.startswith('cuda') and 'T4' not in gpu:
        raise RuntimeError('Use the authorized free T4 GPU for GPU verification')
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = Submission(str(root), args.device)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    per_seed_max = [0.0]*3
    differences, changes, out_rows = [], [], []
    total_frames = 0
    robust_contract = None
    benchmark_start = time.perf_counter()
    with zipfile.ZipFile(args.train_zip) as archive:
        for index, row in enumerate(rows):
            ep = row['episode_id']
            # The inference module sees only public inputs, no source/label metadata.
            choices = [{'id': c['id'], 'text': c['text']} for c in row['choices']]
            images = []
            try:
                for name in records[ep]['image_members']:
                    with Image.open(io.BytesIO(archive.read(name))) as raw:
                        images.append(raw.convert('RGB'))
                total_frames += len(images)
                details = model.score_details(images, choices)
                scores = np.asarray(details['scores'], dtype=np.float64)
                reference = []
                for k, (seed, values) in enumerate(zip(seeds, details['seed_scores'])):
                    old = expected[k][ep]
                    previous = np.asarray([old['scores'][c['id']] for c in choices], dtype=np.float64)
                    delta = np.abs(np.asarray(values)-previous)
                    per_seed_max[k] = max(per_seed_max[k], float(delta.max()))
                    reference.append(previous)
                reference = np.mean(reference, axis=0)
                delta = np.abs(scores-reference)
                differences.extend(delta.tolist())
                ranking = np.argsort(-scores, kind='stable').tolist()
                old_ranking = np.argsort(-reference, kind='stable').tolist()
                if ranking != old_ranking:
                    changes.append({'episode_id':ep, 'top1_changed':ranking[0]!=old_ranking[0],
                                    'maximum_score_drift':float(delta.max())})
                answer = row['correct_choice_id']
                rank = [choices[i]['id'] for i in ranking].index(answer)+1
                reward = [1., .5, .25, .125, 0.][rank-1]
                out_rows.append({'episode_id':ep, 'scores':{c['id']:s for c,s in zip(choices,scores.tolist())},
                                 'ranking':[choices[i]['id'] for i in ranking],
                                 'correct_rank':rank,'reward':reward})
                if index == 0:
                    # Check candidate order and small variable-size inputs using real official text/images.
                    reverse = list(reversed(choices))
                    reversed_scores = model.score(images, reverse)
                    if not np.allclose(reversed_scores, scores[::-1], atol=0.002, rtol=0):
                        raise ValueError('Candidate permutation changes matching scores')
                    lone = model.predict(images[:1], choices[:1])
                    if lone != [choices[0]['id']]:
                        raise ValueError('Variable-length input contract failed')
                    ranking_api = model.predict(images, choices)
                    if sorted(ranking_api) != sorted(c['id'] for c in choices):
                        raise ValueError('API must return every ID exactly once')
                    if not np.allclose(model.score(images, choices), scores, atol=1e-6, rtol=0):
                        raise ValueError('Repeated episode changed scores')
                    robust_contract = {'choice_permutation':True,'one_frame_one_choice':True,
                                       'all_ids_once':True,'independent_repeat':True}
            finally:
                for image in images:
                    image.close()
            if (index+1)%25 == 0 or index+1 == len(rows):
                print(f'Offline raw-input validation {index+1}/{len(rows)}; max drift={max(per_seed_max):.7f}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-benchmark_start
    if artifact_hashes(root) != before_hashes:
        raise ValueError('Inference modified the submitted artifact')
    passed = max(per_seed_max) <= args.tolerance
    report = {'passed':passed, 'artifact_hashes':before_hashes, 'device':args.device,'gpu':gpu,
              't4_verified':bool(gpu and 'T4' in gpu),'network_calls_blocked':True,
              'empty_hf_cache':os.environ.get('HF_HOME'),'episodes':len(rows),'full_validation':len(rows)==834,
              'frames':total_frames,'load_seconds':load_seconds,'inference_and_checks_seconds':elapsed,
              'total_seconds':load_seconds+elapsed,'no_feature_cache':True,
              'timing_limit_note':'834 local episodes are not the hidden test. Its size and timing can differ.',
              'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if gpu else None,
              'score_tolerance':args.tolerance,'per_seed_max_abs_drift':dict(zip(map(str,seeds),per_seed_max)),
              'ensemble_drift_quantiles':dict(zip(['median','p95','max'],np.quantile(differences,[.5,.95,1]).tolist())),
              'ranking_changes':changes,'top1_changes':sum(x['top1_changed'] for x in changes),
              'contract_checks':robust_contract,
              'local_team_reward':float(np.mean([x['reward'] for x in out_rows])),
              'local_team_top1':float(np.mean([x['correct_rank']==1 for x in out_rows])),
              'predictions':out_rows}
    output=Path(args.report);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'artifact_hashes','predictions','ranking_changes'}},indent=2),flush=True)
    if not passed:
        raise ValueError('Exported score drift exceeds bound. No submission ZIP will be generated; inspect report.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('artifact-dir','run-dir','contracts','train-zip','report'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--episodes',type=int,default=834)
    p.add_argument('--device',default='cuda')
    p.add_argument('--tolerance',type=float,default=.002)
    run(p.parse_args())
