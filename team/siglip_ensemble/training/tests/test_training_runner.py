"""Tiny CPU integration using the actual sampler, head training and scoring."""
from __future__ import annotations
import copy
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from comparison_protocol import rank_prediction
from experiment_io import atomic_json,digest,file_record,read_json
from hard_negative_data import build_suite
from minimal_training_data import prepare_train_minimal,MinimalTrainingSchedules
from run_comparison import training_schedule
from run_minimal_training import load_baseline,score_run,summarize,paired_intervals,saved_drift
from run_sentence_comparison import evaluate,train_head
from sentence_models import FrozenSiglipResidual
from training_schedules import TrainingSchedules
from test_training_data import Tokenizer


def fixture(root):
    train=[f't{i}' for i in range(5)];val=[f'v{i}' for i in range(5)]
    modifiers=['green triangular','red circular','blue square','yellow triangular','black circular']
    manifest={'episodes':[{'episode_id':ep,'description':f'They hold a {modifiers[i%5]} paper. Symbol_{ep}.'}
                          for i,ep in enumerate(train+val)]}
    split={'train_ids':train,'val_ids':val};actions={ep:[] for ep in train}
    tokenizer=Tokenizer()
    suite=prepare_train_minimal(manifest,split,tokenizer,actions,root/'augmentation')
    val_suite=build_suite(manifest,split,tokenizer,{**actions,**{ep:[] for ep in val}})
    generator=torch.Generator().manual_seed(17)
    bank={'frames':{},'texts':{},'token_keys':{}}
    sentences={'features':{},'positions':{},'weights':{}}
    ids={}
    for i,row in enumerate(manifest['episodes']):
        ep=row['episode_id'];vector=F.normalize(torch.randn(8,generator=generator),dim=-1)
        bank['texts'][ep]=vector;bank['frames'][ep]=F.normalize(vector[None]+.2*torch.randn(2+i%3,8,generator=generator),dim=-1)
        bank['token_keys'][ep]=digest(ep)
        sentences['features'][ep]=F.normalize(vector[None]+.1*torch.randn(2,8,generator=generator),dim=-1)
        sentences['positions'][ep]=torch.tensor([0.,1.]);sentences['weights'][ep]=torch.tensor([10.,5.])
        ids[row['description']]='text_'+digest(row['description'])
        key=ids[row['description']]
        bank['texts'][key]=bank['texts'][ep]
        for field in sentences:sentences[field][key]=sentences[field][ep]
    for row in suite['records']+val_suite['records']:
        text=row['negative_text'];key='text_'+digest(text);ids[text]=key
        if key not in bank['texts']:
            bank['texts'][key]=F.normalize(torch.randn(8,generator=generator),dim=-1)
            sentences['features'][key]=F.normalize(torch.randn(2,8,generator=generator),dim=-1)
            sentences['positions'][key]=torch.tensor([0.,1.]);sentences['weights'][key]=torch.tensor([10.,5.])
    descriptions={r['episode_id']:r['description'] for r in manifest['episodes']}
    base=TrainingSchedules(train,{ep:descriptions[ep] for ep in train},actions,bank['token_keys'],root/'schedules/base',
                           mix_text='random=2,lexical=1,similar_episode=1',random_schedule_fn=training_schedule)
    minimal=MinimalTrainingSchedules(base,suite,root/'schedules/minimal')
    candidates=[]
    for ep in val:
        choices=[{'id':f'choice_{i}','source_episode_id':source,'text':descriptions[source]}
                 for i,source in enumerate(val)]
        candidates.append({'episode_id':ep,'choices':choices,'correct_choice_id':f'choice_{val.index(ep)}'})
    sets={'random':candidates,'team':copy.deepcopy(candidates)}
    config={'model':{'feature_dim':8,'hidden_dim':8,'dropout':0.,'attention_temperature':.07,'residual_limit':.1},
            'epochs':1,'min_epochs':1,'patience':1,'batch_size':2,'learning_rate':.01,'weight_decay':.01,
            'loss_temperature':.07,'random_candidates_hash':digest(candidates),'team_candidates_hash':digest(candidates)}
    reference={label:evaluate(None,rows,bank,sentences,'cpu',2,.07) for label,rows in sets.items()}
    return manifest,split,bank,sentences,suite,val_suite,ids,base,minimal,sets,config,reference


class TrainingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_actual_sampler_train_score_select_and_resume(self):
        # Keep the actual team's sampling/RNG/fill/record logic. Supply its
        # lexical matrix as a fixed fixture so CPU CI needs no sklearn install.
        matrix=np.ones((5,5),dtype=np.float32);np.fill_diagonal(matrix,-np.inf)
        lexical=patch('team.negatives.lexical_similarity',return_value=matrix)
        lexical.start();self.addCleanup(lexical.stop)
        manifest,split,bank,sentences,suite,val_suite,ids,base,minimal,sets,config,reference=fixture(self.root)
        model_runs=[]
        for name,store in [('control',base),('minimal',minimal)]:
            folder=self.root/name
            result=train_head('sentence',2026,config,bank,sentences,sets,reference,folder,'cpu','request-'+name,
                              training_mode='mixed',schedule_store=store)
            initial=torch.load(folder/'initial.pt',weights_only=True)
            last=torch.load(folder/'last.pt',weights_only=True)
            best=torch.load(folder/'best.pt',weights_only=True)
            self.assertEqual(result['optimizer_steps'],3)
            self.assertIn(best['epoch'],[0,1]);self.assertEqual(best['epoch'],result['best_epoch'])
            self.assertTrue(any(not torch.equal(initial['state_dict'][key],last['state_dict'][key]) for key in last['state_dict']))
            self.assertTrue(result['zero_gate']['passed'])
            self.assertEqual(set(r['episode_id'] for r in result['team']['predictions']),set(split['val_ids']))
            used=result['used_negative_schedules'][0]
            self.assertEqual(used['pool'],'training_only')
            if name=='minimal':
                self.assertGreater(used['replaced_episodes'],0)
                self.assertEqual(set(suite['by_episode']),set(split['train_ids']))
                raw=read_json(minimal.output_dir/'seed2026/epoch-000.json.gz.json')
                payload=__import__('json').loads(gzip.decompress((minimal.output_dir/raw['file']['path']).read_bytes()))
                for row in payload['records']:
                    self.assertIn(row['episode_id'],split['train_ids'])
                    self.assertFalse(set(row['choices_before']) & set(split['val_ids']))
            again=train_head('sentence',2026,config,bank,sentences,sets,reference,folder,'cpu','request-'+name,
                             training_mode='mixed',schedule_store=store)
            self.assertEqual(again,result)
            model=FrozenSiglipResidual('sentence',**config['model']).eval();model.load_state_dict(best['state_dict'])
            scored=score_run('E0_original' if name=='control' else 'E0_minimal',2026,model,sets,val_suite['records'],
                             ids,bank,sentences,'cpu',config,self.root/'predictions',{'best_epoch':best['epoch']})
            self.assertEqual(scored['minimal']['minimal']['episodes'],len(split['val_ids']))
            model_runs.append(scored)
        # Original initialization is paired, but the actual training choices differ.
        self.assertEqual(model_runs[0]['seed'],model_runs[1]['seed'])
        summaries=summarize(model_runs)
        self.assertIn('minimal_episode_win',summaries['E0_minimal']['metrics'])
        differences=paired_intervals(model_runs,bootstrap=100)
        self.assertEqual(differences['E0_minimal_minus_E0_original']['team']['episodes'],5)
        # Simulate interruption after checkpoint output but before complete marker.
        folder=self.root/'minimal';before=torch.load(folder/'last.pt',weights_only=True)['state_dict']
        (folder/'complete.json').unlink()
        repeated=train_head('sentence',2026,config,bank,sentences,sets,reference,folder,'cpu','request-minimal',
                            training_mode='mixed',schedule_store=minimal)
        after=torch.load(folder/'last.pt',weights_only=True)['state_dict']
        self.assertTrue(all(torch.equal(before[k],after[k]) for k in before))
        self.assertEqual(repeated['optimizer_steps'],3)

    def test_load_baseline_checks_provenance_and_real_weights(self):
        config={'feature_dim':8,'hidden_dim':8,'dropout':0.}
        model=FrozenSiglipResidual('sentence',**config)
        path=self.root/'weights/E0/seed2026/best.pt';path.parent.mkdir(parents=True)
        checkpoint={'run_hash':'source','experiment':'E0','seed':2026,'epoch':2,'frame_transformer':False,
                    'model_config':config,'state_dict':model.state_dict()}
        torch.save(checkpoint,path)
        data={'run_hash':'source','run_config':{'model':config},'heads':[{'experiment':'E0','seed':2026,'best_epoch':2,
                                                                     'checkpoint_record':file_record(path)}]}
        restored,_,_=load_baseline(data,self.root/'weights',2026,'cpu')
        self.assertFalse(restored.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
        self.assertTrue(all(torch.equal(value,restored.state_dict()[key]) for key,value in model.state_dict().items()))
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt'):load_baseline(data,self.root/'weights',2026,'cpu')

    def test_intervals_use_real_rank_prediction_field(self):
        choices=[{'id':f'c{i}'} for i in range(5)]
        old=rank_prediction('v0',choices,[.9,.8,.7,.6,.5],'c1')
        new=rank_prediction('v0',choices,[.8,.9,.7,.6,.5],'c1')
        self.assertIn('rank_reward',old);self.assertNotIn('reward',old)
        def row(name,prediction,win):
            return {'model':name,'original':{'team':{'predictions':[prediction]}},
                    'predictions':[{'episode_id':'v0','win':win}]}
        result=paired_intervals([row('E0_original',old,0),row('E0_minimal',new,1)],bootstrap=20)
        self.assertEqual(result['E0_minimal_minus_E0_original']['team']['mean_delta'],.5)
        self.assertEqual(result['E0_minimal_minus_E0_original']['minimal_pair']['mean_delta'],1.)

    def test_stale_drift_diagnostic_does_not_hide_identity_errors(self):
        path=self.root/'previous_drift.json';atomic_json(path,{'passed':False,'maximum_score_difference':.003})
        current={'predictions':[{'episode_id':'v0','correct_choice_id':'a','scores':{'a':.3,'b':.2},'ranking':['a','b']}]}
        previous=copy.deepcopy(current);previous['predictions'][0]['correct_choice_id']='b'
        with self.assertRaisesRegex(ValueError,'choices differ'):
            saved_drift(current,previous,path)


if __name__=='__main__':unittest.main()
