"""Synthetic CPU checks for leakage, paired sampling and schedule recovery."""
from __future__ import annotations
import copy
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_io import atomic_json,digest,file_record
from minimal_training_data import prepare_train_minimal,MinimalTrainingSchedules,BASE_MIX,full_input_key


class Tokenizer:
    is_fast=True;eos_token_id=1;pad_token_id=1
    def encode(self,text,add_special_tokens=False,truncation=False):
        ids=[2+int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%50000 for x in re.findall(r'\w+|[^\w\s]',text)]
        return ids+[1] if add_special_tokens else ids
    def num_special_tokens_to_add(self,pair=False):return 1
    def build_inputs_with_special_tokens(self,ids):return ids+[1]
    def __call__(self,text,**kwargs):
        ids=self.encode(text)
        if kwargs.get('truncation'):ids=ids[:kwargs['max_length']-1]
        if kwargs.get('add_special_tokens',True):ids=ids+[1]
        if kwargs.get('padding')=='max_length':ids=ids+[1]*(kwargs['max_length']-len(ids))
        return {'input_ids':ids}


class BaseStore:
    def __init__(self,root):
        self.train_ids=[f't{i}' for i in range(5)]
        self.mix=BASE_MIX.copy();self.fingerprint='base';self.output_dir=Path(root)
        self.descriptions={ep:'symbol_'+ep for ep in self.train_ids}
    def epoch(self,mode,seed,epoch):
        rows=[]
        for i,ep in enumerate(self.train_ids):
            sources=self.train_ids[i+1:]+self.train_ids[:i]
            rows.append({'episode_id':ep,'negative_sources':[{'type':kind,'source':source} for kind,source in zip(['random','random','lexical','similar_episode'],sources)]})
        raw={'records':rows};path=self.output_dir/f'{seed}-{epoch}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(raw).encode(),mtime=0))
        schedule=[(r['episode_id'],[r['episode_id']]+[v['source'] for v in r['negative_sources']]) for r in rows]
        return schedule,{'file':{'path':path.name,**file_record(path)}}


def suite_for(base):
    suite={'records':[],'by_episode':{ep:[] for ep in base.train_ids},
           'full_input_keys':{ep:digest(ep) for ep in base.train_ids}}
    for ep in base.train_ids:
        for group,count in [('color',9),('action',1)]:
            for i in range(count):
                text=f'symbol_{ep}_{group}_{i}'
                row={'id':digest(text),'episode_id':ep,'source_split':'train',
                     'positive_text':base.descriptions[ep],'negative_text':text,'aug_id':'text_'+digest(text),
                     'type':'attribute_color' if group=='color' else 'action_lift_lower','semantic_group':group,
                     'positive_full_input_key':suite['full_input_keys'][ep],'full_input_key':digest(text)}
                suite['records'].append(row);suite['by_episode'][ep].append(row)
    suite['fingerprint']=digest(suite)
    return suite


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_training_vocabulary_and_generation_exclude_validation_content(self):
        manifest={'episodes':[{'episode_id':'t0','description':'They hold a green triangular paper.'},
                              {'episode_id':'t1','description':'They hold a red circular paper.'},
                              {'episode_id':'v0','description':'They hold a blue square paper.'}]}
        split={'train_ids':['t0','t1'],'val_ids':['v0']};actions={'t0':[],'t1':[]}
        suite=prepare_train_minimal(manifest,split,Tokenizer(),actions,self.root/'data')
        self.assertTrue(suite['records'])
        self.assertEqual(set(suite['by_episode']),{'t0','t1'})
        self.assertTrue(all(r['episode_id'] in split['train_ids'] for r in suite['records']))
        self.assertFalse(any(r['evidence']['new'].casefold()=='blue' for r in suite['records']))
        self.assertEqual(suite['audit']['validation_sources'],0)
        # Changing only a held-out description must not change generated train data.
        changed=copy.deepcopy(manifest);changed['episodes'][-1]['description']='symbol_validation_only'
        other=prepare_train_minimal(changed,split,Tokenizer(),actions,self.root/'other')
        self.assertEqual(suite,other)
        repeat=prepare_train_minimal(manifest,split,Tokenizer(),actions,self.root/'data')
        self.assertEqual(suite,repeat)

    def test_validation_action_labels_are_rejected(self):
        manifest={'episodes':[{'episode_id':'t','description':'symbol_a'},{'episode_id':'v','description':'symbol_b'}]}
        with self.assertRaisesRegex(ValueError,'exactly training'):
            prepare_train_minimal(manifest,{'train_ids':['t'],'val_ids':['v']},Tokenizer(),{'t':[],'v':[]},self.root/'data')

    def test_changes_after_base_window_are_retained_but_full_collisions_are_not(self):
        tokenizer=Tokenizer();a=' '.join(['symbol']*70)+' token_a';b=' '.join(['symbol']*70)+' token_b'
        self.assertEqual(tokenizer(a,truncation=True,max_length=64)['input_ids'],tokenizer(b,truncation=True,max_length=64)['input_ids'])
        self.assertNotEqual(full_input_key(tokenizer,a),full_input_key(tokenizer,b))

    def test_only_lexical_slot_changes_and_sampling_is_group_balanced(self):
        base=BaseStore(self.root/'base');suite=suite_for(base)
        store=MinimalTrainingSchedules(base,suite,self.root/'minimal')
        totals=Counter()
        for epoch in range(40):
            changed,audit=store.epoch('team',2026,epoch);original,_=base.epoch('team',2026,epoch)
            for (ep,new),(before_ep,old) in zip(changed,original):
                self.assertEqual(ep,before_ep);self.assertEqual(new[:3],old[:3]);self.assertEqual(new[4],old[4])
                self.assertNotEqual(new[3],old[3]);self.assertTrue(new[3].startswith('text_'))
            totals.update(audit['semantic_groups_used'])
            self.assertEqual(audit['replaced_episodes'],5)
        self.assertEqual(sum(totals.values()),200)
        self.assertTrue(60<=totals['action']<=140,totals)  # Nine color candidates must not give 9:1 sampling.

    def test_duplicate_inputs_fall_back_without_touching_other_slots(self):
        base=BaseStore(self.root/'base');suite=suite_for(base)
        for row in suite['by_episode']['t0']: row['full_input_key']=suite['full_input_keys']['t1']
        suite['fingerprint']=digest({k:v for k,v in suite.items() if k!='fingerprint'})
        store=MinimalTrainingSchedules(base,suite,self.root/'minimal')
        new,audit=store.epoch('team',7,2);old,_=base.epoch('team',7,2)
        self.assertEqual(new[0],old[0]);self.assertEqual(audit['fallback_episodes'],1)
        self.assertEqual(audit['fallback_reasons'],{'no_noncolliding_minimal_candidate':1})

    def test_resume_is_stable_and_corruption_rejected(self):
        base=BaseStore(self.root/'base');store=MinimalTrainingSchedules(base,suite_for(base),self.root/'minimal')
        first,audit=store.epoch('team',2026,1);second,again=store.epoch('team',2026,1)
        self.assertEqual(first,second);self.assertEqual(audit,again)
        path=store.output_dir/audit['file']['path'];path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt'):store.epoch('team',2026,1)

    def test_candidate_source_leakage_is_rejected(self):
        base=BaseStore(self.root/'base');suite=suite_for(base)
        suite['by_episode']['t0'][0]['source_split']='validation'
        suite['fingerprint']=digest({k:v for k,v in suite.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'own training'):
            MinimalTrainingSchedules(base,suite,self.root/'minimal')


if __name__=='__main__':unittest.main()
