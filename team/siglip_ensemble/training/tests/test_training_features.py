"""CPU fixtures for validation-only frame/text caching; no pretrained downloads."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]/'siglip-hard-eval-v2'))
import training_features as fresh
from experiment_io import read_json, file_record
from siglip_model import make_text_plan, MODEL_ID
from sentence_features import MODEL_REVISION


class Tokenizer:
    is_fast=True; eos_token_id=1;pad_token_id=1
    def encode(self,text,add_special_tokens=False,truncation=False):
        values=[2+sum(map(ord,w))%41 for w in text.split()]
        return values+[1] if add_special_tokens else values
    def num_special_tokens_to_add(self,pair=False):return 1
    def __call__(self,text,padding=None,max_length=64,**kwargs):
        ids=self.encode(text)[:max_length-1]+[1]
        return {'input_ids':ids+[1]*(max_length-len(ids))}


class Encoder:
    feature_dim=8;tokenizer=Tokenizer();text_batch_size=2;image_batch_size=2
    provenance={'model_id':MODEL_ID,'snapshot_commit':MODEL_REVISION,'tokenizer':{'backend_sha256':'fixture'}}
    def __init__(self):self.calls=[]
    def text_features(self,choices):
        self.calls.extend(c['id'] for c in choices)
        return {'truncate64':np.tile(np.eye(8,dtype=np.float32)[0],(len(choices),1))},[make_text_plan(self.tokenizer,c['text']) for c in choices]


def frame_vectors(encoder,archive,members):
    return np.eye(8,dtype=np.float32)[np.arange(len(members))%8]


def sentence_vectors(encoder,plans):
    return [np.tile(np.eye(8,dtype=np.float32)[1],(len(p['segments']),1)) for p in plans]


class TrainingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.root=Path(self.tmp.name)
        self.zip=self.root/'source.zip'
        with zipfile.ZipFile(self.zip,'w') as z:z.writestr('fixture.txt','fixture')
        self.manifest={'source':{'fingerprint':'test-source'},'episodes':[
            {'episode_id':'train','description':'Train placeholder.','image_members':['train/1.jpg']},
            {'episode_id':'val1','description':'First placeholder. Second placeholder.', 'image_members':['val1/1.jpg','val1/2.jpg','val1/3.jpg']},
            {'episode_id':'val2','description':'Long '+'placeholder '*80+'.','image_members':['val2/1.jpg','val2/2.jpg']}]}
        self.split={'train_ids':['train'],'val_ids':['val2','val1']};self.encoder=Encoder()
        self.output=self.root/'cache'

    def tearDown(self):self.tmp.cleanup()

    def run_cache(self,**kwargs):
        with patch.object(fresh,'read_manifest',return_value=self.manifest), \
             patch.object(fresh.FrameEncoder,'frames',frame_vectors), \
             patch.object(fresh,'_encode_plans',sentence_vectors):
            return fresh.prepare_training_features(self.manifest,self.split,self.encoder,self.output,
                                             train_zip=self.zip,shard_size=1,**kwargs)

    def test_all_originals_are_encoded_and_real_split_is_bound(self):
        bank,sentences=self.run_cache()
        self.assertEqual(list(bank['frames']),['train','val1','val2'])
        self.assertEqual(self.encoder.calls,['train','val1','val2'])
        self.assertEqual([len(x) for x in bank['frames'].values()],[1,3,2])
        self.assertEqual(len(sentences['features']['val2']),2)
        self.assertFalse(bank['index']['validation_only'])
        self.assertEqual(bank['index']['train_ids'],['train'])
        self.assertEqual(bank['index']['validation_ids'],['val2','val1'])
        self.encoder.calls=[]
        before={p.name:file_record(p) for p in self.output.iterdir() if p.is_file()}
        self.run_cache()
        self.assertEqual(self.encoder.calls,[])
        self.assertEqual(before,{p.name:file_record(p) for p in self.output.iterdir() if p.is_file()})

    def test_json_integer_keys_and_tuple_metadata_resume(self):
        self.encoder.provenance=copy.deepcopy(self.encoder.provenance)
        self.encoder.provenance['model_config']={'id2label':{0:'LABEL_0',1:'LABEL_1'},'size':(224,224)}
        self.run_cache();self.encoder.calls=[]
        before={p.name:file_record(p) for p in self.output.iterdir() if p.is_file()}
        self.run_cache()
        self.assertEqual(self.encoder.calls,[])
        self.assertEqual(before,{p.name:file_record(p) for p in self.output.iterdir() if p.is_file()})
        self.encoder.provenance['model_config']['size']=(256,256)
        with self.assertRaisesRegex(ValueError,'different content'):self.run_cache()

    def test_corrupt_shard_refuses_reuse(self):
        self.run_cache();path=self.output/'shard-00000.npz'
        with path.open('ab') as stream:stream.write(b'corrupt')
        self.encoder.calls=[]
        with self.assertRaisesRegex(ValueError,'corrupt'):self.run_cache()
        self.assertEqual(self.encoder.calls,[])

    def test_partial_run_reuses_completed_shards(self):
        self.run_cache()
        (self.output/'status.json').unlink();(self.output/'index.json').unlink()
        (self.output/'shard-00001.npz.json').unlink();(self.output/'shard-00001.npz').unlink()
        self.encoder.calls=[]
        self.run_cache()
        self.assertEqual(self.encoder.calls,['val1'])

    def test_split_overlap_and_incomplete_coverage_refused(self):
        self.split['train_ids'].append('val1')
        with self.assertRaisesRegex(ValueError,'disjoint'):self.run_cache()
        self.split={'train_ids':['train'],'val_ids':['val1']}
        with self.assertRaisesRegex(ValueError,'disjoint'):self.run_cache()

    def test_source_zip_mismatch_refused(self):
        with patch.object(fresh,'read_manifest',return_value={'wrong':'source'}):
            with self.assertRaisesRegex(ValueError,'Source ZIP'):
                fresh.prepare_training_features(self.manifest,self.split,self.encoder,self.output,train_zip=self.zip)

    def test_generated_cache_resume_keeps_original_bank_immutable(self):
        bank,sentences=self.run_cache();self.encoder.calls=[]
        rows=[{'positive_text':'Train placeholder.','negative_text':'Synthetic symbol.'}]
        # Deliberately return different vectors in the new-text encoder. A
        # positive original caption must still alias its cached sentence bank.
        def changed_sentence_vectors(encoder,plans):
            return [np.tile(np.eye(8,dtype=np.float32)[2],(len(p['segments']),1)) for p in plans]
        with patch.object(fresh,'_encode_plans',changed_sentence_vectors):
            augmented,aug_sentences,meta=fresh.prepare_generated_texts(rows,bank,sentences,self.encoder,self.root/'generated',shard_size=1)
            self.assertEqual(self.encoder.calls,[meta['text_ids']['Synthetic symbol.']])
            self.assertNotIn(meta['text_ids']['Train placeholder.'],self.encoder.calls)
            before={p.name:file_record(p) for p in (self.root/'generated').iterdir() if p.is_file()}
            self.encoder.calls=[]
            again,_,again_meta=fresh.prepare_generated_texts(rows,bank,sentences,self.encoder,self.root/'generated',shard_size=1)
        self.assertEqual(self.encoder.calls,[])
        self.assertEqual(before,{p.name:file_record(p) for p in (self.root/'generated').iterdir() if p.is_file()})
        key=meta['text_ids']['Train placeholder.']
        self.assertTrue(torch.equal(augmented['texts'][key],bank['texts']['train']))
        self.assertEqual(set(bank['texts']),{'train','val1','val2'})
        self.assertEqual(set(sentences['features']),{'train','val1','val2'})
        self.assertEqual(len(augmented['texts']),5)
        self.assertEqual(meta['original_texts_reused'],1)
        self.assertEqual(meta['new_texts_encoded'],1)
        self.assertIs(aug_sentences['features'][key],sentences['features']['train'])
        self.assertTrue(torch.equal(aug_sentences['features'][key],torch.eye(8)[1:2]))
        self.assertTrue(torch.equal(aug_sentences['features'][meta['text_ids']['Synthetic symbol.']],torch.eye(8)[2:3]))
        self.assertEqual(meta['fingerprint'],again_meta['fingerprint'])
        self.assertTrue(torch.equal(augmented['texts'][key],again['texts'][key]))

    def test_generated_cache_rejects_text_change_or_corruption(self):
        bank,sentences=self.run_cache()
        folder=self.root/'generated'
        with patch.object(fresh,'_encode_plans',sentence_vectors):
            fresh.prepare_generated_texts(['Synthetic symbol.'],bank,sentences,self.encoder,folder)
            with self.assertRaisesRegex(ValueError,'different content'):
                fresh.prepare_generated_texts(['Changed symbol.'],bank,sentences,self.encoder,folder)
            with (folder/'shard-00000.npz').open('ab') as stream:stream.write(b'bad')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                fresh.prepare_generated_texts(['Synthetic symbol.'],bank,sentences,self.encoder,folder)


if __name__=='__main__':unittest.main()
