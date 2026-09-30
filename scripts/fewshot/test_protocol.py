"""Leakage and mathematical invariants for Few-shot Pilot-0."""

import json
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from evm_opcode import OPCODES
from polarity_query_model import build_model
from fewshot.pilot_core import query_mixture,random_orthogonal_basis,gradient_residual_basis
from fewshot.build_lovo_splits import read_ids_only,read_rows
from evm_tokenizer import EVMOpcodeTokenizer


class FewshotProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load((ROOT/"configs/fewshot_query_pilot/pilot0.yaml").read_text(encoding="utf-8"))
        cls.data=ROOT/cls.config["dataset"]
        cls.train=read_rows(cls.data/"train.jsonl")
        cls.labels=cls.config["label_names"]
        cls.train_by_id={row["id"]:row for row in cls.train}
        cls.test_ids=set(read_ids_only(cls.data/"test.jsonl"))
        cls.valid_ids=set(read_ids_only(cls.data/"valid.jsonl"))

    def test_all_lovo_sets_are_disjoint_and_test_id_only_safe(self):
        for label in self.config["heldout_labels"]:
            split=json.loads((ROOT/self.config["result_root"]/"splits"/f"{label.replace(' ','_')}.json").read_text())
            bt=set(split["base_train_ids"]);bd=set(split["base_dev_ids"])
            sp=set(split["support_positive_pool_ids"]);sn=set(split["support_negative_pool_ids"]);support=sp|sn
            self.assertFalse(bt&bd);self.assertFalse(bt&support);self.assertFalse(bd&support)
            self.assertFalse((bt|bd|support)&self.valid_ids);self.assertFalse((bt|bd|support)&self.test_ids)
            novel_i=split["novel_label_index"]
            self.assertTrue(all(self.train_by_id[cid]["multi_labels"][novel_i]==0 for cid in bt|bd))
            for episode in split["episodes"]:
                p=episode["positive_ids"];n=episode["negative_ids"]
                self.assertEqual(len(p),self.config["support_k"]);self.assertEqual(len(n),self.config["support_k"])
                self.assertEqual(len(set(p)),len(p));self.assertEqual(len(set(n)),len(n));self.assertFalse(set(p)&set(n))
                self.assertTrue(all(self.train_by_id[cid]["multi_labels"][novel_i]==1 for cid in p))
                self.assertTrue(all(self.train_by_id[cid]["multi_labels"][novel_i]==0 for cid in n))
                self.assertFalse((set(p)|set(n))& (bt|bd|self.valid_ids|self.test_ids))

    def test_default_test_lock(self):
        self.assertFalse(self.config["allow_test"])
        self.assertEqual(self.config["heldout_labels"], ["Reentrancy","Arithmetic","DoS","Bad Randomness"])

    def test_lovo_model_has_only_seven_query_rows(self):
        base=yaml.safe_load((ROOT/self.config["e3_base_config"]).read_text(encoding="utf-8"))
        base["label_names"]=[name for name in base["label_names"] if name!="Reentrancy"]
        base["num_labels"]=7;base["positive_auxiliary_label_multiplier"]=[1.0]*7;base["negative_auxiliary_label_multiplier"]=[1.0]*7
        tok=EVMOpcodeTokenizer.from_vocab_file(ROOT/self.config["vocab_path"])
        model=build_model("P11",base,len(tok),tok.pad_token_id)
        self.assertEqual(model.queries.shape,(7,2,512));self.assertEqual(model.label_scorer.shape,(7,512));self.assertEqual(model.branch_bias.shape,(7,2))

    def test_mixture_alpha_and_residual_geometry(self):
        torch.manual_seed(42);q=torch.randn(7,2,512);ap=torch.randn(7);am=torch.randn(7)
        self.assertAlmostEqual(float(torch.softmax(ap,0).sum()),1.,places=6)
        self.assertAlmostEqual(float(torch.softmax(am,0).sum()),1.,places=6)
        mixed=query_mixture(q,ap,am);self.assertEqual(tuple(mixed.shape),(2,512))
        bp,span=random_orthogonal_basis(q[:,0],4,10)
        self.assertLess(float((span.T@bp).norm()),1e-5);self.assertLessEqual(bp.shape[1],4)
        grads=torch.randn(20,512)
        basis,diag=gradient_residual_basis(q[:,0],grads,4)
        self.assertLess(float((span.T@basis).norm()),1e-4);self.assertLessEqual(basis.shape[1],4)
        self.assertGreaterEqual(diag["ratio"],0);self.assertGreaterEqual(diag["c4"],0)


if __name__=="__main__":
    unittest.main()
