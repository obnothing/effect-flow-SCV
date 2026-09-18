"""Focused structural tests for aligned relevance-polarity models."""

from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from relevance_polarity_query_model import build_relevance_polarity_model


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        self.config={"embedding_dim":8,"gru_hidden_size":8,"gru_layers":1,
                     "bidirectional":True,"attention_heads":4,"num_labels":6}
        self.ids=torch.tensor([[1,2,3,4,0,0],[4,3,2,1,5,6]])
        self.lengths=torch.tensor([4,6]); self.mask=self.ids!=0

    def model(self,variant):
        torch.manual_seed(42)
        return build_relevance_polarity_model(variant,self.config,32,0)

    def test_shapes_attention_and_padding(self):
        for variant in ("B1","B2"):
            model=self.model(variant); out=model(self.ids,self.lengths,self.mask,diagnostics=True)
            self.assertEqual(out["logits"].shape,(2,6))
            self.assertEqual(out["energies"].shape,(2,6,2))
            self.assertEqual(out["alpha_rel"].shape,(2,6,4,6))
            self.assertEqual(out["z_rel"].shape,(2,6,4,4))
            self.assertEqual(out["gates"].shape,(6,2,4,4))
            torch.testing.assert_close(out["alpha_rel"].sum(-1),torch.ones(2,6,4))
            self.assertEqual(float(out["alpha_rel"][0,:,:,4:].detach().abs().max()),0.0)

    def test_parameter_fairness_and_shared_initialization(self):
        b1=self.model("B1"); b2=self.model("B2")
        self.assertEqual(sum(p.numel() for p in b2.parameters())-sum(p.numel() for p in b1.parameters()),6*16)
        state1=b1.state_dict(); state2=b2.state_dict()
        for name,value in state1.items():
            torch.testing.assert_close(value,state2[name],rtol=0,atol=0)

    def test_b2_polarity_cannot_move_token_support(self):
        model=self.model("B2"); base=model(self.ids,self.lengths,self.mask,diagnostics=True)
        swapped=model(self.ids,self.lengths,self.mask,diagnostics=True,swap_polarities=True)
        torch.testing.assert_close(base["alpha_rel"],swapped["alpha_rel"])
        with torch.no_grad(): model.polarity_queries.add_(torch.randn_like(model.polarity_queries))
        changed=model(self.ids,self.lengths,self.mask,diagnostics=True)
        torch.testing.assert_close(base["alpha_rel"],changed["alpha_rel"])
        permuted=model(self.ids,self.lengths,self.mask,diagnostics=True,
                       relevance_permutation=torch.tensor([1,2,3,4,5,0]))
        self.assertFalse(torch.equal(base["alpha_rel"],permuted["alpha_rel"]))

    def test_b1_positive_query_is_locator(self):
        model=self.model("B1"); base=model(self.ids,self.lengths,self.mask,diagnostics=True)
        with torch.no_grad(): model.polarity_queries[:,1].add_(3.0)
        negative_changed=model(self.ids,self.lengths,self.mask,diagnostics=True)
        torch.testing.assert_close(base["alpha_rel"],negative_changed["alpha_rel"])
        with torch.no_grad(): model.polarity_queries[:,0].add_(3.0)
        positive_changed=model(self.ids,self.lengths,self.mask,diagnostics=True)
        self.assertFalse(torch.equal(base["alpha_rel"],positive_changed["alpha_rel"]))

    def test_score_modes_use_same_energies(self):
        model=self.model("B2")
        full=model(self.ids,self.lengths,self.mask)
        positive=model(self.ids,self.lengths,self.mask,score_mode="positive_only")
        negative=model(self.ids,self.lengths,self.mask,score_mode="negative_only")
        torch.testing.assert_close(full["energies"],positive["energies"])
        torch.testing.assert_close(full["energies"],negative["energies"])
        torch.testing.assert_close(full["logits"],positive["logits"]+negative["logits"])


if __name__=="__main__": unittest.main()
