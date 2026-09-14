"""Behavioral regression tests for polarity queries and training state."""
import copy
import inspect
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from polarity_query_model import PolarityQueryNet, SharedMultiHeadCrossAttention, build_model, loss_terms, encoder_state, tensor_hash
from run_polarity_queries import initialize, load_config, optimizer_for


class Tokenizer:
    pad_token_id=0
    def __len__(self): return 32


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        self.c=dict(load_config(),embedding_dim=8,gru_hidden_size=8)
        self.x=torch.tensor([[1,2,3,4,0,0],[3,2,1,4,5,6]])
        self.lengths=torch.tensor([4,6]); self.mask=self.x!=0
        self.y=torch.tensor([[1.,0.,1.,0.,1.,0.],[0.,1.,0.,1.,0.,1.]])

    def test_shared_encoder_and_capacity_controls(self):
        hashes=[]; models=[]
        for mode in ("P0","P1","P2","P3","P4"):
            model,h=initialize(mode,self.c,Tokenizer()); hashes.append(h); models.append(model)
            self.assertEqual(model(self.x,self.lengths,self.mask)["logits"].shape,(2,6))
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models[2:]}),1)
        self.assertEqual(len({tensor_hash(m.state_dict()) for m in models[2:]}),1)
        for m in models[1:]:
            self.assertEqual(sum(isinstance(x,SharedMultiHeadCrossAttention) for x in m.modules()),1)
            self.assertEqual(m.cross_attention.num_heads,4)
            self.assertNotIn("targets",inspect.signature(m.forward).parameters)

    def test_padding_and_attention_normalization(self):
        m,_=initialize("P4",self.c,Tokenizer()); m.eval()
        out=m(self.x,self.lengths,self.mask,True)
        self.assertEqual(out["attention"].shape,(2,6,2,6))
        self.assertEqual(out["attention"][0,:,:,4:].abs().sum().item(),0)
        torch.testing.assert_close(out["attention"].sum(-1),torch.ones(2,6,2))
        altered=self.x.clone(); altered[0,4:]=9
        torch.testing.assert_close(out["logits"],m(altered,self.lengths,self.mask,True)["logits"])
        with self.assertRaises(ValueError):m(self.x,self.lengths,torch.zeros_like(self.mask))

    def test_label_local_gradients_and_auxiliary_targets(self):
        m,_=initialize("P4",self.c,Tokenizer()); out=m(self.x,self.lengths,self.mask)
        _,_,aux=loss_terms(out,self.y,torch.ones(6),.1)
        expected=.5*(F.binary_cross_entropy_with_logits(out["energies"][...,0],self.y)
                       +F.binary_cross_entropy_with_logits(out["energies"][...,1],1-self.y))
        torch.testing.assert_close(aux,expected)
        total,cls,_=loss_terms(out,self.y,torch.ones(6),0); torch.testing.assert_close(total,cls)
        F.binary_cross_entropy_with_logits(out["logits"][:,2],self.y[:,2]).backward()
        self.assertTrue((m.queries.grad[2].norm(dim=-1)>0).all())
        self.assertEqual(m.queries.grad[[0,1,3,4,5]].abs().sum().item(),0)
        self.assertGreater(m.cross_attention.q_proj.weight.grad.abs().sum().item(),0)

    def test_competition_and_branch_removal(self):
        m,_=initialize("P4",self.c,Tokenizer()); out=m(self.x,self.lengths,self.mask)
        z=out["representations"].detach()
        original=m.score(z)[0]
        with torch.no_grad():m.branch_bias.copy_(torch.randn_like(m.branch_bias))
        original=m.score(z)[0]
        with torch.no_grad():m.branch_bias.copy_(m.branch_bias.flip(1))
        torch.testing.assert_close(m.score(z.flip(2))[0],-original)
        erased=z.clone(); erased[:,:,1]=0
        self.assertFalse(torch.equal(m.score(erased)[0],m.score(z)[0]))

    def test_restore_optimizer_rng_next_step(self):
        model,_=initialize("P4",self.c,Tokenizer()); opt=optimizer_for(model,self.c)
        def step(m,o):
            o.zero_grad(); out=m(self.x,self.lengths,self.mask)
            loss=loss_terms(out,self.y,torch.ones(6),.1)[0]; loss.backward(); o.step()
        step(model,opt)
        state=copy.deepcopy(model.state_dict()); optimizer=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state()
        step(model,opt)
        recovered,_=initialize("P4",self.c,Tokenizer()); other=optimizer_for(recovered,self.c)
        recovered.load_state_dict(state); other.load_state_dict(optimizer); torch.set_rng_state(rng); step(recovered,other)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,recovered.state_dict()[k],rtol=0,atol=0)

    def test_preflight_cannot_change_initialization(self):
        first,h=initialize("P4",self.c,Tokenizer())
        torch.rand(1234)
        second,h2=initialize("P4",self.c,Tokenizer())
        self.assertEqual(h,h2); self.assertEqual(tensor_hash(first.state_dict()),tensor_hash(second.state_dict()))


if __name__=="__main__":unittest.main()
