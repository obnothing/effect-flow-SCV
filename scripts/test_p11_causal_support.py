"""Regression test for manual P11 attention interventions."""

from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from diagnose_p11_causal_support import attention_forward
from polarity_query_model import PolarityQueryNet


class Tests(unittest.TestCase):
    def test_manual_forward_and_support_interventions(self):
        torch.manual_seed(42)
        model=PolarityQueryNet("P11",32,0,8,8,6,4,True,1).eval()
        ids=torch.tensor([[1,2,3,4,0],[4,3,2,1,5]]); lengths=torch.tensor([4,5]); mask=ids!=0
        standard=model(ids,lengths,mask,diagnostics=True)
        normal=attention_forward(model,ids,lengths,mask,"normal",True)
        torch.testing.assert_close(standard["logits"],normal["logits"])
        torch.testing.assert_close(standard["energies"],normal["energies"])
        average=attention_forward(model,ids,lengths,mask,"average")
        swapped=attention_forward(model,ids,lengths,mask,"swap")
        self.assertEqual(average["logits"].shape,(2,6)); self.assertEqual(swapped["logits"].shape,(2,6))
        self.assertFalse(torch.equal(normal["logits"],swapped["logits"]))


if __name__=="__main__": unittest.main()
