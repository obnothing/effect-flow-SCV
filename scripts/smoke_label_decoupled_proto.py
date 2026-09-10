"""Focused tests for label-decoupled prototype objectives."""

import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from label_decoupled_prototype import LabelDecoupledPrototype, pairwise_supervised_contrast  # noqa: E402


def main():
    torch.manual_seed(42)
    representations=torch.randn(12,6,16,requires_grad=True)
    labels=torch.tensor([[1,0,0,1,0,0],[0,1,0,0,0,1],[1,1,0,0,1,0],[0,0,1,0,0,0]]*3,dtype=torch.float32)
    module=LabelDecoupledPrototype(6,16,momentum=0.95,temperature=0.1)
    module.update(representations,labels)
    assert bool(module.initialized.all())
    loss, active=module.prototype_loss(representations,labels)
    pair=pairwise_supervised_contrast(representations,labels,0.1)
    total=loss+pair
    total.backward()
    assert torch.isfinite(total)
    assert representations.grad is not None and float(representations.grad.norm())>0
    assert module.positive_prototypes.grad is None and module.negative_prototypes.grad is None
    assert active==list(range(6))
    result={"status":"ok","active_labels":active,"prototype_loss":float(loss.detach()),"pairwise_loss":float(pair.detach()),"prototype_requires_grad":False,"test_checked":False}
    output=ROOT/"results/light_label/prototype/smoke_report.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
