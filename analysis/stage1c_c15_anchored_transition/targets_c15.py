"""Anchored diagnostic objectives only; never imported by production code."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from c15_common import Objectives,sample,check_final,combine_terms

@dataclass
class Term:
    value: object
    status: str

def anchor(logits,pairs):
    check_final(logits,pairs)
    z=logits.double()[:,1]-logits.double()[:,0]
    return .5*(F.softplus(-sample(z,pairs.plus))+F.softplus(sample(z,pairs.minus))).mean(1)

def safe_anchor(logits,pairs):
    if not pairs.valid:return Term(None,pairs.status)
    value=anchor(logits,pairs)
    if not bool(torch.isfinite(value).all()):return Term(None,'nonfinite')
    return Term(value,'valid')

class Targets:
    def __init__(self,pairs,permuted,labels,regions):
        self.pairs,self.permuted=pairs,permuted
        self.original=Objectives('L',pairs,labels=labels,regions=regions)
    def base(self,logits):
        from c15_common import transition
        old=self.original(logits)
        return dict(boundary=old['boundary_ce_w2'],anchor=anchor(logits,self.pairs),transition=old['transition_m1'],
                    permuted=transition(logits,self.permuted),region=old['region_ce'])
    def __call__(self,logits,plan):return combine_terms(self.base(logits),plan)
