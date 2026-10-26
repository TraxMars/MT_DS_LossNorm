"""C15 objective algebra, budget controls and frozen-node contracts."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
import torch
import torch.nn.functional as F
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'analysis/stage1c_c15_anchored_transition'))
from c15_common import (ROOT,b,STAGES,Capture,attenuate,adjacent_pairs,permuted_partners,
                        output_path,gradient_status,fixed_channels,combine_terms,dose_alpha,direction,archived_aggregate)
from targets_c15 import anchor,safe_anchor,Targets
from algebra_c15 import gram,calibrate,conflict,crossing,combination_squared
from objectives import normalize_loss
from objectives_c0 import transition

class C15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(142)
    def fixture(self):
        y=np.zeros((32,32,32),np.int64);y[8:24,8:24,8:24]=1
        p=adjacent_pairs(y,maximum=128,seed_parts=('C15_test',))
        z=torch.randn(1,2,32,32,32,dtype=torch.float64,requires_grad=True)
        return y,p,z
    def test_anchor_endpoint_CE_including_reuse_weights(self):
        y,p,x=self.fixture();yp=torch.ones(len(p.plus),dtype=torch.long);ym=torch.zeros_like(yp)
        a=x[0,:,p.plus[:,0],p.plus[:,1],p.plus[:,2]].T
        c=x[0,:,p.minus[:,0],p.minus[:,1],p.minus[:,2]].T
        expected=.5*(F.cross_entropy(a,yp)+F.cross_entropy(c,ym))
        torch.testing.assert_close(anchor(x,p)[0],expected,rtol=1e-14,atol=1e-14)
    def test_permutation_full_multiset_and_shared_anchor(self):
        y,p,x=self.fixture();q=permuted_partners(p,('C1',0))
        np.testing.assert_array_equal(q.plus,p.plus)
        np.testing.assert_array_equal(np.sort(np.ravel_multi_index(q.minus.T,p.shape)),np.sort(np.ravel_multi_index(p.minus.T,p.shape)))
        torch.testing.assert_close(anchor(x,p),anchor(x,q),rtol=1e-14,atol=1e-14)
    def test_softmax_gauge_both_targets_invariant(self):
        _,p,x=self.fixture();shift=torch.randn_like(x[:,:1])
        for objective in (anchor,transition):torch.testing.assert_close(objective(x,p),objective(x+shift,p),rtol=1e-14,atol=1e-14)
    def test_common_z_shift_changes_anchor_not_transition(self):
        y,p,_=self.fixture();z=torch.tensor(np.where(y,4.,-4.))[None]
        x=torch.stack([torch.zeros_like(z),z],1);changed=x.clone();changed[:,1]+=10
        torch.testing.assert_close(transition(x,p),transition(changed,p),rtol=0,atol=0)
        self.assertGreater(float(anchor(changed,p)),float(anchor(x,p)))
    def test_anchor_correct_direction_in_all_four_class_cases(self):
        y,p,_=self.fixture()
        for plus,minus in ((3.,-2.),(-2.,-3.),(3.,2.),(-3.,2.)):
            z=torch.tensor(np.where(y,plus,minus))[None]
            x=torch.stack([torch.zeros_like(z),z],1).requires_grad_()
            for lam in (0.,.1,1.):
                loss=anchor(x,p)+lam*transition(x,p)
                g,=torch.autograd.grad(loss.sum(),x,retain_graph=True)
                self.assertTrue((g[0,1][tuple(p.plus.T)]<0).all());self.assertTrue((g[0,1][tuple(p.minus.T)]>0).all())
    def test_satisfied_margin_still_wrong_anchor_not_saturated(self):
        y,p,_=self.fixture();z=torch.tensor(np.where(y,15.,5.))[None]
        x=torch.stack([torch.zeros_like(z),z],1).requires_grad_();ga,=torch.autograd.grad(anchor(x,p).sum(),x)
        self.assertTrue((ga[0,1][tuple(p.minus.T)]>.001).all())
        self.assertGreater(float(anchor(x,p).detach()),2.)
    def test_threshold_anchor_not_geometric_distance(self):
        t,ok=crossing(np.array([1000.,4.,-4.]),np.array([-50.,2.,-2.]))
        self.assertAlmostEqual(t[0],1000/1050);self.assertTrue(ok[0]);self.assertFalse(ok[1] or ok[2]);self.assertTrue(np.isnan(t[1:]).all())
    def test_empty_and_nonfinite_explicit_status(self):
        y,p,x=self.fixture();empty=adjacent_pairs(np.zeros_like(y))
        self.assertIsNone(safe_anchor(x,empty).value);self.assertNotEqual(safe_anchor(x,empty).status,'valid')
        self.assertEqual(safe_anchor(x*float('nan'),p).status,'nonfinite')
    def test_gradient_zero_nonfinite_below_threshold_distinct(self):
        self.assertEqual(gradient_status([1.,0.,1e-20,float('nan')]).tolist(),['valid','exact_zero','below_threshold','nonfinite'])
        self.assertEqual(conflict(0.,1.,0.,.3)['status'],'zero_or_below_threshold')
    def test_budget_only_discovery_and_same_lambda_for_perm(self):
        g=gram(dict(anchor=np.array([1.,0]),transition=np.array([3.,4.]),permuted=np.array([-4.,8.]),boundary=np.array([2.,1.]),supervised=np.array([1.,2.])))
        p=calibrate([g,g]);self.assertAlmostEqual(p['lambdas']['r30'],.3/5)
        x={k:torch.tensor([1.]) for k in ('anchor','boundary','transition','permuted','region')};j=combine_terms(x,p)
        torch.testing.assert_close(j['true_r30'],j['perm_r30']);self.assertEqual(p['outer_budget'],.1)
    def test_joint_conflict_distinguishes_increment_and_total(self):
        a=conflict(1.,1.,-1.,.3);self.assertLess(a['dot'],0);self.assertFalse(a['joint_may_increase_anchor'])
        self.assertTrue(conflict(1.,1.,-1.,2.)['joint_may_increase_anchor']);self.assertAlmostEqual(a['critical_lambda'],1.)
    def test_norm_gram_reconstructs_joint(self):
        a=np.array([1.,2.]);t=np.array([-2.,1.]);g=gram(dict(anchor=a,transition=t))
        self.assertAlmostEqual(combination_squared(g,dict(anchor=1.,transition=.3)),float(((a+.3*t)**2).sum()))
    def test_original_lossnorm_joint_reconstruction(self):
        p=torch.tensor([.7,-.2],requires_grad=True);s=(p-1).square().mean();a=F.softplus(p).mean();beta=.05
        gs,=torch.autograd.grad(s,p,retain_graph=True);ga,=torch.autograd.grad(a,p,retain_graph=True)
        joint=s+beta*a;gj,=torch.autograd.grad(normalize_loss(joint),p)
        torch.testing.assert_close(gj,(gs+beta*ga)/joint.detach(),rtol=1e-6,atol=1e-7)
    def test_tiny_independent_lossnorm_can_amplify(self):
        x=torch.tensor(0.,requires_grad=True);tiny=1e-6*F.softplus(x)
        raw,=torch.autograd.grad(tiny,x,retain_graph=True);normal,=torch.autograd.grad(normalize_loss(tiny),x)
        self.assertGreater(float(normal/raw),1e5)
    def test_final_path_node_identity_skip_and_parameter_separation(self):
        m=b.VNetDS(n_channels=1,n_filters=2,normalization='batchnorm',has_dropout=True).eval();p=b.stage_parameters(m)
        before=b.state_snapshot(m);rng=torch.get_rng_state().clone();x=torch.ones(1,1,32,32,32)
        with Capture(m) as c:out,f=m(x,deepsup=True,return_features=True)
        self.assertEqual(len(c.verify(f)),5);grads=torch.autograd.grad(out[-1].square().mean(),list(c.nodes.values())+[p['D3'].parameter],retain_graph=True)
        self.assertNotEqual(grads[2].shape,grads[-1].shape)
        for s in ('D3','D2'):
            with torch.no_grad(),attenuate(m,s,[0],torch.ones(1,1,*c.nodes[s].shape[2:]),[.03],c.nodes[s]) as t:m(x)
            self.assertEqual(t['calls'],1)
        self.assertTrue(b.unchanged(m,before));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(x.grad is None for x in m.parameters()))
    def test_teacher_no_grad_and_L_only_data_contract(self):
        model=torch.nn.Linear(2,2).requires_grad_(False).eval();x=torch.ones(1,2,requires_grad=True);s=(x-model(x).detach()).square().mean()
        torch.autograd.grad(s,x);self.assertTrue(all(p.grad is None for p in model.parameters()))
        import run_c15
        source=Path(run_c15.__file__).read_text();self.assertIn("load_case_and_crops('L'",source)
        self.assertNotIn("load_case_and_crops('U'",source);self.assertNotIn('optimizer.step',source)
    def test_fixed_channels_and_output_guards(self):
        self.assertEqual(set(fixed_channels()),{'E3','D3','D2','D1'});self.assertTrue(all(len(c)==3 for c in fixed_channels().values()))
        self.assertEqual(output_path(ROOT/'runs/unit'),ROOT/'runs/unit')
        for p in (PROJECT/'Results',ROOT/'runs',b.A_RUN):
            with self.assertRaises(ValueError):output_path(p)
    def test_no_alpha_clipping_or_forced_nearzero_sign(self):
        self.assertEqual(dose_alpha(.01,.1)[1],'alpha_cap_exceeded');self.assertEqual(direction(1e-9,1e-9),'indeterminate')
    def test_archived_unreachable_heads_are_not_invalid_measurements(self):
        a=np.array([np.nan,np.nan,np.nan,8.])[:,None,None]
        np.testing.assert_array_equal(archived_aggregate(a,'D1'),[[2.]])
        with self.assertRaises(ValueError):archived_aggregate(a,'E3')
        with self.assertRaises(ValueError):archived_aggregate(np.ones_like(a),'D1')
    def test_reductions_preserve_signed_and_absolute_noncontiguous_products(self):
        from reductions_c15 import feature_statistics,feature_cosine
        from c15_common import cosine_rows
        rng=np.random.default_rng(142);a=rng.normal(size=(7,8,9,4)).transpose(3,0,1,2).astype(np.float32);f=a.copy()*-2;m=rng.random(a.shape[1:]).astype(np.float32)
        a[0]=0;a[1]*=1e-20;s=feature_statistics(a,f,m);expected=a.astype(float)*f.astype(float)*m.astype(float)
        np.testing.assert_allclose(s['inner'],expected.sum((1,2,3)),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(s['abs_gxa'],abs(expected).sum((1,2,3)),atol=1e-12,rtol=1e-12)
        old=cosine_rows(a,f);new=feature_cosine(a,f)
        np.testing.assert_allclose(new[0],old[0],equal_nan=True);np.testing.assert_array_equal(new[3],old[3])
    def test_softplus_threshold_branch_and_extreme_endpoint_CE(self):
        from verify_c15 import numpy_softplus
        x=np.array([-100.,-20.01,0.,20.,20.01,100.])
        np.testing.assert_allclose(numpy_softplus(x),F.softplus(torch.tensor(x)).numpy(),rtol=1e-14,atol=1e-14)
        y,p,_=self.fixture();z=torch.tensor(np.where(y,-20.01,20.01))[None];logits=torch.stack([torch.zeros_like(z),z],1)
        value=anchor(logits,p)[0]
        fg=logits[0,:,p.plus[:,0],p.plus[:,1],p.plus[:,2]].T;bg=logits[0,:,p.minus[:,0],p.minus[:,1],p.minus[:,2]].T
        ce=.5*(F.cross_entropy(fg,torch.ones(len(fg),dtype=torch.long))+F.cross_entropy(bg,torch.zeros(len(bg),dtype=torch.long)))
        self.assertLess(abs(float(value)-float(ce)),2.1e-9)
        self.assertLess(abs(float(value)-float(np.logaddexp(0,20.01))),2.1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
