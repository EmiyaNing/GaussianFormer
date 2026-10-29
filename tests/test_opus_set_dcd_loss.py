import torch
from loss.opus_set_dcd_loss import density_aware_terms,OPUSSetDCDLoss
from loss.opus_set_loss import OPUSSetLoss,_nearest_indices


def test_matches_official_directional_values_and_gradients_for_unequal_counts():
    torch.manual_seed(928)
    x=torch.randn(7,3,requires_grad=True);y=torch.randn(5,3)
    p=_nearest_indices(x,y,20);g=_nearest_indices(y,x,20)
    a,b,_,_=density_aware_terms(x,y,p,g,alpha=.7,distance_unit=.4,density_power=1.)
    # Independent tensorized transcription of official calc_dcd(non_reg=False).
    d=torch.cdist(x,y).square();pi=d.argmin(1);gi=d.argmin(0)
    wp=(torch.bincount(pi)[pi].float()+1e-6).reciprocal()*(7/5)
    wg=(torch.bincount(gi,minlength=7)[gi].float()+1e-6).reciprocal()*(5/7)
    ra=1-wp*torch.exp(-.7*d[torch.arange(7),pi]/.4**2)
    rb=1-wg*torch.exp(-.7*d[gi,torch.arange(5)]/.4**2)
    torch.testing.assert_close(a,ra);torch.testing.assert_close(b,rb)
    grad=torch.autograd.grad(a.mean()+b.mean(),x,retain_graph=True)[0]
    reference=torch.autograd.grad(ra.mean()+rb.mean(),x)[0]
    torch.testing.assert_close(grad,reference,atol=1e-6,rtol=1e-5)


def test_duplicate_predictions_have_higher_dcd_than_balanced_predictions():
    y=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    def loss(x):
        p=_nearest_indices(x,y,20);g=_nearest_indices(y,x,20)
        a,b,_,_=density_aware_terms(x,y,p,g)
        return (a.mean()+b.mean())/2
    assert loss(torch.zeros(2,3))>loss(y)+.1


def test_classification_and_gradients_are_unchanged():
    torch.manual_seed(7)
    points=torch.randn(1,9,3,requires_grad=True)
    logits=torch.randn(1,9,17,requires_grad=True)
    value=dict(opus_pred_points=[points],opus_pred_logits=[logits],
        metas=dict(occ_label=torch.tensor([[4,17,2,5,17,4]]),occ_xyz=torch.randn(1,6,3),
                   occ_cam_mask=torch.tensor([[True,True,False,True,False,True]])))
    ref,rm=OPUSSetLoss()(value);actual,am=OPUSSetDCDLoss()(value)
    torch.testing.assert_close(rm['loss_cls'],am['loss_cls'])
    torch.testing.assert_close(torch.autograd.grad(ref,logits,retain_graph=True)[0],
                               torch.autograd.grad(actual,logits,retain_graph=True)[0])
    gradient=torch.autograd.grad(actual,points)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_dcd_gradcheck_away_from_assignment_boundaries():
    x=torch.tensor([[.1,.2,.3],[.7,.4,.2]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[.2,.1,.1],[.8,.6,.1]],dtype=torch.float64)
    p=_nearest_indices(x,y,20);g=_nearest_indices(y,x,20)
    def f(v):
        a,b,_,_=density_aware_terms(v,y,p,g)
        return a.mean()+b.mean()
    assert torch.autograd.gradcheck(f,(x,),atol=1e-5)
