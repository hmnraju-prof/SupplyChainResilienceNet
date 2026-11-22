import numpy as np
import pytest
import torch
from scrn.synthetic import make_demo
from scrn.model import SupplyChainResilienceNet
from scrn.training import temporal_splits


def test_demo_graph_types_and_labels():
    seq=make_demo(days=18,per_type=3,seed=11)
    assert seq.T==18 and seq.N==18
    assert len(np.unique(seq.node_type))==6
    assert len(np.unique(seq.edge_type))==6
    assert np.all(seq.labels["type"][seq.labels["risk"]==0]==-1)
    assert seq.manifest["source"]=="SYNTHETIC_DEMO_ONLY"


def test_past_only_target_index():
    seq=make_demo(days=18,per_type=3).as_torch()
    x,attrs,active=seq.window(5,5)
    assert torch.equal(x[0],seq.x[0])
    assert torch.equal(x[-1],seq.x[4])
    assert not torch.equal(x[-1],seq.x[5])
    with pytest.raises(IndexError):seq.window(4,5)


def test_graph_backward_pass():
    seq=make_demo(days=18,per_type=3,seed=11).as_torch()
    model=SupplyChainResilienceNet(16,hidden=16,graph_layers=1,graph_heads=4,
                                    temporal_layers=1,temporal_heads=4,feed_forward=32,dropout=.0)
    x,attrs,active=seq.window(7,5)
    out=model(x,seq.edge_index,seq.edge_type,attrs,seq.node_type,active)
    assert out["risk_logit"].shape==(18,)
    assert out["type_logit"].shape==(18,9)
    assert out["severity_logit"].shape==(18,4)
    loss=out["risk_logit"].square().mean()+out["loss"].mean()
    loss.backward()
    assert model.spatial[0].q.weight.grad is not None
    assert torch.isfinite(model.spatial[0].q.weight.grad).all()


def test_temporal_split():
    splits=temporal_splits(221,7,5)
    assert len(splits)==5
    tr,va,te=splits[-1]
    assert (len(tr),len(va),len(te))==(148,33,33)
    assert max(tr)<min(va)<min(te)
    assert te[-1]==220
