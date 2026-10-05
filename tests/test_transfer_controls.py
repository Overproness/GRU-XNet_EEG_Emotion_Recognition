"""Matched initialization, exact exposure, and source-head gradient isolation."""
import numpy as np
import pandas as pd
import torch

from gruxnet.transfer_controls import FeatureMLP, balanced_batches


def test_separate_heads_start_with_identical_target_logits_and_backbone():
    torch.manual_seed(42)
    shared=FeatureMLP(False).eval()
    torch.manual_seed(42)
    separate=FeatureMLP(True).eval()
    x=torch.randn(8,56)
    datasets=torch.zeros(8,dtype=torch.long)
    torch.testing.assert_close(shared(x,datasets),separate(x,datasets),rtol=0,atol=0)
    for a,b in zip(shared.backbone.parameters(),separate.backbone.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for head in separate.heads:
        torch.testing.assert_close(head.weight,shared.heads[0].weight,rtol=0,atol=0)


def test_source_batch_does_not_backpropagate_into_the_target_head():
    model=FeatureMLP(True)
    loss=torch.nn.functional.cross_entropy(model(torch.randn(8,56),torch.ones(8,dtype=torch.long)),torch.arange(8)%2)
    loss.backward()
    assert model.heads[0].weight.grad.abs().sum()==0
    assert model.heads[1].weight.grad.abs().sum()>0
    assert model.backbone[0].weight.grad.abs().sum()>0
    assert all(p.grad is not None for p in model.parameters())


def test_equal_dataset_class_quotas_and_exact_target_draw_prefixes():
    table=pd.DataFrame({"dataset":[d for d in ["SEEDIV","DEAP","GAMEEMO"] for _ in range(8)],
                        "label":list([0]*5+[1]*3)*3})
    single=balanced_batches(table,["SEEDIV"],4,42,0)
    joint=balanced_batches(table,["SEEDIV","DEAP","GAMEEMO"],12,42,0)
    for batch in joint:
        assert table.iloc[batch].groupby(["dataset","label"]).size().tolist()==[10]*6
    np.testing.assert_array_equal(single[:,:30].reshape(-1),joint[:,:10].reshape(-1))
    np.testing.assert_array_equal(single[:,30:].reshape(-1),joint[:,10:20].reshape(-1))
    assert len(single.reshape(-1))==240
    assert (table.iloc[joint.reshape(-1)].dataset=="SEEDIV").sum()==240
