import torch

from classification.context_head import GatedContextHead


def test_context_head_shapes_and_scalar_gate():
    head = GatedContextHead(feature_dim=8, hidden_dim=4, dropout=0.1, num_classes=3)
    logits, alpha = head(torch.randn(5, 8), torch.randn(5, 8), return_gate=True)
    assert logits.shape == (5, 3)
    assert alpha.shape == (5, 1)
    assert torch.all((alpha >= 0) & (alpha <= 1))

