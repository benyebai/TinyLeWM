import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import tests._bootstrap as _bootstrap  # noqa: F401

from models.predictor import ARPredictor

if __name__ == "__main__":
    model = ARPredictor()
    x = torch.randn(2, 4, 192)
    c = torch.randn(2, 4, 192)

    out = model(x, c)
    print("out shape:", out.shape)
    assert out.shape == (2, 4, 192)
    assert not torch.isnan(out).any()

    n = sum(p.numel() for p in model.parameters())
    print("params", n)
