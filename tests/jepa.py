import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import tests._bootstrap as _bootstrap  # noqa: F401

from models.jepa import Jepa

if __name__ == "__main__":
    B, T = 2, 4
    pixels = torch.randn(B, T, 3, 224, 224)
    actions = torch.randn(B, T, 30)

    jepa = Jepa()

    emb, act_emb = jepa.encode(pixels, actions)
    print("emb: ", emb.shape)
    print("act_emb: ", act_emb.shape)
    assert emb.shape == (B, T, 192)
    assert act_emb.shape == (B, T, 192)

    next_emb = jepa.predict(emb, act_emb)
    print("next emb: ", next_emb.shape)
    assert next_emb.shape == (B, T, 192)

    n_params = sum(p.numel() for p in jepa.parameters())
    print(n_params)
