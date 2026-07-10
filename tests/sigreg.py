import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import tests._bootstrap as _bootstrap  # noqa: F401

from models.sigreg import SigReg

if __name__ == "__main__":
    sigreg = SigReg()

    collapsed = torch.zeros(4, 128, 192)
    healthy = torch.randn(4, 128, 192)

    collapsed_score = sigreg(collapsed)
    healthy_score = sigreg(healthy)
    print(collapsed_score)
    print(healthy_score)
    assert collapsed_score > healthy_score
