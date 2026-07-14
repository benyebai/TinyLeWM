"""Minimal LeMario forward-pass example using synthetic data."""

import torch
import torch.nn.functional as F

from models.jepa import Jepa


def main() -> None:
    torch.manual_seed(0)
    model = Jepa().eval()

    # Four normalized RGB observations and four five-frame NES action blocks.
    frames = torch.rand(2, 4, 3, 224, 224) * 2 - 1
    actions = torch.randint(0, 2, (2, 4, 5, 6)).float()

    with torch.inference_mode():
        embeddings, action_embeddings = model.encode(frames, actions)
        predictions = model.predict(embeddings, action_embeddings)
        prediction_loss = F.mse_loss(predictions[:, :-1], embeddings[:, 1:])

    print(f"embeddings:       {tuple(embeddings.shape)}")
    print(f"action embeddings:{tuple(action_embeddings.shape)}")
    print(f"predictions:      {tuple(predictions.shape)}")
    print(f"prediction loss:  {prediction_loss.item():.6f}")


if __name__ == "__main__":
    main()
