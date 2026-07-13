import pytest

torch = pytest.importorskip("torch")

from prism.models.action_autoencoder import (
    ActionSegmentAutoencoder,
    ActionSegmentAutoencoderConfig,
    action_segment_autoencoder_loss,
)


def test_action_segment_autoencoder_shapes_and_loss():
    model = ActionSegmentAutoencoder(
        ActionSegmentAutoencoderConfig(action_dim=3, chunk_size=4, latent_dim=5, hidden_dim=16, num_layers=1)
    )
    segments = torch.randn(2, 3, 4, 3)
    segments[..., -1] = (segments[..., -1] > 0).float()
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    output = model(segments)
    loss, metrics = action_segment_autoencoder_loss(
        model,
        segments,
        mask,
        gripper_indices=[-1],
        distance_loss_weight=0.1,
    )

    assert tuple(output.latents.shape) == (2, 3, 5)
    assert tuple(output.reconstruction.shape) == (2, 3, 4, 3)
    assert loss.item() >= 0.0
    assert "segment_ae_rec_loss" in metrics
