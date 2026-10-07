import torch
from nazarite.delta import DeltaEncoder
from nazarite.delta.encoder import _center_context
from nazarite.delta.sampling import make_base_reference_grid


def test_delta_supports_both_map_resolutions() -> None:
  encoder = DeltaEncoder(proprio_dim=45)
  proprio = torch.randn(2, 45)
  assert encoder(proprio, torch.randn(2, 26, 16, 3)).shape == (2, 64)
  assert encoder(proprio, torch.randn(2, 41, 25, 3)).shape == (2, 64)


def test_delta_backward_and_auxiliary_shapes() -> None:
  encoder = DeltaEncoder(proprio_dim=45)
  proprio = torch.randn(2, 45, requires_grad=True)
  terrain = torch.randn(2, 26, 16, 3, requires_grad=True)
  output, aux = encoder(proprio, terrain, return_aux=True)
  output.square().mean().backward()
  assert output.shape == (2, 64)
  assert len(aux) == 3
  assert aux[-1]["locations"].shape == (2, 4, 8, 2)
  assert aux[-1]["attention"].shape == (2, 4, 8)
  assert torch.isfinite(output).all()
  assert proprio.grad is not None and torch.isfinite(proprio.grad).all()


def test_delta_attention_cache_is_detached() -> None:
  encoder = DeltaEncoder(proprio_dim=45)
  encoder.enable_attention_cache(True)
  output = encoder(torch.randn(1, 45), torch.randn(1, 16, 26, 3))
  assert output.shape == (1, 64)
  snapshot = encoder.get_attention()
  assert len(snapshot) == 3
  assert snapshot[-1]["locations"].shape == (1, 4, 8, 2)
  assert not snapshot[-1]["locations"].requires_grad


def test_delta_patch_encoding_uses_center_relative_height_only() -> None:
  patch = torch.zeros(1, 1, 1, 5, 3, 3)
  patch[:, :, :, 0] = 0.4
  patch[:, :, :, 1] = -0.2
  patch[:, :, :, 2] = torch.tensor([[1.0, 1.1, 1.2], [0.9, 1.0, 1.3], [0.8, 0.7, 1.4]])
  patch[:, :, :, 3] = 1.0
  patch[:, :, :, 4] = 0.5

  center, context = _center_context(patch, final=False)

  assert context.shape[-1] == 0
  assert center.shape == (1, 1, 1, 9)
  assert center[0, 0, 0, 4] == 0.0
  assert torch.allclose(center.max(), torch.tensor(0.4))
  assert torch.allclose(center.min(), torch.tensor(-0.3))


def test_eight_base_references_cover_both_map_axes() -> None:
  torch.manual_seed(7)
  references = make_base_reference_grid(
    heads=4, samples=8, limits=(1.0, 0.7), perturbation=0.0
  )

  assert references.shape == (4, 8, 2)
  assert torch.unique(references[0, :, 0]).numel() == 4
  assert torch.unique(references[0, :, 1]).numel() == 2
