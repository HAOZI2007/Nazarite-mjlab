import torch
from nazarite.delta import DeltaEncoder


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
