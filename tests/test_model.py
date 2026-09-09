import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.model import GPT, RMSNorm, RotaryEmbedding, apply_rotary


def make_model(seed: int = 0, **overrides: int) -> GPT:
    torch.manual_seed(seed)
    cfg = {"vocab_size": 17, "block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 32}
    cfg.update(overrides)
    return GPT(ModelConfig(**cfg)).eval()


def test_forward_shape() -> None:
    model = make_model()
    idx = torch.randint(0, 17, (3, 11))
    assert model(idx).shape == (3, 11, 17)


def test_rejects_sequence_longer_than_block_size() -> None:
    model = make_model(block_size=8)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.zeros(1, 9, dtype=torch.long))


@pytest.mark.parametrize("cut", [1, 5, 10, 15])
def test_causal_mask_no_future_leakage(cut: int) -> None:
    """Changing tokens at positions >= cut must leave logits at positions < cut untouched."""
    model = make_model()
    gen = torch.Generator().manual_seed(cut)
    idx = torch.randint(0, 17, (2, 16), generator=gen)
    perturbed = idx.clone()
    perturbed[:, cut:] = torch.randint(0, 17, (2, 16 - cut), generator=gen)
    assert not torch.equal(idx[:, cut:], perturbed[:, cut:])

    with torch.no_grad():
        a = model(idx)
        b = model(perturbed)
    torch.testing.assert_close(a[:, :cut], b[:, :cut], rtol=0, atol=1e-6)
    # And the perturbation is visible from the cut onwards, so the test isn't vacuous.
    assert not torch.allclose(a[:, cut:], b[:, cut:])


def test_gradient_does_not_flow_from_past_logits_to_future_tokens() -> None:
    """Same property from the other side: d logits[t] / d embedding[s] == 0 for s > t."""
    model = make_model()
    idx = torch.randint(0, 17, (1, 12))
    emb = model.tok_emb(idx).detach().requires_grad_(True)

    x = emb
    mask = model.causal_mask[:12, :12]
    for block in model.blocks:
        x = block(x, model.rope, mask)
    logits = model.lm_head(model.norm(x))

    t = 4
    logits[0, t].sum().backward()
    assert emb.grad is not None
    assert torch.count_nonzero(emb.grad[0, t + 1 :]) == 0
    assert torch.count_nonzero(emb.grad[0, : t + 1]) > 0


def test_tied_embeddings_share_storage_and_count_once() -> None:
    model = make_model()
    assert model.lm_head.weight is model.tok_emb.weight
    all_params = list(model.named_parameters(remove_duplicate=False))
    assert len(all_params) == len(list(model.parameters())) + 1
    emb = model.tok_emb.weight.numel()
    assert model.num_params() == sum(p.numel() for _, p in all_params) - emb


def test_rmsnorm_output_has_unit_rms() -> None:
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 7 + 3
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)


def test_rope_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=16, max_len=64)
    x = torch.randn(1, 1, 64, 16)
    torch.testing.assert_close(rope(x).norm(dim=-1), x.norm(dim=-1))


def test_rope_position_zero_is_identity() -> None:
    rope = RotaryEmbedding(head_dim=8, max_len=4)
    x = torch.randn(1, 1, 1, 8)
    torch.testing.assert_close(rope(x, start=0), x)


@pytest.mark.parametrize(("m", "n", "shift"), [(3, 1, 5), (0, 7, 20), (10, 10, 13)])
def test_rope_dot_product_depends_only_on_relative_position(m: int, n: int, shift: int) -> None:
    head_dim = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_len=64)
    gen = torch.Generator().manual_seed(m * 100 + n)
    q = torch.randn(head_dim, generator=gen)
    k = torch.randn(head_dim, generator=gen)

    def rotated(vec: torch.Tensor, pos: int) -> torch.Tensor:
        return apply_rotary(vec, rope.cos[pos], rope.sin[pos])

    base = rotated(q, m) @ rotated(k, n)
    shifted = rotated(q, m + shift) @ rotated(k, n + shift)
    torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-5)
    # Different relative offset gives a different score.
    other = rotated(q, m + 1) @ rotated(k, n)
    assert not torch.isclose(base, other)


def test_rope_start_offset_matches_slice() -> None:
    rope = RotaryEmbedding(head_dim=8, max_len=16)
    x = torch.randn(1, 2, 10, 8)
    full = rope(x)
    tail = rope(x[:, :, 6:], start=6)
    torch.testing.assert_close(full[:, :, 6:], tail)


def test_rope_rejects_positions_past_table() -> None:
    rope = RotaryEmbedding(head_dim=8, max_len=4)
    with pytest.raises(ValueError, match="beyond the rotary table"):
        rope(torch.randn(1, 1, 3, 8), start=2)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, n_embd=30, n_head=4)
    with pytest.raises(ValueError, match="even"):
        ModelConfig(vocab_size=10, n_embd=12, n_head=4)
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict({"vocab_size": 10, "n_heads": 2})


def test_config_round_trip() -> None:
    cfg = ModelConfig(vocab_size=65, n_layer=3, mlp_hidden=100)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.hidden_dim == 100
    assert ModelConfig(vocab_size=65, n_embd=96, n_head=4).hidden_dim == 256
