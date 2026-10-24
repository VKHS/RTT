"""Actual reference continuations for every paper dataset family, without downloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def models():
    source = Path(__file__).resolve().parents[1] / "src" / "rtt" / "models.py"
    spec = importlib.util.spec_from_file_location("rtt_reference_models_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module


def node_inputs():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(6, 4, generator=generator)
    edge = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return x, {"edge_index": edge}


def molecule_inputs():
    x = torch.tensor([[5, 0, 2, 5, 2, 0, 2, 0, 0],
                      [7, 0, 1, 5, 1, 0, 2, 0, 0],
                      [5, 0, 2, 5, 2, 0, 2, 0, 0],
                      [5, 0, 2, 5, 2, 0, 2, 0, 0],
                      [6, 0, 1, 5, 1, 0, 2, 0, 0],
                      [8, 0, 1, 5, 1, 0, 2, 0, 0]])
    edge = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    return x, {"edge_index": edge, "edge_attr": torch.zeros(8, 3, dtype=torch.long),
               "batch": torch.tensor([0, 0, 0, 1, 1, 1])}


def test_transport_orientation_and_isolated_node(models):
    h = torch.ones(4, 2)
    edge = torch.tensor([[0, 1, 1], [1, 2, 1]])
    P = models.graph_transport({"edge_index": edge}, 4, h).to_dense()
    expected = torch.tensor([[1., 0, 0, 0], [.5, .5, 0, 0], [0, .5, .5, 0], [0, 0, 0, 1.]])
    torch.testing.assert_close(P, expected, rtol=0, atol=0)
    torch.testing.assert_close(P.sum(1), torch.ones(4), rtol=0, atol=0)
    neighbor = models.graph_transport({"edge_index": edge}, 4, h, self_loops=False).to_dense()
    assert torch.equal(neighbor[3], torch.zeros(4))


@pytest.mark.parametrize("family", ["tanh_diffusion", "appnp", "gcnii", "grand", "gcn", "graphsage", "residual_tanh_mlp"])
def test_node_and_tabular_tail_replay_gradients_and_checkpoint(models, family, tmp_path):
    torch.manual_seed(27)
    config = {"T": 3, "hidden": 6, "enc_hidden": 9}
    model = models.build_model(family, 4, 3, config).freeze()
    x, context = node_inputs()
    full = model(x, dict(context))
    work = dict(context)
    state = model.encode(x, work)
    original_keys = set(work)
    for depth in range(model.T):
        state = model.transition(state, depth, work)
        assert state.shape[1] == model.state_channels(depth)
        torch.testing.assert_close(model.tail(state, depth + 1, work), full, rtol=0, atol=0)
        assert set(work) == original_keys, "A reference transition mutated the call context"
    final_state = state.detach().requires_grad_(True)
    gradient, = torch.autograd.grad(model.logits(final_state, work).square().sum(), final_state)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not any(p.requires_grad for p in model.parameters())
    checkpoint = tmp_path / f"{family}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = models.build_model(family, 4, 3, config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(x, dict(context)), full, rtol=0, atol=0)


def test_appnp_is_affine_in_current_state_with_fixed_restart(models):
    torch.manual_seed(4)
    model = models.build_model("appnp", 4, 3, {"hidden": 5, "T": 4}).freeze()
    x, context = node_inputs()
    state = model.encode(x, context)
    first, second = torch.randn_like(state), torch.randn_like(state)
    a, b = 0.3, 0.7
    torch.testing.assert_close(model.tail(a * first + b * second, 1, context),
                               a * model.tail(first, 1, context) + b * model.tail(second, 1, context),
                               atol=3e-7, rtol=3e-6)


def test_gcnii_uses_restart_identity_mapping_and_relu(models):
    model = models.build_model("gcnii", 4, 2, {"hidden": 4, "T": 3, "alpha": .2, "lambda": .5}).eval()
    x, context = node_inputs()
    h0 = model.encode(x, context)
    state = torch.randn_like(h0)
    beta = __import__("math").log1p(.5 / 2)
    base = .8 * (context["transport"].to_dense() @ state) + .2 * h0
    expected = torch.relu((1 - beta) * base + beta * model.transforms[1](base))
    torch.testing.assert_close(model.transition(state, 1, context), expected)
    assert torch.all(expected >= 0)


def test_grand_diffusion_is_attention_free_and_preserves_constant_fields(models):
    model = models.build_model("grand", 4, 2, {"hidden": 5, "T": 3, "reaction": 0., "dt": .2}).eval()
    x, context = node_inputs()
    model.encode(x, context)
    state = torch.tensor([1., -2., 3., -.5, 0.]).expand(6, -1).clone()
    torch.testing.assert_close(model.transition(state, 1, context), state, rtol=0, atol=0)
    changed = state.clone()
    changed[0, 0] += 1
    result = model.transition(changed, 0, context)
    assert result[1, 0] > state[1, 0]
    torch.testing.assert_close(result[5], state[5], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["contractive_diffusion", "centered_gin", "gin", "gin_virtual"])
def test_molecular_family_pooling_permutation_tail_and_checkpoint(models, family, tmp_path):
    torch.manual_seed(8)
    config = {"T": 3, "hidden": 8}
    model = models.build_model(family, 9, 128, config).freeze()
    x, original = molecule_inputs()
    context = dict(original)
    full = model(x, context)
    assert full.shape == (2, 128)
    assert torch.isfinite(full).all()
    work = dict(original)
    h = model.encode(x, work)
    assert h.shape[0] == x.shape[0] + (2 if family == "gin_virtual" else 0)
    old_keys = set(work)
    for depth in range(model.T):
        h = model.transition(h, depth, work)
        torch.testing.assert_close(model.tail(h, depth + 1, work), full, rtol=0, atol=0)
        assert set(work) == old_keys
    permutation = torch.tensor([2, 0, 1, 5, 3, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    perm_context = {"batch": original["batch"][permutation],
                    "edge_index": inverse[original["edge_index"]], "edge_attr": original["edge_attr"]}
    torch.testing.assert_close(model(x[permutation], perm_context), full, atol=3e-7, rtol=3e-6)
    checkpoint = tmp_path / f"{family}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = models.build_model(family, 9, 128, config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(x, dict(original)), full, rtol=0, atol=0)
    state = h.detach().requires_grad_(True)
    gradient, = torch.autograd.grad(model.logits(state, work).square().sum(), state)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_virtual_node_transmits_between_disconnected_atoms(models):
    torch.manual_seed(52)
    model = models.build_model("gin_virtual", 9, 2, {"hidden": 8, "T": 3}).freeze()
    x, _ = molecule_inputs()
    context = {"batch": torch.zeros(6, dtype=torch.long), "edge_index": torch.empty(2, 0, dtype=torch.long),
               "edge_attr": torch.empty(0, 3, dtype=torch.long)}
    h = model.encode(x, context)
    changed = h.clone()
    changed[0] += 4.0
    for depth in range(2):
        h = model.transition(h, depth, context)
        changed = model.transition(changed, depth, context)
    assert not torch.allclose(h[1], changed[1]), "Virtual state did not communicate across atoms"


def test_molecular_contractivity_on_fixed_call(models):
    torch.manual_seed(9)
    model = models.build_model("contractive_diffusion", 9, 128,
                               {"hidden": 7, "T": 3, "w_spec": .85}).freeze()
    x, context = molecule_inputs()
    first = model.encode(x, context)
    second = first + torch.randn_like(first)
    bound = (1 - model.alpha) * ((1 - model.tau) + model.tau * model.w_spec)
    incoming = torch.linalg.vector_norm(first - second, dim=1).max()
    outgoing = torch.linalg.vector_norm(model.transition(first, 0, context) - model.transition(second, 0, context), dim=1).max()
    assert outgoing <= bound * incoming + 1e-6
    assert bound < 1


def test_molecular_features_and_cross_graph_edges_are_validated(models):
    model = models.build_model("gin", 9, 128, {"hidden": 5, "T": 2})
    x, context = molecule_inputs()
    invalid = x.clone()
    invalid[0, 0] = 1000
    with pytest.raises(ValueError, match="Categorical feature"):
        model(invalid, dict(context))
    invalid_context = dict(context)
    invalid_context["edge_index"] = context["edge_index"].clone()
    invalid_context["edge_index"][1, 0] = 5
    with pytest.raises(ValueError, match="different graphs"):
        model(x, invalid_context)


def test_resnet50_actual_four_block_tail_and_state_gradient(models):
    torch.manual_seed(15)
    model = models.build_model("resnet50", 3, 1000, {"pretrained": False}).freeze()
    assert sum(parameter.numel() for parameter in model.backbone.parameters()) == 25_557_032
    assert [len(getattr(model.backbone, f"layer{i}")) for i in range(1, 5)] == [3, 4, 6, 3]
    x = torch.randn(1, 3, 32, 32)
    context = {}
    with torch.no_grad():
        full = model.backbone(x)
        h = model.encode(x, context)
        assert h.shape[1] == 1024
        for depth in range(4):
            h = model.transition(h, depth, context)
            assert h.shape[1] == model.state_channels(depth)
            torch.testing.assert_close(model.tail(h, depth + 1, context), full, atol=0, rtol=0)
        torch.testing.assert_close(model(x, {}), full, atol=0, rtol=0)
    state = h.detach().requires_grad_(True)
    gradient, = torch.autograd.grad(model.logits(state, context).square().sum(), state)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_resnet50_offline_checkpoint_exact_roundtrip(models, tmp_path):
    torch.manual_seed(16)
    original = models.build_model("resnet50", 3, 7, {"pretrained": False}).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = original(x, {})
    checkpoint = tmp_path / "resnet50_state.pt"
    torch.save({"state_dict": original.backbone.state_dict()}, checkpoint)
    loaded = models.build_model("resnet50", 3, 7, {"checkpoint": str(checkpoint)}).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x, {}), expected, rtol=0, atol=0)


def test_resnet50_never_downloads_without_opt_in(models, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    def unexpected_download(*args, **kwargs):
        raise AssertionError("Network access was not authorized")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", unexpected_download)
    with pytest.raises(FileNotFoundError, match="explicitly enable --download"):
        models.build_model("resnet50", 3, 1000, {"pretrained": True, "allow_download": False})


def test_unknown_family_and_bad_horizon_rejected(models):
    with pytest.raises(ValueError, match="Unknown reference family"):
        models.build_model("not_a_paper_model", 4, 2)
    with pytest.raises(ValueError, match="positive integer"):
        models.build_model("appnp", 4, 2, {"T": 0})
    with pytest.raises(ValueError, match="exactly the last four"):
        models.build_model("resnet50", 3, 1000, {"T": 3})
