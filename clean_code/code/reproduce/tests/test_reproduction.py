import numpy as np
import pandas as pd
import pytest
import torch

from resacag.data import TrainMinMax, darknet_labels, endpoint_ids, split_indices
from resacag.experiment import metric_dict
from resacag.graph import FlowGraph
from resacag.models import EdgeClassifier, VARIANTS, context_broadcast


def test_endpoint_schema_never_invents_topology():
    with pytest.raises(ValueError, match="Missing real graph endpoints"):
        endpoint_ids(pd.DataFrame({"Dst Port": [80], "Label": ["attack"]}))
    data = pd.DataFrame({"Src IP": ["a", "a"], "Src Port": [1, 1], "Dst IP": ["a", "b"], "Dst Port": [1, 2]})
    result = endpoint_ids(data)
    assert result[0, 0] == result[1, 0]
    assert not set(result[:, 0]) & set(result[:, 1])


def test_label_hierarchy_and_unknown_rejection():
    data = pd.DataFrame({"Label": ["NonVPN", "Tor", "VPN"], "Label.1": ["Chat", "P2P", "Video-streaming"]})
    assert darknet_labels(data).tolist() == [0, 5, 7]
    data.loc[1, "Label.1"] = "unknown"
    with pytest.raises(ValueError):
        darknet_labels(data)


def test_scaler_never_fits_on_heldout_values():
    scaler = TrainMinMax().fit(np.array([[0., 5., np.inf], [2., 5., 3.]]))
    before = scaler.minimum.copy()
    got = scaler.transform(np.array([[100., -100., np.nan], [-10., 999., -np.inf]]))
    assert np.array_equal(before, scaler.minimum)
    assert np.isfinite(got).all()
    assert got.tolist() == [[1., 0., 0.], [0., 0., 0.]]


def test_all_rows_in_split_and_no_overlap():
    y = np.tile(np.arange(7), 101)
    parts = split_indices(y)
    combined = np.concatenate(list(parts.values()))
    assert sorted(combined) == list(range(len(y)))
    for part in parts.values():
        assert set(y[part]) == set(range(7))
    assert abs(len(parts["train"]) / len(y) - 0.5) < 0.01
    assert abs(len(parts["test"]) / len(y) - 0.3) < 0.01


def test_sampling_equation_and_two_hop_reachability():
    # Chain: 0--3--1--4--2, encoded bipartite, with four distinct flow edges.
    ends = np.array([[0, 3], [1, 3], [1, 4], [2, 4]])
    graph = FlowGraph(ends)
    assert graph.edge_neighbor_mean == 1.5
    blocks = graph.sample(np.array([0]), (8, 8), np.random.default_rng(1))
    assert 4 in blocks[0].input_nodes  # genuinely expands beyond first-hop node set
    assert len(blocks) == 2
    for block in blocks:
        for i, node in enumerate(block.output_nodes):
            ids = block.edge_ids[i][block.mask[i]]
            assert len(ids) == len(np.unique(ids))
            assert all(node in ends[e] for e in ids)
            assert np.array_equal(block.input_nodes[block.own_index], block.output_nodes)


def test_disjoint_graph_sampling_does_not_read_heldout_edges():
    endpoints = np.array([[0, 3], [1, 3]])
    graph = FlowGraph(endpoints, num_nodes=6)
    blocks = graph.sample(np.array([0]), (8, 8), np.random.default_rng(7))
    assert all(np.all(b.edge_ids[b.mask] < 2) for b in blocks)


def test_context_broadcast_preserves_mean_and_halves_deviations():
    x = torch.randn(12, 8)
    output = context_broadcast(x)
    assert torch.allclose(x.mean(0), output.mean(0), atol=1e-6)
    assert torch.allclose(output - output.mean(0), (x - x.mean(0)) / 2, atol=1e-6)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_has_finite_end_to_end_encoder_gradients(variant):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    graph = FlowGraph(np.array([[0, 4], [1, 4], [1, 5], [2, 5], [3, 4], [2, 6]]))
    features = torch.rand(6, 7)
    topology = torch.rand(7, 8)
    model = EdgeClassifier(7, 8, 3, dropout=0.0, **VARIANTS[variant])
    logits, _ = model(np.arange(6), graph, features, topology, (3, 2), np.random.default_rng(9))
    assert logits.shape == (6, 3)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1, 2])).backward()
    encoder_parameters = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
    assert all(p.grad is not None for p in encoder_parameters), "An encoder branch was detached from the loss"
    encoder_grad = [p.grad for p in encoder_parameters]
    assert encoder_grad and all(torch.isfinite(g).all() for g in encoder_grad)
    assert sum(float(g.abs().sum()) for g in encoder_grad) > 0


def test_metrics_penalize_missing_classes_not_silently_shrink_macro():
    metrics = metric_dict(np.array([0, 0]), np.array([0, 0]), ["a", "b", "c"])
    assert metrics["f1_macro"] == pytest.approx(1 / 3)
    assert metrics["f1_weighted"] == 1.0


def test_model_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = EdgeClassifier(3, 8, 2, dropout=0).eval()
    graph = FlowGraph(np.array([[0, 2], [1, 2], [1, 3]]))
    feature, topology = torch.rand(3, 3), torch.rand(4, 8)
    a, _ = model(np.arange(3), graph, feature, topology, (2, 2), np.random.default_rng(2))
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    other = EdgeClassifier(3, 8, 2, dropout=0).eval()
    other.load_state_dict(torch.load(path, weights_only=True))
    b, _ = other(np.arange(3), graph, feature, topology, (2, 2), np.random.default_rng(2))
    assert torch.equal(a, b)
