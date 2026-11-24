import numpy as np
import torch

from resacag.data import split_indices
from resacag.experiment import evaluate
from resacag.graph import FlowGraph, node2vec
from resacag.models import EdgeClassifier


def test_evaluation_includes_partial_final_batch_and_disables_dropout():
    graph = FlowGraph(np.array([[0, 2], [1, 2], [1, 3], [0, 3], [0, 2]]))
    feature = torch.rand(5, 3)
    topology = torch.rand(4, 8)
    model = EdgeClassifier(3, 8, 2, kind="mlp", dropout=0.8)
    # Data loaders guarantee int64 class indices; numpy's Windows default is int32.
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    a, loss, _, _ = evaluate(model, graph, feature, topology, labels, (2, 2), 2, 42)
    b, _, _, _ = evaluate(model, graph, feature, topology, labels, (2, 2), 2, 42)
    assert len(a) == 5
    assert np.array_equal(a, b)
    assert np.isfinite(loss)
    assert not model.training


def test_node2vec_zeros_nodes_absent_from_training_graph():
    torch.set_num_threads(1)
    graph = FlowGraph(np.array([[0, 2], [1, 2]]), num_nodes=5)
    embeddings, loss = node2vec(graph, 8, 42, "cpu", epochs=1, walks_per_node=1, walk_length=4, window=1)
    assert np.isfinite(embeddings).all()
    assert np.array_equal(embeddings[[3, 4]], np.zeros((2, 8)))
    assert np.linalg.norm(embeddings[0]) > 0
    assert np.isfinite(loss).all()


def test_sequential_diagnostic_preserves_original_order_and_reveals_missing_class():
    y = np.array([0] * 80 + [1] * 20)
    split = split_indices(y, mode="sequential")
    assert split["train"].tolist() == list(range(50))
    assert split["test"].tolist() == list(range(50, 80))
    assert split["validation"].tolist() == list(range(80, 100))
    assert set(y[split["train"]]) == {0}
