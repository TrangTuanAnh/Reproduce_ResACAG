"""Bounded two-hop multigraph sampling and train-only node2vec features."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Block:
    output_nodes: np.ndarray
    input_nodes: np.ndarray
    own_index: np.ndarray
    neighbor_index: np.ndarray
    edge_ids: np.ndarray
    mask: np.ndarray


class FlowGraph:
    def __init__(self, endpoints, num_nodes=None):
        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.num_nodes = int(endpoints.max()) + 1 if num_nodes is None else int(num_nodes)
        node = self.endpoints.ravel()
        edges = np.repeat(np.arange(len(endpoints)), 2)
        self.degree = np.bincount(node, minlength=self.num_nodes)
        self.offset = np.concatenate([[0], np.cumsum(self.degree)])
        self.incident = edges[np.argsort(node, kind="stable")]
        self.active_nodes = np.flatnonzero(self.degree)
        # Eq. 2 counts parallel incident edges as distinct flows.
        self.edge_neighbor_mean = float(np.mean(self.degree[endpoints].sum(1) - 2))
        self.node_degree_mean = float(self.degree[self.active_nodes].mean())

    def fanouts(self, strategy="adaptive", cap=(8, 4), fixed=(8, 4)):
        if strategy == "fixed":
            return tuple(fixed)
        value = self.edge_neighbor_mean if strategy == "adaptive" else self.node_degree_mean
        if strategy not in ["adaptive", "degree", "fixed"]:
            raise ValueError(strategy)
        return tuple(max(1, min(c, math.ceil(value))) for c in cap)

    def sample(self, edge_ids, fanouts, rng):
        nodes = np.unique(self.endpoints[edge_ids])
        blocks = []
        for fanout in fanouts:
            sampled = np.zeros((len(nodes), fanout), dtype=np.int64)
            mask = np.zeros_like(sampled, dtype=bool)
            neighbor = np.repeat(nodes[:, None], fanout, axis=1)
            for row, node in enumerate(nodes):
                choices = self.incident[self.offset[node]:self.offset[node + 1]]
                count = min(fanout, len(choices))
                if not count:
                    continue
                chosen = choices if count == len(choices) else rng.choice(choices, count, replace=False)
                sampled[row, :count] = chosen
                mask[row, :count] = True
                pairs = self.endpoints[chosen]
                neighbor[row, :count] = np.where(pairs[:, 0] == node, pairs[:, 1], pairs[:, 0])
            inputs = np.union1d(nodes, neighbor[mask])
            blocks.append(Block(nodes, inputs, np.searchsorted(inputs, nodes),
                                np.searchsorted(inputs, neighbor), sampled, mask))
            nodes = inputs
        return blocks[::-1]


def node2vec(graph, dimension, seed, device, epochs=2, walks_per_node=2,
             walk_length=10, window=2, negatives=3, p=1.0, q=1.0):
    """Streaming skip-gram negative sampling over node2vec random walks.

    p=q=1 is the unbiased node2vec special case. Non-unit p/q use second-order
    transition weights. No labels or evaluation edges enter this function.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    source = nn.Embedding(graph.num_nodes, dimension, sparse=True).to(device)
    context = nn.Embedding(graph.num_nodes, dimension, sparse=True).to(device)
    nn.init.uniform_(source.weight, -0.5 / dimension, 0.5 / dimension)
    nn.init.zeros_(context.weight)
    optimizer = torch.optim.SparseAdam([*source.parameters(), *context.parameters()], lr=0.01)
    if p <= 0 or q <= 0:
        raise ValueError("node2vec p and q must be positive")
    neighbor_sets = None
    if p != 1 or q != 1:
        neighbor_sets = {}
        for node in graph.active_nodes:
            e = graph.incident[graph.offset[node]:graph.offset[node + 1]]
            neighbor_sets[node] = set(graph.endpoints[e].ravel()) - {node}
    losses = []
    for epoch in range(epochs):
        loss_sum, steps = 0.0, 0
        for _ in range(walks_per_node):
            starts = rng.permutation(graph.active_nodes)
            for start in range(0, len(starts), 2048):
                current = starts[start:start + 2048].copy()
                walk = [current.copy()]
                previous = None
                for _ in range(walk_length - 1):
                    if neighbor_sets is None:
                        offset = graph.offset[current] + (rng.random(len(current)) * graph.degree[current]).astype(int)
                        pairs = graph.endpoints[graph.incident[offset]]
                        nxt = np.where(pairs[:, 0] == current, pairs[:, 1], pairs[:, 0])
                    else:
                        nxt = []
                        for i, node in enumerate(current):
                            neighbors = np.array(sorted(neighbor_sets[node]), dtype=np.int64)
                            weights = np.ones(len(neighbors))
                            if previous is not None:
                                weights = np.array([1 / p if n == previous[i] else
                                                    (1 if n in neighbor_sets[previous[i]] else 1 / q) for n in neighbors])
                            nxt.append(rng.choice(neighbors, p=weights / weights.sum()))
                        nxt = np.asarray(nxt)
                    previous, current = current, nxt
                    walk.append(current.copy())
                walks = np.stack(walk, axis=1)
                a, b = [], []
                for distance in range(1, window + 1):
                    a.extend([walks[:, :-distance].ravel(), walks[:, distance:].ravel()])
                    b.extend([walks[:, distance:].ravel(), walks[:, :-distance].ravel()])
                a, b = np.concatenate(a), np.concatenate(b)
                for pos in range(0, len(a), 8192):
                    ai = torch.as_tensor(a[pos:pos + 8192], device=device)
                    bi = torch.as_tensor(b[pos:pos + 8192], device=device)
                    # Uniform negatives over TRAIN active nodes; declared, not paper-specified.
                    neg = torch.as_tensor(rng.choice(graph.active_nodes, (len(ai), negatives)), device=device)
                    emb = source(ai)
                    loss = -F.logsigmoid((emb * context(bi)).sum(-1)).mean()
                    loss -= F.logsigmoid(-(emb[:, None] * context(neg)).sum(-1)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    steps += 1
        losses.append(loss_sum / max(steps, 1))
        print(f"node2vec epoch={epoch + 1} loss={losses[-1]:.5f}", flush=True)
    embeddings = F.normalize(source.weight.detach(), dim=-1).cpu().numpy()
    embeddings[graph.degree == 0] = 0
    return embeddings, losses
