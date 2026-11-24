"""Independent model family. See METHOD.md for non-specified design choices."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def context_broadcast(x):
    """Paper Eq. 10, parameter-free, token axis 0."""
    return 0.5 * (x + x.mean(dim=0, keepdim=True))


class GlobalAttention(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Single head: the paper specifies scaled dot product but not a head count.
        a = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.output(a[0, 0])


class ResACAGLayer(nn.Module):
    def __init__(self, edge_dim, hidden, layer, dropout, internal=True, edge_update="neighbor"):
        super().__init__()
        self.edge_update = edge_update
        self.layer = layer
        self.message = nn.Linear(edge_dim + (hidden if layer > 0 and edge_update == "neighbor" else 0), hidden)
        self.attention = GlobalAttention(hidden, dropout)
        self.gate = nn.Linear(hidden, hidden, bias=False)
        self.internal = internal
        self.update = nn.Linear(hidden * (3 if internal else 2), hidden)

    def forward(self, own, neighbor, edge, mask, topology):
        messages = torch.cat([edge, neighbor], dim=-1) if self.layer > 0 and self.edge_update == "neighbor" else edge
        aggregated = masked_mean(self.message(messages), mask)
        attended = self.attention(aggregated)
        gate = torch.sigmoid(self.gate(attended))
        fused = gate * attended + (1 - gate) * topology
        broadcast = context_broadcast(fused)
        residual = torch.cat([broadcast, attended], dim=-1) if self.internal else broadcast
        return F.relu(self.update(torch.cat([own, residual], dim=-1)))


def masked_mean(x, mask):
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1)


class SageLayer(nn.Module):
    def __init__(self, edge_dim, hidden, layer):
        super().__init__()
        self.layer = layer
        self.update = nn.Linear(edge_dim + hidden * (2 if layer else 1), hidden)

    def forward(self, own, neighbor, edge, mask, topology):
        # Standard edge-aware propagation; no conversion to numpy/detachment.
        message = torch.cat([edge, neighbor], -1) if self.layer else edge
        return F.relu(self.update(torch.cat([own, masked_mean(message, mask)], -1)))


class GATLayer(nn.Module):
    """Edge-aware additive neighborhood attention, followed by endpoint readout."""
    def __init__(self, edge_dim, hidden, dropout):
        super().__init__()
        self.message = nn.Linear(edge_dim + hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(2 * hidden, 1, bias=False)
        self.update = nn.Linear(2 * hidden, hidden)
        self.dropout = dropout

    def forward(self, own, neighbor, edge, mask, topology):
        value = self.message(torch.cat([neighbor, edge], -1))
        query = self.query(own)[:, None].expand_as(value)
        logits = F.leaky_relu(self.score(torch.cat([query, value], -1)).squeeze(-1), 0.2)
        logits = logits.masked_fill(~mask, -1e9)
        weights = F.softmax(logits, dim=1) * mask
        weights = F.dropout(weights, self.dropout, self.training)
        aggregate = (value * weights[..., None]).sum(1)
        return F.elu(self.update(torch.cat([own, aggregate], -1)))


class HopTransformerLayer(nn.Module):
    """Independent Hop-to-Token baseline; not an exact copy of reference [26]."""
    def __init__(self, edge_dim, hidden, dropout):
        super().__init__()
        self.message = nn.Linear(edge_dim + hidden, hidden)
        self.positions = nn.Parameter(torch.zeros(3, hidden))
        self.encoder = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout,
                                                  batch_first=True, norm_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, own, neighbor, edge, mask, topology):
        aggregate = masked_mean(self.message(torch.cat([edge, neighbor], -1)), mask)
        tokens = torch.stack([own, aggregate, topology], dim=1) + self.positions
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.readout(encoded), dim=1)
        return (weights * encoded).sum(1)


class EdgeClassifier(nn.Module):
    def __init__(self, edge_dim, hidden, classes, kind="resacag", dropout=0.5,
                 internal=True, external=True, edge_update="neighbor"):
        super().__init__()
        self.kind = kind
        self.external = external if kind == "resacag" else kind in ["sage_m", "eresgat", "mlp"]
        if kind == "mlp":
            self.mlp = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.ReLU())
            self.classifier = nn.Linear(hidden, classes)
            return
        layers = []
        for level in range(2):
            if kind == "resacag":
                layers.append(ResACAGLayer(edge_dim, hidden, level, dropout, internal, edge_update))
            elif kind in ["sage", "sage_m"]:
                layers.append(SageLayer(edge_dim, hidden, level))
            elif kind in ["gat", "eresgat"]:
                layers.append(GATLayer(edge_dim, hidden, dropout))
            elif kind == "transformer":
                layers.append(HopTransformerLayer(edge_dim, hidden, dropout))
            else:
                raise ValueError(kind)
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(2 * hidden + (edge_dim if self.external else 0), classes)

    def forward(self, edge_ids, graph, features, topology, fanouts, rng):
        device = features.device
        target = torch.as_tensor(edge_ids, device=device)
        if self.kind == "mlp":
            emb = self.mlp(features[target])
            return self.classifier(emb), emb
        blocks = graph.sample(edge_ids, fanouts, rng)
        initial = torch.as_tensor(blocks[0].input_nodes, device=device)
        # E-GraphSAGE/GAT have unit initial features; topology used by ResACAG/Transformer.
        hidden = topology[initial] if self.kind in ["resacag", "transformer"] else torch.ones_like(topology[initial])
        for layer, block in zip(self.layers, blocks):
            own = hidden[torch.as_tensor(block.own_index, device=device)]
            neighbor = hidden[torch.as_tensor(block.neighbor_index, device=device)]
            edge = features[torch.as_tensor(block.edge_ids, device=device)]
            mask = torch.as_tensor(block.mask, device=device)
            topo = topology[torch.as_tensor(block.output_nodes, device=device)]
            hidden = layer(own, neighbor, edge, mask, topo)
        import numpy as np
        position = np.searchsorted(blocks[-1].output_nodes, graph.endpoints[edge_ids])
        endpoints = hidden[torch.as_tensor(position, device=device)].flatten(1)
        emb = torch.cat([endpoints, features[target]], dim=-1) if self.external else endpoints
        return self.classifier(emb), emb


VARIANTS = {
    "resacag": dict(kind="resacag", internal=True, external=True),
    "e_graphsage": dict(kind="sage"),
    "e_graphsage_m": dict(kind="sage_m"),
    "gat": dict(kind="gat"),
    "e_resgat": dict(kind="eresgat"),
    "graph_transformer": dict(kind="transformer"),
    "ablation_a": dict(kind="resacag", internal=False, external=False),
    "ablation_b": dict(kind="resacag", internal=True, external=False),
    "ablation_c": dict(kind="resacag", internal=False, external=True),
    "sampling_s1": dict(kind="resacag", internal=True, external=True),
    "sampling_s2": dict(kind="resacag", internal=True, external=True),
    "mlp": dict(kind="mlp"),
}
