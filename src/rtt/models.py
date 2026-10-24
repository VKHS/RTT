"""Incumbent families with explicit, replayable internal-state transitions.

The tensor state after transition ``l`` is the intervention point at depth ``l``.
``encode`` prepares input-dependent constants in a fresh context; ``transition``
does not modify that context. Freezing parameters preserves derivatives with
respect to an intervened state. The generic RTT engine can therefore evaluate
the *actual* reference continuation of every supported model.

Widths, the GRAND reaction discretization, centered-GIN construction and molecular
hyperparameters are explicit reconstruction choices. They are not recovered
author checkpoints. ResNet follows the canonical [3,4,6,3] bottleneck architecture
with stride in conv2 (V1.5), compatible with official TorchVision state dictionaries.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Context = dict[str, Any]
ATOM_CARDINALITIES = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_CARDINALITIES = (5, 6, 2)
RESNET50_WEIGHTS = {
    "IMAGENET1K_V1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "IMAGENET1K_V2": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
}


def _positive_int(config: Mapping[str, Any], name: str, default: int, *aliases: str) -> int:
    value = next((config[k] for k in (name, *aliases) if k in config), default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(config: Mapping[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def graph_transport(context: Context, n: int, reference: Tensor,
                    *, normalization: str = "mean", self_loops: bool = True) -> Tensor:
    """Incoming source-to-target aggregation, with exactly one optional self-loop.

    ``edge_index[0]`` is the source and ``edge_index[1]`` is the destination.
    The caller supplies any required graph symmetrization. Existing diagonal
    entries are removed before adding self-loops. This is a fixed sparse map,
    cached only by encode; it has no label dependence.
    """
    edge = context.get("edge_index")
    if edge is None:
        edge = torch.empty((2, 0), dtype=torch.long, device=reference.device)
    else:
        edge = torch.as_tensor(edge, dtype=torch.long, device=reference.device)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, number_of_edges]")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= n):
        raise ValueError("edge_index contains an out-of-range node")
    source, target = edge
    weight = context.get("edge_weight")
    if weight is None:
        weight = reference.new_ones(source.numel())
    else:
        weight = torch.as_tensor(weight, dtype=reference.dtype, device=reference.device)
        if weight.shape != source.shape or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("edge_weight must be finite, nonnegative and aligned with edge_index")
    keep = source != target
    source, target, weight = source[keep], target[keep], weight[keep]
    if self_loops:
        diagonal = torch.arange(n, device=reference.device)
        source = torch.cat((source, diagonal))
        target = torch.cat((target, diagonal))
        weight = torch.cat((weight, reference.new_ones(n)))
    degree = reference.new_zeros(n).index_add_(0, target, weight).clamp_min_(1.0)
    if normalization == "mean":
        weight = weight / degree[target]
    elif normalization == "symmetric":
        # For undirected input this is the usual D^(-1/2) A D^(-1/2).
        weight = weight / torch.sqrt(degree[target] * degree[source])
    elif normalization != "sum":
        raise ValueError("normalization must be mean, symmetric or sum")
    indices = torch.stack((target, source))
    return torch.sparse_coo_tensor(indices, weight, (n, n), device=reference.device,
                                   dtype=reference.dtype).coalesce()


def graph_pool(state: Tensor, batch: Tensor, num_graphs: int, reduction: str = "mean") -> Tensor:
    pooled = state.new_zeros((num_graphs, state.shape[1])).index_add_(0, batch, state)
    if reduction == "sum":
        return pooled
    if reduction != "mean":
        raise ValueError("pooling must be mean or sum")
    counts = torch.bincount(batch, minlength=num_graphs).to(state).clamp_min_(1)
    return pooled / counts[:, None]


class ReferenceModel(nn.Module):
    """Common state interface; all logits are unnormalized and temperature-scaled."""

    family = "reference"
    task_level = "node"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any], default_T: int):
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        self.configuration = dict(config)
        self.T = _positive_int(config, "T", default_T, "depth", "layers")
        self.hidden = _positive_int(config, "hidden", 128, "hidden_dim")
        self.embedding_dim = self.hidden
        temperature = _finite(config, "temperature", 1.0)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.register_buffer("temperature", torch.tensor(temperature))

    def state_channels(self, depth: int) -> int:
        if not 0 <= depth < self.T:
            raise IndexError(f"depth must be between 0 and {self.T - 1}")
        return self.hidden

    def encode(self, x: Tensor, context: Context) -> Tensor:
        raise NotImplementedError

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        raise NotImplementedError

    def embedding(self, state: Tensor, context: Context) -> Tensor:
        return state

    def logits(self, state: Tensor, context: Context) -> Tensor:
        return self.head(self.embedding(state, context)) / self.temperature

    def forward(self, x: Tensor, context: Context | None = None) -> Tensor:
        if context is None:
            context = {}
        state = self.encode(x, context)
        for depth in range(self.T):
            state = self.transition(state, depth, context)
        return self.logits(state, context)

    def tail(self, state: Tensor, next_depth: int, context: Context) -> Tensor:
        if not 0 <= next_depth <= self.T:
            raise IndexError("next_depth must be between 0 and T")
        for depth in range(next_depth, self.T):
            state = self.transition(state, depth, context)
        return self.logits(state, context)

    def project_parameters(self) -> None:
        """Optional export-time constraint; overridden only by bounded diffusion."""

    def freeze(self) -> "ReferenceModel":
        self.project_parameters()
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class TanhDiffusion(ReferenceModel):
    family = "tanh_diffusion"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any], *, identity: bool = False):
        super().__init__(input_dim, output_dim, config, default_T=8 if identity else 32)
        self.identity_transport = identity
        if identity:
            self.family, self.task_level = "residual_tanh_mlp", "tabular"
        self.alpha = _finite(config, "alpha", 0.1)
        self.tau = _finite(config, "tau", 0.9)
        self.w_spec = _finite(config, "w_spec", 1.6)
        if not (0 <= self.alpha <= 1 and 0 <= self.tau <= 1 and self.w_spec > 0):
            raise ValueError("Require alpha,tau in [0,1] and w_spec>0")
        middle = _positive_int(config, "enc_hidden", 2 * self.hidden)
        self.encoder = nn.Sequential(nn.Linear(input_dim, middle), nn.Tanh(), nn.Linear(middle, self.hidden))
        self.W = nn.Parameter(torch.empty(self.hidden, self.hidden))
        nn.init.orthogonal_(self.W)
        self.head = nn.Linear(self.hidden, output_dim)
        self.project_parameters()

    @torch.no_grad()
    def project_parameters(self) -> None:
        norm = torch.linalg.matrix_norm(self.W.double(), ord=2)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Cannot export a nonfinite or zero transport matrix")
        self.W.mul_(self.w_spec / float(norm))

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = self.encoder(x.to(dtype=self.W.dtype))
        context["h0"] = h0
        if not self.identity_transport:
            context["transport"] = graph_transport(context, x.shape[0], h0)
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        transported = state if self.identity_transport else torch.sparse.mm(context["transport"], state)
        return self.alpha * context["h0"] + (1 - self.alpha) * (
            (1 - self.tau) * state + self.tau * torch.tanh(transported @ self.W))


class APPNP(ReferenceModel):
    family = "appnp"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=10)
        self.alpha = _finite(config, "alpha", 0.1)
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        middle = _positive_int(config, "enc_hidden", self.hidden)
        self.encoder = nn.Sequential(nn.Linear(input_dim, middle), nn.ReLU(), nn.Linear(middle, self.hidden))
        self.head = nn.Linear(self.hidden, output_dim)
        self.normalization = str(config.get("normalization", "mean"))

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = self.encoder(x.to(self.head.weight.dtype))
        context["h0"] = h0
        context["transport"] = graph_transport(context, x.shape[0], h0, normalization=self.normalization)
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        return self.alpha * context["h0"] + (1 - self.alpha) * torch.sparse.mm(context["transport"], state)


class GCNII(ReferenceModel):
    family = "gcnii"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=16)
        self.alpha = _finite(config, "alpha", 0.1)
        self.lam = _finite(config, "lambda", 0.5)
        if not (0 <= self.alpha <= 1 and self.lam > 0):
            raise ValueError("Require alpha in [0,1] and lambda>0")
        self.encoder = nn.Linear(input_dim, self.hidden)
        self.transforms = nn.ModuleList(nn.Linear(self.hidden, self.hidden, bias=False) for _ in range(self.T))
        self.head = nn.Linear(self.hidden, output_dim)

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = F.relu(self.encoder(x.to(self.head.weight.dtype)))
        context["h0"] = h0
        context["transport"] = graph_transport(context, x.shape[0], h0)
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        beta = math.log1p(self.lam / (depth + 1))
        mixed = (1 - self.alpha) * torch.sparse.mm(context["transport"], state) + self.alpha * context["h0"]
        return F.relu((1 - beta) * mixed + beta * self.transforms[depth](mixed))


class GRAND(ReferenceModel):
    """Attention-free forward-Euler reaction diffusion on a committed horizon.

    dH/dt=(P-I)H + reaction*tanh(H W + b). The step, reaction coefficient and
    horizon are saved choices, not a recovery of unspecified breadth settings.
    """
    family = "grand"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=16)
        self.dt = _finite(config, "dt", 0.1)
        self.reaction = _finite(config, "reaction", 0.1)
        if not (0 < self.dt <= 1 and self.reaction >= 0):
            raise ValueError("Require dt in (0,1] and reaction>=0")
        self.encoder = nn.Sequential(nn.Linear(input_dim, self.hidden), nn.Tanh())
        self.reaction_map = nn.Linear(self.hidden, self.hidden)
        self.head = nn.Linear(self.hidden, output_dim)

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = self.encoder(x.to(self.head.weight.dtype))
        context["h0"] = h0
        context["transport"] = graph_transport(context, x.shape[0], h0)
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        flow = torch.sparse.mm(context["transport"], state) - state
        return state + self.dt * (flow + self.reaction * torch.tanh(self.reaction_map(state)))


class GCN(ReferenceModel):
    """T normalized message-passing layers followed by one affine readout."""
    family = "gcn"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=3)
        self.layers = nn.ModuleList(nn.Linear(input_dim if i == 0 else self.hidden, self.hidden)
                                    for i in range(self.T))
        self.head = nn.Linear(self.hidden, output_dim)

    def encode(self, x: Tensor, context: Context) -> Tensor:
        x = x.to(self.head.weight.dtype)
        context["transport"] = graph_transport(context, x.shape[0], x, normalization="symmetric")
        return x

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        return F.relu(self.layers[depth](torch.sparse.mm(context["transport"], state)))


class GraphSAGE(ReferenceModel):
    """Separate learned self and mean-neighbor maps, T=2 by default."""
    family = "graphsage"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=2)
        self.self_maps = nn.ModuleList(nn.Linear(input_dim if i == 0 else self.hidden, self.hidden)
                                      for i in range(self.T))
        self.neighbor_maps = nn.ModuleList(nn.Linear(input_dim if i == 0 else self.hidden, self.hidden, bias=False)
                                          for i in range(self.T))
        self.head = nn.Linear(self.hidden, output_dim)

    def encode(self, x: Tensor, context: Context) -> Tensor:
        x = x.to(self.head.weight.dtype)
        context["transport"] = graph_transport(context, x.shape[0], x, self_loops=False)
        return x

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        neighbor = torch.sparse.mm(context["transport"], state)
        return F.relu(self.self_maps[depth](state) + self.neighbor_maps[depth](neighbor))


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: tuple[int, ...], hidden: int):
        super().__init__()
        self.cardinalities = tuple(int(x) for x in cardinalities)
        self.embeddings = nn.ModuleList(nn.Embedding(count, hidden) for count in self.cardinalities)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.cardinalities):
            raise ValueError(f"Expected {len(self.cardinalities)} categorical columns")
        if values.is_floating_point() and (not torch.isfinite(values).all() or not torch.equal(values, values.floor())):
            raise ValueError("Categorical molecular features must contain finite integer indices")
        indices = values.long()
        for column, count in enumerate(self.cardinalities):
            if indices.shape[0] and (int(indices[:, column].min()) < 0 or int(indices[:, column].max()) >= count):
                raise ValueError(f"Categorical feature {column} outside [0,{count})")
        result = self.embeddings[0](indices[:, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            result = result + embedding(indices[:, column])
        return result


class MolecularReference(ReferenceModel):
    task_level = "graph"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any], default_T: int = 5):
        super().__init__(input_dim, output_dim, config, default_T)
        cards = tuple(config.get("atom_cardinalities", ATOM_CARDINALITIES))
        if input_dim != len(cards):
            raise ValueError("Molecular input_dim must match atom_cardinalities")
        self.atom_encoder = CategoricalEncoder(cards, self.hidden)
        self.bond_cardinalities = tuple(config.get("bond_cardinalities", BOND_CARDINALITIES))
        self.pooling = str(config.get("pooling", "mean"))
        if self.pooling not in ("mean", "sum"):
            raise ValueError("pooling must be mean or sum")
        self.head = nn.Linear(self.hidden, output_dim)

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = self.atom_encoder(x)
        batch = context.get("batch")
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        batch = torch.as_tensor(batch, dtype=torch.long, device=x.device)
        if batch.shape != (x.shape[0],) or not batch.numel() or int(batch.min()) < 0:
            raise ValueError("Molecular batch must assign every atom to a graph")
        num_graphs = int(batch.max()) + 1
        if torch.bincount(batch, minlength=num_graphs).eq(0).any():
            raise ValueError("Molecular graph IDs must be contiguous and contain atoms")
        context["batch"], context["num_graphs"], context["n_real"] = batch, num_graphs, x.shape[0]
        edge = torch.as_tensor(context.get("edge_index", torch.empty((2, 0))), dtype=torch.long, device=x.device)
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= x.shape[0]):
            raise ValueError("Molecular edge endpoint is outside the atom batch")
        if edge.numel() and not torch.equal(batch[edge[0]], batch[edge[1]]):
            raise ValueError("Molecular edges may not connect different graphs")
        context["edge_index"] = edge
        attrs = context.get("edge_attr")
        if attrs is None:
            attrs = torch.zeros((edge.shape[1], len(self.bond_cardinalities)), dtype=torch.long, device=x.device)
        attrs = torch.as_tensor(attrs, device=x.device)
        if attrs.shape != (edge.shape[1], len(self.bond_cardinalities)):
            raise ValueError("edge_attr must align with every molecular bond")
        context["edge_attr"] = attrs
        context["h0"] = h0
        return h0

    def embedding(self, state: Tensor, context: Context) -> Tensor:
        return graph_pool(state[:context["n_real"]], context["batch"], context["num_graphs"], self.pooling)


class GIN(MolecularReference):
    """Bond-aware GIN; virtual nodes are explicit additional tensor rows.

    centered_gin subtracts each graph's current mean before message formation and
    retains a residual from the uncentered state. This centered residual variant
    is an explicit reconstruction because Table16 does not provide its formula.
    """

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any],
                 *, centered: bool = False, virtual: bool = False):
        super().__init__(input_dim, output_dim, config)
        self.centered, self.virtual = centered, virtual
        self.family = "gin_virtual" if virtual else ("centered_gin" if centered else "gin")
        self.epsilon = nn.Parameter(torch.zeros(self.T))
        self.bond_encoders = nn.ModuleList(CategoricalEncoder(self.bond_cardinalities, self.hidden)
                                           for _ in range(self.T))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(self.hidden, 2 * self.hidden), nn.ReLU(),
                                                  nn.Linear(2 * self.hidden, self.hidden)) for _ in range(self.T))
        if virtual:
            self.virtual_initial = nn.Parameter(torch.zeros(1, self.hidden))
            self.virtual_updates = nn.ModuleList(nn.Sequential(nn.Linear(self.hidden, 2 * self.hidden), nn.ReLU(),
                                                              nn.Linear(2 * self.hidden, self.hidden))
                                                 for _ in range(self.T))

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = super().encode(x, context)
        if self.virtual:
            h0 = torch.cat((h0, self.virtual_initial.expand(context["num_graphs"], -1)), dim=0)
        context["h0"] = h0
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        n = context["n_real"]
        atom = state[:n]
        working = atom
        if self.centered:
            center = graph_pool(atom, context["batch"], context["num_graphs"])
            working = atom - center[context["batch"]]
        if self.virtual:
            working = working + state[n:][context["batch"]]
        source, target = context["edge_index"]
        bonds = self.bond_encoders[depth](context["edge_attr"])
        messages = F.relu(working[source] + bonds)
        aggregate = working.new_zeros(working.shape).index_add_(0, target, messages)
        changed = self.updates[depth]((1 + self.epsilon[depth]) * working + aggregate)
        changed = F.relu(changed + atom if self.centered else changed)
        if self.virtual:
            virtual = state[n:]
            gathered = graph_pool(atom, context["batch"], context["num_graphs"], "sum")
            changed_virtual = virtual + self.virtual_updates[depth](gathered + virtual)
            return torch.cat((changed, changed_virtual), dim=0)
        return changed


class MolecularDiffusion(MolecularReference):
    family = "contractive_diffusion"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=8)
        self.alpha = _finite(config, "alpha", 0.1)
        self.tau = _finite(config, "tau", 0.9)
        self.w_spec = _finite(config, "w_spec", 0.9)
        if not (0 <= self.alpha < 1 and 0 < self.tau <= 1 and 0 < self.w_spec < 1):
            raise ValueError("Contractive diffusion requires alpha in [0,1), tau in (0,1], w_spec in (0,1)")
        self.W = nn.Parameter(torch.empty(self.hidden, self.hidden))
        nn.init.orthogonal_(self.W)
        self.bond_encoder = CategoricalEncoder(self.bond_cardinalities, self.hidden)
        self.project_parameters()

    @torch.no_grad()
    def project_parameters(self) -> None:
        norm = torch.linalg.matrix_norm(self.W.double(), ord=2)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid molecular diffusion transport")
        self.W.mul_(self.w_spec / float(norm))

    def encode(self, x: Tensor, context: Context) -> Tensor:
        h0 = super().encode(x, context)
        context["transport"] = graph_transport(context, x.shape[0], h0)
        source, target = context["edge_index"]
        bond = self.bond_encoder(context["edge_attr"])
        degree = torch.bincount(target, minlength=x.shape[0]).to(h0).clamp_min_(1)
        context["bond_bias"] = h0.new_zeros(h0.shape).index_add_(0, target, bond) / degree[:, None]
        return h0

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        pre = torch.sparse.mm(context["transport"], state) @ self.W + context["bond_bias"]
        return self.alpha * context["h0"] + (1 - self.alpha) * (
            (1 - self.tau) * state + self.tau * torch.tanh(pre))


class Bottleneck(nn.Module):
    """ResNet V1.5 bottleneck with TorchVision-compatible parameter names."""
    expansion = 4

    def __init__(self, input_channels: int, width: int, stride: int = 1):
        super().__init__()
        out = 4 * width
        self.conv1 = nn.Conv2d(input_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or input_channels != out:
            self.downsample = nn.Sequential(nn.Conv2d(input_channels, out, 1, stride=stride, bias=False),
                                             nn.BatchNorm2d(out))

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        value = self.relu(self.bn1(self.conv1(x)))
        value = self.relu(self.bn2(self.conv2(value)))
        return self.relu(self.bn3(self.conv3(value)) + residual)


class ResNet50Backbone(nn.Module):
    """Standard 50-layer convolutional network, no optional compiled extensions."""

    def __init__(self, output_dim: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        current = 64
        for stage, (width, count) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3)), start=1):
            blocks = []
            for index in range(count):
                blocks.append(Bottleneck(current, width, 2 if stage > 1 and index == 0 else 1))
                current = 4 * width
            setattr(self, f"layer{stage}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = stage(x)
        return self.fc(self.avgpool(x).flatten(1))


class ResNet50Reference(ReferenceModel):
    family, task_level = "resnet50", "image"

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        super().__init__(input_dim, output_dim, config, default_T=4)
        if self.T != 4 or input_dim != 3:
            raise ValueError("ResNet-50 RTT uses RGB input and exactly the last four bottleneck blocks")
        self.hidden, self.embedding_dim = 2048, 2048
        self.backbone = ResNet50Backbone(output_dim)
        checkpoint = config.get("checkpoint") or config.get("weights_path")
        requested = config.get("weights")
        pretrained = bool(config.get("pretrained", False))
        if checkpoint:
            checkpoint = Path(str(checkpoint)).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"ResNet-50 checkpoint does not exist: {checkpoint}")
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(weights, dict):
                weights = weights.get("state_dict", weights.get("model", weights))
            self._load_backbone_weights(weights)
        elif requested is not None or pretrained:
            weight_name = str(requested or "IMAGENET1K_V1")
            if weight_name == "DEFAULT":
                weight_name = "IMAGENET1K_V2"
            if weight_name not in RESNET50_WEIGHTS:
                raise ValueError(f"Unknown ResNet50 weights {weight_name!r}; choose {tuple(RESNET50_WEIGHTS)}")
            if output_dim != 1000:
                raise ValueError("Official ImageNet pretrained weights require output_dim=1000")
            url = RESNET50_WEIGHTS[weight_name]
            cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
            if not cached.is_file() and not bool(config.get("allow_download", False)):
                raise FileNotFoundError(
                    "Pretrained ResNet-50 weights are required. Supply model.checkpoint/weights_path, "
                    "or explicitly enable --download (model.allow_download). "
                    f"Expected cached file: {cached}")
            weights = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True,
                                                       progress=bool(config.get("download_progress", True)))
            self._load_backbone_weights(weights)
            self.configuration["resolved_weights"] = weight_name

    def _load_backbone_weights(self, weights: Mapping[str, Tensor]) -> None:
        if not isinstance(weights, Mapping):
            raise ValueError("Expected a tensor state dictionary for ResNet-50")
        weights = dict(weights)
        if weights and all(key.startswith("module.") for key in weights):
            weights = {key[len("module."):]: value for key, value in weights.items()}
        if any(key.startswith("backbone.") for key in weights):
            if "temperature" in weights:
                self.temperature.copy_(weights["temperature"])
            weights = {key[len("backbone."):]: value for key, value in weights.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(weights, strict=True)

    def state_channels(self, depth: int) -> int:
        if not 0 <= depth < 4:
            raise IndexError("ResNet-50 depth must be in [0,3]")
        return 1024 if depth == 0 else 2048

    def encode(self, x: Tensor, context: Context) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("ResNet-50 expects [batch,3,height,width] normalized RGB tensors")
        network = self.backbone
        h = network.maxpool(network.relu(network.bn1(network.conv1(x))))
        h = network.layer2(network.layer1(h))
        for block in tuple(network.layer3.children())[:-1]:
            h = block(h)
        context["h0"] = h
        return h

    def transition(self, state: Tensor, depth: int, context: Context) -> Tensor:
        self.state_channels(depth)
        block = self.backbone.layer3[-1] if depth == 0 else self.backbone.layer4[depth - 1]
        return block(state)

    def embedding(self, state: Tensor, context: Context) -> Tensor:
        return self.backbone.avgpool(state).flatten(1)

    def logits(self, state: Tensor, context: Context) -> Tensor:
        return self.backbone.fc(self.embedding(state, context)) / self.temperature


MODEL_NAMES = ("tanh_diffusion", "appnp", "gcnii", "grand", "gcn", "graphsage",
               "residual_tanh_mlp", "contractive_diffusion", "centered_gin", "gin",
               "gin_virtual", "resnet50")


def build_model(name: str, input_dim: int, output_dim: int,
                config: Mapping[str, Any] | None = None) -> ReferenceModel:
    config = {} if config is None else dict(config)
    aliases = {"tanh": "tanh_diffusion", "diffusion": "tanh_diffusion", "mlp": "residual_tanh_mlp",
               "gcn3": "gcn", "gcn_3": "gcn", "graphsage2": "graphsage", "graphsage_2": "graphsage",
               "resnet50_last4": "resnet50", "gin_virtual_node": "gin_virtual"}
    name = aliases.get(str(name).lower(), str(name).lower())
    if name == "tanh_diffusion":
        return TanhDiffusion(input_dim, output_dim, config)
    if name == "residual_tanh_mlp":
        return TanhDiffusion(input_dim, output_dim, config, identity=True)
    if name == "gin":
        return GIN(input_dim, output_dim, config)
    if name == "centered_gin":
        return GIN(input_dim, output_dim, config, centered=True)
    if name == "gin_virtual":
        return GIN(input_dim, output_dim, config, virtual=True)
    constructors = {"appnp": APPNP, "gcnii": GCNII, "grand": GRAND, "gcn": GCN,
                    "graphsage": GraphSAGE, "contractive_diffusion": MolecularDiffusion,
                    "resnet50": ResNet50Reference}
    if name not in constructors:
        raise ValueError(f"Unknown reference family {name!r}; choose one of {MODEL_NAMES}")
    return constructors[name](input_dim, output_dim, config)
