"""Frozen diffusion incumbent and lightweight neural-network components."""
from __future__ import annotations

from ._imports import (
    Any,
    Dict,
    F,
    Optional,
    TORCH_AVAILABLE,
    copy,
    math,
    nn,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 4: the frozen incumbent, its training, and reference-tail calculus
# ============================================================================

class AdamW:
    """Minimal AdamW (decoupled weight decay).  Kept local so that every stage
    uses identical, auditable optimiser arithmetic."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all():
                continue
            m.mul_(self.b1).add_(g * (1.0 - self.b1))
            v.mul_(self.b2).add_(g * g * (1.0 - self.b2))
            if self.wd > 0:
                p.mul_(1.0 - self.lr * self.wd)
            upd = (m / bc1) / (torch.sqrt(v / bc2) + self.eps)
            p.add_(upd * (-self.lr))


def clip_grad_norm(params, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    tn = float(total.item())
    if math.isfinite(tn) and tn > max_norm:
        s = max_norm / (tn + 1e-12)
        for g in grads:
            g.mul_(s)
    return tn


class MLP(nn.Module):
    def __init__(self, din: int, dh: int, dout: int, layers: int = 2, zero_last: bool = False):
        super().__init__()
        dims = [din] + [dh] * layers + [dout]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        if zero_last:
            with torch.no_grad():
                self.lins[-1].weight.mul_(0.0)
                self.lins[-1].bias.mul_(0.0)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
        return x


class TanhDiffusionIncumbent(nn.Module):
    """F_l(H) = a*H0 + (1-a)*[(1-t)*H + t*act(P H W)], shared W with ||W||_2 = w_spec,
    head z = (H A + b) / temperature.  activation='identity' gives the APPNP-type
    linear diffusion family (Table 6)."""

    def __init__(self, in_dim: int, num_classes: int, cfg: _api.RTTConfig, seed: int = 0):
        super().__init__()
        self.T = int(cfg.T)
        self.alpha = float(cfg.alpha)
        self.tau = float(cfg.tau)
        self.w_spec = float(cfg.w_spec)
        self.activation = str(cfg.activation)
        self.enc_dropout = float(cfg.enc_dropout)
        self.head_dropout = float(cfg.head_dropout)
        self.in_dim = int(in_dim)
        self.hidden = int(cfg.hidden)
        self.num_classes = int(num_classes)
        g = torch.Generator().manual_seed(int(seed)) if hasattr(torch, "Generator") else None
        self.enc1 = nn.Linear(in_dim, cfg.enc_hidden)
        self.enc2 = nn.Linear(cfg.enc_hidden, cfg.hidden)
        d = cfg.hidden
        V0 = np.random.default_rng(_api.stable_seed("V", seed)).standard_normal((d, d)).astype(np.float32) / math.sqrt(d)
        self.V = nn.Parameter(torch.from_numpy(V0))
        self.head = nn.Linear(cfg.hidden, num_classes)
        u0 = np.random.default_rng(_api.stable_seed("u", seed)).standard_normal(d).astype(np.float32)
        self.register_buffer("u_pi", torch.from_numpy(u0 / np.linalg.norm(u0)))
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("W_frozen", torch.zeros(d, d))
        self.frozen = False
        self.meta: Dict[str, Any] = {}

    # ---- transport weight ----
    def weight(self):
        if self.frozen:
            return self.W_frozen
        with torch.no_grad():
            v = self.V.t() @ self.u_pi
            v = v / (torch.linalg.norm(v) + 1e-12)
            u = self.V @ v
            u = u / (torch.linalg.norm(u) + 1e-12)
            self.u_pi.copy_(u)
        sigma = (u.unsqueeze(0) @ self.V @ v.unsqueeze(1)).reshape(())
        return self.V * (self.w_spec / sigma)

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def freeze(self) -> dict:
        V64 = self.V.detach().to(torch.float64)
        s = torch.linalg.svdvals(V64)[0]
        W = (V64 * (self.w_spec / s)).to(torch.float32)
        self.W_frozen.copy_(W)
        self.frozen = True
        wn = float(torch.linalg.svdvals(self.W_frozen.to(torch.float64))[0].item())
        self.meta["w_norm_exact"] = wn
        return {"w_norm_exact": wn}

    # ---- pieces ----
    def encode(self, X, train: bool = False):
        h = F.relu(self.enc1(X))
        if train and self.enc_dropout > 0:
            h = F.dropout(h, p=self.enc_dropout, training=True)
        return self.enc2(h)

    def act(self, Z):
        return torch.tanh(Z) if self.activation == "tanh" else Z

    def act_deriv(self, Z):
        if self.activation == "tanh":
            t = torch.tanh(Z)
            return 1.0 - t * t
        return torch.ones_like(Z)

    def transition(self, H, H0, op: _api.GraphOp, W, ctx: Optional[_api.CallContext], j: int,
                   differentiable: bool = False, return_pre: bool = False):
        Z = op.mm(H) @ W
        Hn = self.alpha * H0 + (1.0 - self.alpha) * ((1.0 - self.tau) * H + self.tau * self.act(Z))
        if ctx is not None:
            Hn = ctx.apply_mask(Hn, j + 1, differentiable=differentiable)
        if return_pre:
            return Hn, Z
        return Hn

    def logits(self, H):
        return F.linear(H, self.head.weight, self.head.bias) / self.temperature

    def A_eff(self, dtype=None):
        dt = dtype or self.head.weight.dtype
        return (self.head.weight.to(dt) / self.temperature.to(dt)).t().contiguous()

    def b_eff(self, dtype=None):
        dt = dtype or self.head.weight.dtype
        return self.head.bias.to(dt) / self.temperature.to(dt)

    def gamma_delta(self) -> float:
        """Gamma_Delta = max_{c,c'} ||w_c - w_c'||_2 over effective class vectors."""
        A = self.A_eff(torch.float64)            # d x C
        G = A.t() @ A
        dg = torch.diagonal(G)
        D2 = dg.unsqueeze(0) + dg.unsqueeze(1) - 2.0 * G
        v = float(torch.sqrt(torch.clamp(D2.max(), min=0.0)).item())
        return v * (1.0 + 1e-12) + 1e-15

    def forward_full(self, X, op, ctx=None, train: bool = False):
        H0 = self.encode(X, train=train)
        W = self.weight()
        H = H0
        for j in range(self.T):
            H = self.transition(H, H0, op, W, ctx, j, differentiable=True)
        if train and self.head_dropout > 0:
            H = F.dropout(H, p=self.head_dropout, training=True)
        return F.linear(H, self.head.weight, self.head.bias)

    def deployed_logits(self, X, op, ctx=None):
        """Logits of the deployed (temperature-scaled) model on a full call."""
        return self.forward_full(X, op, ctx=ctx, train=False) / self.temperature

    def export_state(self) -> dict:
        return {"state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "frozen": self.frozen, "meta": self.meta,
                "arch": {"in_dim": self.in_dim, "num_classes": self.num_classes, "hidden": self.hidden,
                         "T": self.T, "alpha": self.alpha, "tau": self.tau, "w_spec": self.w_spec,
                         "activation": self.activation}}


def build_incumbent_from_state(obj: dict, cfg: _api.RTTConfig, device) -> TanhDiffusionIncumbent:
    arch = obj["arch"]
    c2 = copy.deepcopy(cfg)
    c2.hidden = int(arch["hidden"])
    c2.T = int(arch["T"])
    c2.alpha = float(arch["alpha"])
    c2.tau = float(arch["tau"])
    c2.w_spec = float(arch["w_spec"])
    c2.activation = str(arch["activation"])
    sd = obj["state_dict"]
    c2.enc_hidden = int(sd["enc1.weight"].shape[0])
    inc = TanhDiffusionIncumbent(int(arch["in_dim"]), int(arch["num_classes"]), c2)
    inc.load_state_dict(sd)
    inc.frozen = bool(obj.get("frozen", False))
    inc.meta = dict(obj.get("meta", {}))
    inc.to(device)
    inc.eval()
    for p in inc.parameters():
        p.requires_grad_(False)
    return inc


