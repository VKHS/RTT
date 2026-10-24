"""Incumbent fitting, temperature fitting, and clean evaluation."""
from __future__ import annotations

from ._imports import (
    Callable,
    F,
    Optional,
    TORCH_AVAILABLE,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ---------------------------------------------------------------------------
# incumbent training and temperature
# ---------------------------------------------------------------------------

def train_incumbent(inc: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, roles: _api.DataRoles, cfg: _api.RTTConfig,
                    seed: int, max_epochs: Optional[int] = None) -> dict:
    _api.set_global_seeds(seed)
    device = env.device
    X = env.x
    y = env.y
    op = env.clean_op
    tr = torch.from_numpy(roles.pools["inc_train"]).to(device)
    ho = torch.from_numpy(roles.pools["inc_holdout"]).to(device)
    opt = _api.AdamW([p for p in inc.parameters()], lr=cfg.inc_lr, weight_decay=cfg.inc_wd)
    best = {"acc": -1.0, "epoch": -1, "state": None}
    history = []
    patience = 0
    epochs = int(max_epochs or cfg.inc_epochs)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        inc.train()
        opt.zero_grad()
        logits = inc.forward_full(X, op, ctx=None, train=True)
        loss = F.cross_entropy(logits[tr], y[tr], label_smoothing=cfg.inc_label_smoothing)
        loss.backward()
        gn = _api.clip_grad_norm(inc.parameters(), 5.0)
        opt.step()
        if ep % cfg.inc_eval_every == 0 or ep == epochs:
            inc.eval()
            with torch.no_grad():
                lg = inc.forward_full(X, op, ctx=None, train=False)
                acc_h = float((lg[ho].argmax(1) == y[ho]).float().mean().item())
                acc_t = float((lg[tr].argmax(1) == y[tr]).float().mean().item())
            history.append({"epoch": ep, "loss": float(loss.item()), "acc_train": acc_t,
                            "acc_holdout": acc_h, "grad_norm": gn, "elapsed_s": time.time() - t0})
            if acc_h > best["acc"] + 1e-6:
                best = {"acc": acc_h, "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in inc.state_dict().items()}}
                patience = 0
            else:
                patience += cfg.inc_eval_every
            if ep % (cfg.inc_eval_every * 10) == 0:
                _api.log(f"[INCUMBENT] epoch={ep} loss={loss.item():.4f} acc_train={acc_t:.4f} acc_holdout={acc_h:.4f}")
            if patience >= cfg.inc_patience:
                _api.log(f"[INCUMBENT] early stop at epoch {ep}; best epoch {best['epoch']} holdout acc {best['acc']:.4f}")
                break
    if best["state"] is not None:
        inc.load_state_dict(best["state"])
    inc.eval()
    info = inc.freeze()
    return {"history": history, "best_epoch": best["epoch"], "best_holdout_acc": best["acc"], **info}


def golden_section_min(f: Callable[[float], float], lo: float, hi: float, iters: int = 80) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def fit_temperature(inc: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, idx: np.ndarray, cfg: _api.RTTConfig) -> dict:
    inc.temperature.fill_(1.0)
    X = env.x
    lg = inc.forward_full(X, env.clean_op).to(torch.float64)
    ii = torch.from_numpy(idx).to(env.device)
    z = lg[ii]
    yy = env.y[ii]

    def nll(beta: float) -> float:
        zz = z * beta
        return float((torch.logsumexp(zz, 1) - zz.gather(1, yy.view(-1, 1)).squeeze(1)).mean().item())

    nll1 = nll(1.0)
    beta = golden_section_min(nll, 1.0 / cfg.temp_max, 1.0 / cfg.temp_min)
    temp = 1.0 / beta
    inc.temperature.fill_(float(temp))
    return {"temperature": temp, "nll_before": nll1, "nll_after": nll(beta), "n": int(len(idx))}


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def evaluate_incumbent_clean(inc: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, roles: _api.DataRoles) -> dict:
    """Development-only report; final/reserved labels remain uninspected."""
    lg = inc.deployed_logits(env.x, env.clean_op)
    pred = lg.argmax(1)
    out = {}
    g = env.g
    for name, idx in [("official_train", g.split["train"])] + [
            (k, roles.pools[k]) for k in ("inc_train", "inc_holdout", "temp18")]:
        ii = torch.from_numpy(np.asarray(idx)).to(env.device)
        if len(ii) == 0:
            continue
        out[f"acc_{name}"] = float((pred[ii] == env.y[ii]).float().mean().item())
        out[f"nll_{name}"] = float(_api.ce_rows(lg[ii], env.y[ii]).mean().item())
    return out


