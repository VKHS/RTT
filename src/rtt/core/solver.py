"""Coupled convex admission branches and barrier Newton optimization."""
from __future__ import annotations

from ._imports import (
    Callable,
    List,
    Optional,
    Tuple,
    dataclass,
    math,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 7: the per-depth convex program (eq. 12) and its barrier solver
# ============================================================================

@dataclass
class SolveResult:
    t: np.ndarray
    f: float
    charge: float
    status: str
    newton_iters: int = 0
    outer_iters: int = 0
    branch: str = ""


def _logsumexp_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class HeavyTerms:
    """Row-summed smooth terms evaluated on the device in float64:
    U [nq, C, nv] first-order terminal-logit change per unit coefficient,
    z [nq, C] logits (last depth), p [nq, C] reference predictions, w [nq]."""

    def __init__(self, U, w, p=None, z=None, beta: float = 1e4):
        self.U = U.to(torch.float64)
        self.w = w.to(torch.float64)
        self.p = None if p is None else p.to(torch.float64)
        self.z = None if z is None else z.to(torch.float64)
        self.beta = float(beta)
        self.lse_z = None if z is None else torch.logsumexp(self.z, dim=1)
        self.dev = self.U.device
        self.calls = 0

    def _y(self, t: np.ndarray):
        tt = torch.from_numpy(np.ascontiguousarray(t, dtype=np.float64)).to(self.dev)
        return torch.einsum("icv,v->ic", self.U, tt)

    def _gh(self, probs, need_hess: bool):
        Up = torch.einsum("icv,ic->iv", self.U, probs)
        g = torch.einsum("i,iv->v", self.w, Up)
        if not need_hess:
            return g, None
        H1 = torch.einsum("icv,ic,icu->vu", self.U, probs * self.w.unsqueeze(1), self.U)
        H2 = torch.einsum("iv,i,iu->vu", Up, self.w, Up)
        return g, H1 - H2

    def env(self, t, order: int = 2):
        """e_beta(t) = sum_i w_i [p_i.y_i + (1/beta) LSE(-beta y_i)] >= sum_i w_i max_c <p_i - e_c, y_i>."""
        self.calls += 1
        y = self._y(t)
        b = self.beta
        sm = torch.logsumexp(-b * y, dim=1) / b
        val = float(torch.dot(self.w, (self.p * y).sum(1) + sm).item())
        if order == 0:
            return val, None, None
        pi = torch.softmax(-b * y, dim=1)
        gp, _ = self._gh(self.p, False)
        gpi, Hpi = self._gh(pi, order >= 2)
        g = gp - gpi
        H = None if Hpi is None else b * Hpi
        return val, g.cpu().numpy(), None if H is None else H.cpu().numpy()

    def cplus(self, t, order: int = 2):
        """C_beta(t) = sum_i w_i [LSE(z+y) - LSE(z) + (1/beta) LSE(-beta y)] >= C+(t)."""
        self.calls += 1
        y = self._y(t)
        b = self.beta
        zy = self.z + y
        val = float(torch.dot(self.w, torch.logsumexp(zy, 1) - self.lse_z + torch.logsumexp(-b * y, 1) / b).item())
        if order == 0:
            return val, None, None
        q = torch.softmax(zy, dim=1)
        pi = torch.softmax(-b * y, dim=1)
        gq, Hq = self._gh(q, order >= 2)
        gpi, Hpi = self._gh(pi, order >= 2)
        g = gq - gpi
        H = None if Hq is None else Hq + b * Hpi
        return val, g.cpu().numpy(), None if H is None else H.cpu().numpy()

    def remainder(self, t, order: int = 2):
        """rho_{T-1}(t) = sum_i w_i KL(p_i || softmax(z_i + y_i)) (exact, label-free)."""
        self.calls += 1
        y = self._y(t)
        zy = self.z + y
        val = float(torch.dot(self.w, torch.logsumexp(zy, 1) - self.lse_z - (self.p * y).sum(1)).item())
        if order == 0:
            return val, None, None
        q = torch.softmax(zy, dim=1)
        gq, Hq = self._gh(q, order >= 2)
        gp, _ = self._gh(self.p, False)
        return val, (gq - gp).cpu().numpy(), None if Hq is None else Hq.cpu().numpy()

    def exact_cplus(self, t) -> float:
        y = self._y(t)
        zy = self.z + y
        return float(torch.dot(self.w, torch.logsumexp(zy, 1) - self.lse_z - y.min(1).values).item())

    def exact_env_linear(self, t) -> float:
        y = self._y(t)
        return float(torch.dot(self.w, (self.p * y).sum(1) - y.min(1).values).item())


def build_block_gauge(G_rows, lam_rows, a_t_k: np.ndarray, n_angles: int):
    """Inscribed-polygon gauge of sigma_k(v) = sum_i Lambda_i ||v_i|| on span(D_k).
    Returns A_k [G, M] with sigma_k(D_k t) <= max_j A_kj . t for every t, or None if
    the block has no positive-price direction."""
    M = G_rows.shape[1]
    Gk = G_rows.sum(0)
    Gk_np = 0.5 * (Gk + Gk.t()).cpu().numpy()
    ev, V = np.linalg.eigh(Gk_np)
    ev = ev[::-1]
    V = V[:, ::-1]
    if not np.isfinite(ev).all() or ev[0] <= 0:
        return None
    r = int(np.sum(ev > 1e-12 * ev[0]))
    if r > 2:
        # sublinear fallback: sigma(sum_m t_m d_m) <= sum_m t_m sigma(d_m) for t >= 0
        diag = torch.sqrt(torch.clamp(torch.diagonal(G_rows, dim1=1, dim2=2), min=0.0))   # [n_k, M]
        fm = (lam_rows.view(-1, 1) * diag).sum(0).cpu().numpy()
        return fm.reshape(1, -1)
    r = max(1, r)
    Vr = V[:, :r]
    lr = ev[:r]
    ax = (Vr.T @ a_t_k) / np.sqrt(lr)
    if r == 2:
        na = float(np.linalg.norm(ax))
        if na > 1e-300:
            e1 = -ax / na
        else:
            C0 = Vr / np.sqrt(lr)[None, :]
            Gam0 = torch.einsum("ma,imn,nb->iab", torch.from_numpy(C0).to(G_rows), G_rows, torch.from_numpy(C0).to(G_rows))
            S0 = (lam_rows.view(-1, 1, 1) * Gam0).sum(0).cpu().numpy()
            w_, U_ = np.linalg.eigh(0.5 * (S0 + S0.T))
            e1 = U_[:, -1]
        e2 = np.array([-e1[1], e1[0]])
        R = np.stack([e1, e2], 1)
    else:
        R = np.ones((1, 1))
    C = (Vr / np.sqrt(lr)[None, :]) @ R                    # M x r
    Ct = torch.from_numpy(np.ascontiguousarray(C)).to(G_rows)
    Gam = torch.einsum("ma,imn,nb->iab", Ct, G_rows, Ct)      # [n_k, r, r]
    Tk = R.T @ (np.sqrt(lr)[:, None] * Vr.T)                 # r x M  (y = Tk t)
    if r == 2:
        th = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
        Y = np.stack([np.cos(th), np.sin(th)], 1)            # [G, 2]
        Yt = torch.from_numpy(Y).to(G_rows)
        quad = torch.einsum("ga,iab,gb->ig", Yt, Gam, Yt)
        f = (lam_rows.view(-1, 1) * torch.sqrt(torch.clamp(quad, min=0.0))).sum(0).cpu().numpy()
        if not np.all(f > 0) or not np.all(np.isfinite(f)):
            return None
        Zv = Y / f[:, None]
        facets = []
        for j in range(len(th)):
            z1 = Zv[j]
            z2 = Zv[(j + 1) % len(th)]
            Mz = np.stack([z1, z2], 0)
            det = np.linalg.det(Mz)
            if abs(det) < 1e-300:
                return None
            facets.append(np.linalg.solve(Mz, np.ones(2)))
        Af = np.stack(facets, 0) @ Tk                          # [G, M]
    else:
        fp = float((lam_rows * torch.sqrt(torch.clamp(Gam[:, 0, 0], min=0.0))).sum().item())
        if not fp > 0:
            return None
        Af = np.stack([fp * Tk[0], -fp * Tk[0]], 0)
    return Af


def barrier_solve(n_x: int, f_oracle: Callable, A_lin: np.ndarray, b_lin: np.ndarray,
                  nl: List[Callable], x0: np.ndarray, tol: float, max_newton: int = 60,
                  mu: float = 20.0, max_outer: int = 40) -> Tuple[np.ndarray, dict]:
    """Primal log-barrier method.  f_oracle(x, order) -> (val, grad, hess);
    each nl(x, order) -> (g, grad, hess) with feasibility g < 0; linear A x < b."""
    x = np.array(x0, dtype=np.float64)
    m = A_lin.shape[0] + len(nl)
    if x.shape != (n_x,) or not np.isfinite(x).all():
        return np.zeros(n_x), {"status": "nonfinite_or_invalid_start", "feasible": False}
    if not np.isfinite(A_lin).all() or not np.isfinite(b_lin).all():
        return x, {"status": "nonfinite_constraints", "feasible": False}
    if not (math.isfinite(tol) and tol > 0 and math.isfinite(mu) and mu > 1):
        raise ValueError("barrier_solve requires finite tol > 0 and mu > 1")
    if np.any(A_lin @ x >= b_lin):
        return x, {"status": "infeasible_start", "feasible": False}
    nlv = [c(x, 0)[0] for c in nl]
    if any(not np.isfinite(v) for v in nlv):
        return x, {"status": "nonfinite_start_nl", "feasible": False}
    if any(v >= 0 for v in nlv):
        return x, {"status": "infeasible_start_nl", "feasible": False}
    last_failure = ""

    def total(x, sb, order):
        nonlocal last_failure
        if not np.isfinite(x).all():
            last_failure = "nonfinite_iterate"
            return None
        r = b_lin - A_lin @ x
        if not np.isfinite(r).all() or np.any(r <= 0):
            last_failure = "lost_feasibility"
            return None
        vals = []
        grads = []
        hess = []
        for c in nl:
            v, g, h = c(x, order)
            if not (np.isfinite(v) and v < 0):
                last_failure = "nonfinite_constraints" if not np.isfinite(v) else "lost_feasibility"
                return None
            vals.append(v)
            grads.append(g)
            hess.append(h)
        fv, fg, fh = f_oracle(x, order)
        tot = sb * fv - np.sum(np.log(r)) - sum(math.log(-v) for v in vals)
        if not np.isfinite(fv) or not np.isfinite(tot):
            last_failure = "nonfinite_objective"
            return None
        if order == 0:
            return tot, None, None, fv
        inv = 1.0 / r
        g = sb * fg + A_lin.T @ inv
        for v, gg in zip(vals, grads):
            g = g + gg / (-v)
        if not np.isfinite(g).all():
            last_failure = "nonfinite_gradient"
            return None
        if order < 2:
            return tot, g, None, fv
        H = sb * fh + (A_lin.T * (inv * inv)) @ A_lin
        for v, gg, hh in zip(vals, grads, hess):
            H = H + np.outer(gg, gg) / (v * v) + hh / (-v)
        if not np.isfinite(H).all():
            last_failure = "nonfinite_hessian"
            return None
        return tot, g, H, fv

    # initial barrier weight: best centrality
    out = total(x, 1.0, 1)
    if out is None:
        return x, {"status": last_failure, "feasible": False, "newton": 0, "outer": 0}
    fv0, fg0, _ = f_oracle(x, 1)
    if not np.isfinite(fv0) or not np.isfinite(fg0).all():
        return x, {"status": "nonfinite_initial_derivatives", "feasible": False, "newton": 0, "outer": 0}
    gphi = out[1] - fg0
    denom = float(fg0 @ fg0)
    sb = 1.0
    if np.isfinite(denom) and denom > 0:
        sb = float(-(fg0 @ gphi) / denom)
    if not np.isfinite(sb):
        sb = 1.0
    sb = min(max(sb, 1.0), 1e10)
    newton = 0
    outer = 0
    status = "max_outer_iterations"
    centered = False
    for outer in range(1, max(0, max_outer) + 1):
        centered = False
        inner_failure = None
        for _ in range(max(0, max_newton)):
            res = total(x, sb, 2)
            if res is None:
                inner_failure = last_failure
                break
            tot, g, H, _ = res
            Hs = 0.5 * (H + H.T)
            reg = 1e-14 * (np.trace(Hs) / max(1, n_x) + 1e-300)
            try:
                dx = -np.linalg.solve(Hs + reg * np.eye(n_x), g)
            except np.linalg.LinAlgError:
                try:
                    dx = -np.linalg.lstsq(Hs + reg * np.eye(n_x), g, rcond=None)[0]
                except np.linalg.LinAlgError:
                    inner_failure = "linear_solve_failed"
                    break
            if not np.isfinite(dx).all():
                inner_failure = "nonfinite_newton_step"
                break
            lam2 = float(-(g @ dx))
            newton += 1
            if not np.isfinite(lam2):
                inner_failure = "nonfinite_newton_decrement"
                break
            if lam2 < 0:
                inner_failure = "non_descent_newton_step"
                break
            if lam2 / 2.0 <= 1e-10:
                centered = True
                break
            step = 1.0
            accepted = False
            for _ls in range(80):
                xn = x + step * dx
                rn = total(xn, sb, 0)
                if rn is not None and rn[0] <= tot + 0.25 * step * float(g @ dx):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                inner_failure = "line_search_failed"
                break
            if np.array_equal(xn, x):
                inner_failure = "stalled_step"
                break
            x = xn
        if inner_failure is not None:
            status = inner_failure
            break
        if not centered:
            status = "max_newton_iterations"
            break
        if m / sb < tol:
            status = "optimal"
            break
        if outer >= max_outer:
            status = "max_outer_iterations"
            break
        sb *= mu
        if not np.isfinite(sb):
            status = "nonfinite_barrier_scale"
            break
    final = total(x, sb, 0) if np.isfinite(sb) else None
    feasible = final is not None
    if status == "optimal" and not feasible:
        status = last_failure or "final_feasibility_failed"
    return x, {"status": status, "newton": newton, "outer": outer,
               "gap": m / sb if centered and np.isfinite(sb) else None,
               "barrier_gap": m / sb if np.isfinite(sb) else None,
               "centered": centered, "feasible": feasible}


def solve_block_program(prob: dict, cfg: _api.RTTConfig) -> SolveResult:
    """Minimise Phi(t) over t >= 0, per-block sum_m t_km <= cap_k and the branch
    budget.  prob keys:
      nv, blocks [(start, stop)], caps [K_a], a_t [nv], S_t [nv,nv], Q_t [nv,nv], kappa,
      branch in {'gauge','env','cplus'}, B, gauge [list of A_k] (gauge),
      heavy (HeavyTerms; env/cplus/last), Qe [nv,nv] (env quadratic),
      last (bool), Qr [nv,nv] (last-depth regulariser, divided by omega), omega."""
    nv = int(prob["nv"])
    B = float(prob["B"])
    zero = SolveResult(t=np.zeros(nv), f=0.0, charge=0.0, status="zero", branch=prob["branch"])
    if nv == 0 or not (B > 0) or not math.isfinite(float(prob["kappa"])):
        zero.status = "zero_budget_or_no_direction"
        return zero
    blocks = prob["blocks"]
    caps = np.asarray(prob["caps"], dtype=np.float64)
    a_t = np.asarray(prob["a_t"], dtype=np.float64)
    S = np.asarray(prob["S_t"], dtype=np.float64)
    Q = np.asarray(prob["Q_t"], dtype=np.float64)
    kappa = float(prob["kappa"])
    heavy: Optional[HeavyTerms] = prob.get("heavy")
    last = bool(prob.get("last", False))
    omega = float(prob.get("omega", cfg.omega))
    Qr = prob.get("Qr")
    branch = prob["branch"]
    K = len(blocks)
    if (not math.isfinite(B) or not math.isfinite(omega) or omega <= 0
            or any(not np.isfinite(v).all() for v in (caps, a_t, S, Q))):
        zero.status = "zero_invalid_program_data"
        return zero
    # exact first-order optimality of the zero step: every other term of Phi is convex,
    # nonnegative and vanishes to first order at t = 0, so Phi(t) >= a.t + kappa ||S^1/2 t|| >= 0
    # for all t >= 0 whenever kappa sqrt(lambda_min(S)) >= ||min(a, 0)||_2
    a_neg = float(np.linalg.norm(np.minimum(a_t, 0.0)))
    if a_neg == 0.0:
        zero.status = "zero_optimal(no_descent_slope)"
        return zero
    if kappa > 0.0:
        lam_min = float(np.linalg.eigvalsh(0.5 * (S + S.T)).min())
        if lam_min > 0.0 and kappa * math.sqrt(lam_min) >= a_neg:
            zero.status = "zero_optimal(robust_first_order)"
            return zero
    eps_n = 1e-12 * (1.0 + math.sqrt(max(float(np.trace(S)), 0.0)))
    use_gauge = branch == "gauge"
    n_x = nv + (K if use_gauge else 0)

    def f_oracle(x, order):
        t = x[:nv]
        q = float(t @ S @ t)
        nr = math.sqrt(q + eps_n * eps_n)
        val = float(a_t @ t) + kappa * nr
        g = a_t + kappa * (S @ t) / nr
        H = kappa * (S / nr - np.outer(S @ t, S @ t) / nr ** 3)
        if last:
            rv, rg, rh = heavy.remainder(t, order)
            val += (rv + 0.5 * float(t @ Qr @ t)) / omega
            if order >= 1:
                g = g + (rg + Qr @ t) / omega
            if order >= 2:
                H = H + (rh + Qr) / omega
        else:
            val += 0.5 * float(t @ Q @ t)
            g = g + Q @ t
            H = H + Q
        if n_x > nv:
            g = np.concatenate([g, np.zeros(n_x - nv)])
            Hf = np.zeros((n_x, n_x))
            Hf[:nv, :nv] = H
            H = Hf
        return val, g, H

    rows = []
    rhs = []
    for i in range(nv):                        # -t <= 0
        r = np.zeros(n_x)
        r[i] = -1.0
        rows.append(r)
        rhs.append(0.0)
    for k, (s0, s1) in enumerate(blocks):      # sum_m t_km <= cap_k
        r = np.zeros(n_x)
        r[s0:s1] = 1.0
        rows.append(r)
        rhs.append(float(caps[k]))
    nl: List[Callable] = []
    if use_gauge:
        for k, (s0, s1) in enumerate(blocks):
            Ak = prob["gauge"][k]
            for j in range(Ak.shape[0]):
                r = np.zeros(n_x)
                r[s0:s1] = Ak[j]
                r[nv + k] = -1.0
                rows.append(r)
                rhs.append(0.0)
        r = np.zeros(n_x)
        r[nv:] = 1.0
        rows.append(r)
        rhs.append(B)
    else:
        if branch == "env":
            Qe = np.asarray(prob["Qe"], dtype=np.float64)

            def c_env(x, order):
                t = x[:nv]
                v, g, H = heavy.env(t, order)
                v = v + 0.5 * float(t @ Qe @ t) - B
                if order >= 1:
                    g = g + Qe @ t
                if order >= 2:
                    H = H + Qe
                return v, g, H
            nl.append(c_env)
        elif branch == "cplus":
            def c_cp(x, order):
                v, g, H = heavy.cplus(x[:nv], order)
                return v - B, g, H
            nl.append(c_cp)
        else:
            raise ValueError(branch)
    A_lin = np.stack(rows, 0)
    b_lin = np.asarray(rhs, dtype=np.float64)

    # strictly feasible start
    t0 = np.zeros(nv)
    for k, (s0, s1) in enumerate(blocks):
        t0[s0:s1] = caps[k] / (2.0 * max(1, s1 - s0))
    theta = 1.0
    x0 = None
    for _ in range(200):
        t = theta * t0
        if use_gauge:
            s = np.array([float(np.max(prob["gauge"][k] @ t[s0:s1])) for k, (s0, s1) in enumerate(blocks)])
            used = float(np.sum(np.maximum(s, 0.0)))
            if used < 0.5 * B:
                slack = (B - used) * 0.25 / max(1, K)
                x0 = np.concatenate([t, np.maximum(s, 0.0) + slack])
                break
        else:
            if all(c(np.concatenate([t]), 0)[0] < -0.5 * B for c in nl):
                x0 = t.copy()
                break
        theta *= 0.5
    if x0 is None:
        zero.status = "no_feasible_start"
        return zero
    x, info = barrier_solve(n_x, f_oracle, A_lin, b_lin, nl, x0, tol=cfg.solver_tol * max(1.0, B),
                            max_newton=cfg.solver_max_newton)
    if np.shape(x) != (n_x,) or not np.isfinite(x).all():
        zero.status = "zero_invalid_solver_iterate(nonfinite_or_shape)"
        zero.newton_iters, zero.outer_iters = info.get("newton", 0), info.get("outer", 0)
        return zero
    t = np.maximum(x[:nv], 0.0)
    fval = f_oracle(np.concatenate([t, x[nv:]]) if n_x > nv else t, 0)[0]
    if use_gauge:
        charge = float(sum(float(np.max(prob["gauge"][k] @ t[s0:s1])) for k, (s0, s1) in enumerate(blocks)))
    elif branch == "env":
        # budget consumption uses the exact (non-smoothed) provisional envelope; the smoothed
        # surrogate only defines the barrier constraint and dominates it (so feasibility holds)
        charge = heavy.exact_env_linear(t) + 0.5 * float(t @ prob["Qe"] @ t)
    else:
        charge = heavy.exact_cplus(t)
    status = info.get("status", "?")
    feasibility_tol = 64.0 * np.finfo(np.float64).eps * max(1.0, B, float(np.max(caps)))
    if (not math.isfinite(float(fval)) or not math.isfinite(float(charge))
            or np.any(x[:nv] < -feasibility_tol)
            or any(float(t[s0:s1].sum()) > caps[k] + feasibility_tol for k, (s0, s1) in enumerate(blocks))
            or charge > B + feasibility_tol):
        return SolveResult(t=np.zeros(nv), f=0.0, charge=0.0, status=f"zero_invalid_solver_result({status})",
                           newton_iters=info.get("newton", 0), outer_iters=info.get("outer", 0), branch=branch)
    if fval > 0:        # the zero step is always feasible and has Phi = 0
        return SolveResult(t=np.zeros(nv), f=0.0, charge=0.0, status=f"zero_better({status})",
                           newton_iters=info.get("newton", 0), outer_iters=info.get("outer", 0), branch=branch)
    return SolveResult(t=t, f=float(fval), charge=float(max(charge, 0.0)), status=status,
                       newton_iters=info.get("newton", 0), outer_iters=info.get("outer", 0), branch=branch)

