"""Regression-rate, betting, finite-population, and comparison statistics."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    math,
    np,
)
from . import api as _api

# ============================================================================
# Section 14: statistical assurance (Sec. 3.5, Theorems 3-4, Appendix D)
# ============================================================================

def _scipy_stats():
    try:
        import scipy.stats as st  # type: ignore
        return st
    except Exception as e:  # pragma: no cover
        raise _api.RTTFailure("PREREQUISITE", f"scipy is required for the statistics: {e}")


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1-delta) Clopper-Pearson upper bound (eq. 21); u = 1 when k = n."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    st = _scipy_stats()
    return float(st.beta.ppf(1.0 - delta, k + 1, n - k))


def paired_t_interval(x: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    st = _scipy_stats()
    se = float(x.std(ddof=1) / math.sqrt(n))
    tq = float(st.t.ppf(0.5 + level / 2.0, n - 1))
    return m, m - tq * se, m + tq * se


def one_sided_p_positive(x: np.ndarray) -> float:
    """p-value of H0: E[x] <= 0 against E[x] > 0 (paired t)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd == 0:
        return 0.0 if x.mean() > 0 else 1.0
    st = _scipy_stats()
    tval = float(x.mean() / (sd / math.sqrt(n)))
    return float(1.0 - st.t.cdf(tval, n - 1))


def holm_adjust(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted([(p, k) for k, p in pvals.items() if p == p], key=lambda z: z[0])
    m = len(items)
    out = {}
    running = 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    for k, p in pvals.items():
        if k not in out:
            out[k] = float("nan")
    return out


def _predictable_var(d: np.ndarray, prior_var: float) -> np.ndarray:
    """sigma^2_{j-1} using data before j (predictable), with a prior pseudo-observation."""
    n = len(d)
    mu = np.zeros(n)
    var = np.zeros(n)
    s, s2, cnt = 0.0, prior_var, 1.0
    m_prev = 0.0
    for j in range(n):
        var[j] = s2 / cnt
        mu[j] = m_prev
        s += d[j]
        m_prev = s / (j + 1)
        s2 += (d[j] - mu[j]) ** 2
        cnt += 1.0
    return var


def betting_ucb(d: Sequence[float], H_plus: float, delta: float, mode: str = "tuned",
                return_path: bool = False, path_points: int = 64, prior_var: Optional[float] = None):
    """Upper confidence bound on E[d] from d_j <= H+ only (Theorem 3(iii), Lemma 8).
    Bets lambda_j(m) = min{c_j, 3/(4(H+ - m))} with predictable c_j not depending on m;
    UCB_t = sup{m < H+ : W_t(m) < 1/delta}.  Modes: 'tuned' (registered bets tuned to
    N), 'horizon_free', 'kelly' (grid mixture), 'hedged' (tuned bets at delta/2)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if not math.isfinite(H_plus) or H_plus < 0 or not 0.0 < delta < 1.0:
        raise ValueError("betting bound requires finite H_plus >= 0 and 0 < delta < 1")
    if not np.isfinite(d).all():
        raise _api.RTTFailure("SCIENTIFIC", "betting bound requires a complete finite audit sequence")
    if n == 0:
        return (float(H_plus), []) if return_path else float(H_plus)
    if np.any(d > H_plus * (1 + 1e-9) + 1e-12):
        raise _api.RTTFailure("SCIENTIFIC", f"betting bound requires d <= H+ = {H_plus}; max d = {d.max()}")
    d = np.minimum(d, H_plus)
    lvl = delta / 2.0 if mode == "hedged" else delta
    thr = math.log(1.0 / lvl)
    # Registered before the evaluated sequence: never use its min/std/range.
    # A caller-supplied value must likewise be fixed on independent development
    # data. The default depends only on the deterministic contract budget.
    prior = max(H_plus * H_plus / 4.0, 1e-18) if prior_var is None else float(prior_var)
    if not math.isfinite(prior) or prior <= 0.0:
        raise ValueError("betting variance prior must be finite and positive")
    var = _predictable_var(d, prior)
    j = np.arange(1, n + 1, dtype=np.float64)
    if mode == "horizon_free":
        c = np.sqrt(2.0 * thr / (np.maximum(var, 1e-300) * j * np.log1p(j)))
    elif mode in ("tuned", "hedged"):
        c = np.sqrt(2.0 * thr / (np.maximum(var, 1e-300) * n))
    elif mode == "kelly":
        c = None
    else:
        raise ValueError(mode)
    cmax = 1e12

    def logwealth(m: float, upto: int) -> float:
        cap = 0.75 / max(H_plus - m, 1e-300)
        dd = d[:upto]
        if c is None:
            grid = np.geomspace(1e-3, 1.0, 24) * cap
            fac = 1.0 + np.outer(grid, (m - dd))
            fac = np.maximum(fac, 1e-300)
            lw = np.log(fac).sum(1)
            mx = lw.max()
            return float(mx + math.log(np.mean(np.exp(lw - mx))))
        lam = np.minimum(np.minimum(c[:upto], cmax), cap)
        return float(np.log(np.maximum(1.0 + lam * (m - dd), 1e-300)).sum())

    def ucb_at(upto: int) -> float:
        hi = H_plus - 1e-15 * max(1.0, abs(H_plus))
        if logwealth(hi, upto) < thr:
            return float(H_plus)
        lo = float(np.min(d[:upto])) - 1.0 * (H_plus - float(np.min(d[:upto])) + 1e-12)
        k = 0
        while logwealth(lo, upto) >= thr and k < 200:
            lo = lo - 2.0 * (H_plus - lo)
            k += 1
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if logwealth(mid, upto) >= thr:
                hi = mid
            else:
                lo = mid
        return float(hi)

    ucb = ucb_at(n)
    if not return_path:
        return ucb
    pts = sorted(set(int(x) for x in np.unique(np.geomspace(min(8, n), n, min(path_points, n)).astype(int)) if 1 <= x <= n))
    path = [(t, ucb_at(t)) for t in pts]
    return ucb, path


def normal_ucb(d: Sequence[float], delta: float) -> float:
    d = np.asarray(d, dtype=np.float64)
    st = _scipy_stats()
    return float(d.mean() + st.norm.ppf(1 - delta) * d.std(ddof=1) / math.sqrt(len(d)))


def hypergeom_node_bound(n_N: int, n_cal: int, k: int, delta: float) -> Tuple[float, int]:
    """Theorem 4(ii): F-hat = max{F' : P_Hyp(n_N, F', n_cal)(X <= k) >= delta}; the flip
    rate on the uninspected nodes is at most (F-hat - k)/(n_N - n_cal) w.p. >= 1-delta."""
    st = _scipy_stats()
    if n_cal >= n_N:
        return (0.0, k)
    Fh = k
    hi = n_N - (n_cal - k)
    lo_ok = k
    lo, hi_ = k, hi
    # cdf is nonincreasing in F'; binary search for the largest F' with cdf >= delta
    while lo < hi_:
        mid = (lo + hi_ + 1) // 2
        if st.hypergeom(n_N, mid, n_cal).cdf(k) >= delta:
            lo = mid
        else:
            hi_ = mid - 1
    Fh = lo
    return (float(Fh - k) / float(n_N - n_cal), int(Fh))


def empirical_bernstein_ucb01(x: np.ndarray, delta: float) -> float:
    """Maurer-Pontil empirical Bernstein upper bound for the mean of [0,1] data."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 1.0
    v = float(x.var(ddof=1))
    L = math.log(2.0 / delta)
    return float(min(1.0, x.mean() + math.sqrt(2.0 * v * L / n) + 7.0 * L / (3.0 * (n - 1))))


def kl_robust_curve(d: Sequence[float], H_plus: float, delta: float, rhos: Sequence[float],
                    lambdas: Optional[Sequence[float]] = None) -> List[dict]:
    """Theorem 3(v): with simultaneous UCBs U_j on E_P[exp(lambda_j (d - H+))], every Q
    with KL(Q||P) <= rho has E_Q[d] <= H+ + min_j (rho + log U_j)/lambda_j."""
    d = np.asarray(d, dtype=np.float64)
    if not math.isfinite(H_plus) or H_plus < 0 or not 0.0 < delta < 1.0 or not np.isfinite(d).all():
        raise ValueError("KL robustness requires finite data, finite H_plus >= 0 and 0 < delta < 1")
    if np.any(d > H_plus * (1.0 + 1e-9) + 1e-12):
        raise _api.RTTFailure("SCIENTIFIC", "KL robustness requires the registered upper damage bound")
    if lambdas is None:
        # Fixed from H_plus, independently of the evaluated outcomes. A supplied
        # grid must be preregistered or chosen on disjoint development data.
        scale = max(float(H_plus), 1e-3)
        lambdas = list(np.geomspace(0.1 / scale, 100.0 / scale, 24))
    lambdas = [float(lam) for lam in lambdas]
    if not lambdas or any(not math.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError("KL robustness lambda grid must be finite, positive and nonempty")
    J = len(lambdas)
    U = []
    for lam in lambdas:
        X = np.exp(lam * (np.minimum(d, H_plus) - H_plus))
        U.append(max(empirical_bernstein_ucb01(X, delta / J), 1e-300))
    out = []
    for rho in rhos:
        vals = [H_plus + (rho + math.log(Uj)) / lam for Uj, lam in zip(U, lambdas)]
        ub = float(min(vals))
        out.append({"rho": float(rho), "E_Q_down_upper": ub, "certified_mean_gain": -ub,
                    "lambda_grid": lambdas, "grid_source": "registered_independently_of_audit"})
    return out


def rate_statement(events: int, n: int, delta: float, k_simul: int = 10) -> dict:
    return {"events": int(events), "n": int(n), "rate": (events / n) if n else float("nan"),
            "u_delta": clopper_pearson_upper(events, n, delta),
            "u_simultaneous": clopper_pearson_upper(events, n, delta / k_simul)}


def summarize_policy(recs: List[dict], cfg: _api.RTTConfig, delta: float, H_plus: Optional[float] = None,
                     certified: bool = True) -> dict:
    attempted = [r for r in recs if r.get("kind") != "group"]
    ok = [r for r in attempted if r.get("status") == "OK"]
    failed = len(attempted) - len(ok)
    n = len(ok)
    if n == 0:
        return {"n": 0, "n_attempted": len(attempted), "n_failed": failed,
                "status": "INCOMPLETE" if attempted else "NOT_RUN",
                "certificate_status": _api.CERTIFICATE_STATUS}
    gains = np.array([r["gain"] for r in ok], dtype=np.float64)
    downs = -gains
    m, lo, hi = paired_t_interval(gains)
    nz = int(sum(1 for r in ok if r["down"] > cfg.zeta))
    latency = np.asarray([r.get("latency_ratio", float("nan")) for r in ok], dtype=np.float64)
    latency = latency[np.isfinite(latency)]
    out = {"n": n, "n_attempted": len(attempted), "n_failed": failed,
           "status": "INCOMPLETE" if failed else "COMPLETE",
           "certificate_status": _api.CERTIFICATE_STATUS,
           "mean_gain": m, "ci_lo": None if failed else lo, "ci_hi": None if failed else hi, "N_zeta": nz,
           "u": None if failed else clopper_pearson_upper(nz, n, delta),
           "latency_ratio": float(latency.mean()) if latency.size else None,
           "latency_ratio_median": float(np.median(latency)) if latency.size else None,
           "acc_change_pp": float(np.mean([r["acc_change_pp"] for r in ok])),
           "acc_r": float(np.mean([r["acc_r"] for r in ok])), "acc_s": float(np.mean([r["acc_s"] for r in ok])),
           "neg_flip_frac": float(np.mean([r["neg_flip_frac"] for r in ok])),
           "exact_S_plus_max": float(np.max([r.get("exact_S_plus", 0.0) for r in ok]))}
    hp = H_plus if H_plus is not None else (ok[0].get("H_plus") if ok[0].get("H_plus") is not None else None)
    if failed:
        out["betting_lb"] = None
        out["betting_note"] = "Incomplete audit: failed calls cannot be dropped from an inference sample"
    elif certified and hp is not None and np.all(downs <= hp * (1 + 1e-9) + 1e-12):
        out["betting_lb"] = -betting_ucb(downs, hp, delta, "tuned")
    else:
        out["betting_lb"] = None
        if certified and hp is not None:
            out["betting_note"] = "d exceeded H+ on some call: G2 not satisfied, no betting bound"
    modes = {}
    for r in ok:
        modes[r.get("mode", "")] = modes.get(r.get("mode", ""), 0) + 1
    out["modes"] = modes
    viol = sum(1 for r in ok if r.get("violation_damage"))
    rviol = sum(1 for r in ok if r.get("violation_row"))
    out["violations_damage"] = viol
    out["violations_row"] = rviol
    return out


def release_ledger(recs: List[dict]) -> dict:
    ok = [r for r in recs if r.get("status") == "OK" and r.get("kind") != "group"]
    N = len(ok)
    R1 = sum(1 for r in ok if r.get("mode") in ("R1", "NO_STEP"))
    R2 = sum(1 for r in ok if r.get("mode") == "R2")
    Fb = sum(1 for r in ok if r.get("mode") == "F")
    fail = R2 + Fb
    by = {"damage": 0, "row": 0, "descent": 0}
    by_r2 = {"damage": 0, "row": 0, "descent": 0}
    by_f = {"damage": 0, "row": 0, "descent": 0}
    for r in ok:
        ff = r.get("first_fail") or ""
        if r.get("mode") in ("R2", "F") and ff in by:
            by[ff] += 1
            (by_r2 if r.get("mode") == "R2" else by_f)[ff] += 1
    assert 0 <= Fb <= fail <= N, "release ledger inconsistent"
    return {"N": N, "first_pass_release": R1, "no_step_calls": sum(1 for r in ok if r.get("mode") == "NO_STEP"),
            "failed_first_pass": fail, "failed_by_predicate": by, "recertified_release": R2,
            "recertified_by_predicate": by_r2, "fallback": Fb, "fallback_by_predicate": by_f,
            "share_R1": R1 / N if N else float("nan"), "share_fail": fail / N if N else float("nan"),
            "share_F": Fb / N if N else float("nan")}

