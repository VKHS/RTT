"""Checkpoint assembly and registered policy loading."""
from __future__ import annotations

from ._imports import (
    Any,
    Dict,
    Optional,
    Sequence,
    np,
)
from . import api as _api

# ----------------------------------------------------------------------------
# loading a complete system for a proposal
# ----------------------------------------------------------------------------

class Loaded:
    def __init__(self):
        self.g: Optional[_api.GraphData] = None
        self.roles: Optional[_api.DataRoles] = None
        self.env: Optional[_api.BaseEnv] = None
        self.inc: Optional[_api.TanhDiffusionIncumbent] = None
        self.system: Optional[_api.RTTSystem] = None
        self.checker: Optional[_api.Checker] = None
        self.models: Dict[str, Any] = {}
        self.registry: Dict[str, Any] = {}


def load_base(suite: _api.Suite, args, cfg: _api.RTTConfig, device, incumbent: str = "main") -> Loaded:
    L = Loaded()
    L.g = _api.load_graph(args)
    L.roles = _api.make_data_roles(L.g, cfg, seed=cfg.seed)
    L.env = _api.BaseEnv(L.g, L.roles, cfg, device, suite_seed=cfg.seed)
    obj = suite.load(f"incumbent_{incumbent}.pt")
    expected_graph = obj.get("meta", {}).get("graph", {}).get("sha256")
    if expected_graph is not None and L.g.fingerprint()["sha256"] != expected_graph:
        raise _api.RTTFailure("PREREQUISITE", "Dataset features, edges, labels, years or official splits differ from the frozen incumbent")
    L.inc = _api.build_incumbent_from_state(obj, cfg, device)
    L.env.attach_incumbent(L.inc)
    return L


def build_proposal(suite: _api.Suite, L: Loaded, cfg: _api.RTTConfig, name: str, device):
    if name == "native":
        obj = suite.load("proposal_native.pt")
        prop = _api.NativeProposal(cfg.hidden, cfg, seed=cfg.seed)
        prop.load_state_dict(obj["state_dict"])
        prop.to(device).eval()
        for p in prop.parameters():
            p.requires_grad_(False)
        return prop
    if name == "v2":
        v2 = _api.build_incumbent_from_state(suite.load("incumbent_v2.pt"), cfg, device)
        return _api.build_state_proposal("v2", L.inc, L.env, cfg, v2)
    if name == "lora16":
        obj = suite.load("proposal_lora16.pt")
        lora = _api.LoRAIncumbent(L.inc, 16, cfg.seed).to(device)
        lora.load_state_dict(obj["state_dict"], strict=False)
        return _api.build_state_proposal("lora16", L.inc, L.env, cfg, lora)
    if name == "gprgnn":
        obj = suite.load("proposal_gprgnn.pt")
        m = _api.GPRGNN(L.g.num_features, cfg.hidden, L.g.num_classes).to(device)
        m.load_state_dict(obj["state_dict"])
        m.eval()
        return _api.build_state_proposal("gprgnn", L.inc, L.env, cfg, m)
    raise _api.RTTFailure("ORCHESTRATION", f"unknown proposal {name}")


def load_system(suite: _api.Suite, args, cfg: _api.RTTConfig, device, proposal: str = "native",
                need_heads: bool = True, need_models: Sequence[str] = (), registry_stage: str = "final") -> Loaded:
    L = load_base(suite, args, cfg, device)
    prop = build_proposal(suite, L, cfg, proposal, device)
    reg = {"gamma": L.inc.gamma_delta(), "open_depths": cfg.open_depths(),
           "probes": _api.make_probes(cfg.hidden, cfg.probes_per_sign, cfg.seed), "caps_train": _api.uniform_caps(cfg)}
    if suite.has(f"phase2_{proposal}.json"):
        p2 = suite.load(f"phase2_{proposal}.json", kind="json")
        reg["role_thresholds"] = p2["role_thresholds"]
    if registry_stage == "final" and suite.has(f"registry_{proposal}.json"):
        rj = suite.load(f"registry_{proposal}.json", kind="json")
        reg["caps"] = np.asarray(rj["caps"], dtype=np.float64)
        reg["caps_spend"] = np.asarray(rj["caps_spend"], dtype=np.float64)
        reg["growth"] = np.asarray(rj["growth"], dtype=np.float64)
        reg["trust_share"] = {int(k): float(v) for k, v in rj["trust_share"].items()}
        reg["trust_share_spend"] = {int(k): float(v) for k, v in rj["trust_share_spend"].items()}
        reg["placement"] = rj
    n_open = len(cfg.open_depths())
    Fin = _api.feature_dim(L.inc.num_classes, n_open, cfg.probes_per_sign)
    heads = None
    extra = {}
    tr_heads = {}
    if need_heads:
        hobj = suite.load(f"heads_{proposal}.pt")
        heads = _api.RTTHeads(Fin, L.inc.num_classes, n_open, cfg).to(device)
        heads.load_state_dict(hobj["heads"])
        heads.eval()
        for key in ("tr_matched", "tr_e2e"):
            if key in hobj:
                h = _api.TrustRuleHead(Fin, cfg.K_roles, n_open, cfg).to(device)
                h.load_state_dict(hobj[key])
                h.eval()
                tr_heads[key] = h
        for key in ("heads_mean", "heads_probe"):
            if suite.has(f"{key}_{proposal}.pt"):
                hh = _api.RTTHeads(Fin, L.inc.num_classes, n_open, cfg).to(device)
                hh.load_state_dict(suite.load(f"{key}_{proposal}.pt")["heads"])
                hh.eval()
                extra[key] = hh
        for m in list(heads.parameters()) + [p for h in tr_heads.values() for p in h.parameters()] + \
                [p for h in extra.values() for p in h.parameters()]:
            m.requires_grad_(False)
    L.system = _api.RTTSystem(L.inc, prop, heads, reg, cfg, device, tr_heads=tr_heads, extra_heads=extra)
    L.checker = _api.Checker(L.system)
    L.registry = reg
    P2 = 2 * cfg.probes_per_sign
    if "correctors" in need_models and suite.has(f"correctors_{proposal}.pt"):
        cobj = suite.load(f"correctors_{proposal}.pt")
        cm = {}
        for name, hops, rows in _api.CORRECTOR_SPECS:
            if name in cobj:
                width = int(cobj[name]["net.lins.0.weight"].shape[0])
                m = _api.OnePassCorrector(L.inc.hidden, L.inc.num_classes, hops, rows, P2, width, name).to(device)
                m.load_state_dict(cobj[name])
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
                cm[name] = m
        L.models["correctors"] = cm
    if "crossfit" in need_models and suite.has("crossfit.pt"):
        cobj = suite.load("crossfit.pt")
        ms = []
        for f, sd in enumerate(cobj["folds"]):
            m = _api.OnePassCorrector(L.inc.hidden, L.inc.num_classes, 4, True, P2, 2 * cfg.head_hidden, f"crossfit_{f}").to(device)
            m.load_state_dict(sd)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            ms.append(m)
        # the cross-fitted posterior reads the native proposal's candidate rows
        L.models["crossfit"] = _api.CrossFitPosterior(ms, cobj["temperature"])
    if "ft" in need_models:
        fts = {}
        for v in _api.FT_VARIANTS:
            if suite.has(f"{v}.pt"):
                fts[v] = _api.build_incumbent_from_state(suite.load(f"{v}.pt"), cfg, device)
        L.models["ft"] = fts
    return L


def member_by_name(name: str) -> _api.MemberSpec:
    for m in _api.default_library():
        if m.name == name:
            return m
    raise _api.RTTFailure("ORCHESTRATION", f"unknown member {name}")


def load_radii(suite: _api.Suite, key: str) -> Optional[Dict[int, float]]:
    name = f"radii_{key}.json"
    if not suite.has(name):
        return None
    obj = suite.load(name, kind="json")
    return {int(k): float(v) for k, v in obj["radii"].items()}


SWAPS = {
    # name: (description, PolicySpec overrides, calibration settings)
    "swap_no_row": ("No per-row budget (H_row = inf)", {"H_row": float("inf")}),
    "swap_spend_caps": ("Spend-based caps instead of efficiency placement", {"caps_key": "caps_spend"}),
    "swap_env_only": ("Window solved under the envelope only", {"window_mode": "env_only"}),
    "swap_mean_features": ("Mean-pooled candidate features (not hull-invariant)", {"feature_mode": "mean", "heads_key": "heads_mean"}),
    "swap_probe_slopes": ("Probe-loss slopes instead of full-loss slopes", {"heads_key": "heads_probe"}),
    "swap_sym_005": ("Symmetric charges at the same budget", {"pricing": "symmetric"}),
    "swap_sym_030": ("Symmetric charges at 0.3 nats", {"pricing": "symmetric", "H_plus": 0.3}),
    "swap_global_factors": ("Global factors only (no interval tube)", {"factor_mode": "global"}),
    "swap_late8": ("Late-8 acceptance (l >= T-8 only)", {"open_depths": "late8"}),
}


def policy_for_member(cfg: _api.RTTConfig, member: _api.MemberSpec, radii: Optional[Dict[int, float]], name: Optional[str] = None,
                      overrides: Optional[dict] = None) -> _api.PolicySpec:
    spec = _api.PolicySpec(name=name or f"rtt_{member.name}", kind="rtt", H_plus=member.H_plus(cfg), H_row=cfg.H_row,
                      radii=radii, caps_key="caps", member=member.name)
    for k, v in (overrides or {}).items():
        if k == "open_depths" and v == "late8":
            v = list(range(max(cfg.open_first, cfg.T - 8), cfg.T))
        setattr(spec, k, v)
    return spec


def trust_share_for(reg: dict, overrides: Optional[dict]) -> Dict[int, float]:
    if overrides and overrides.get("caps_key") == "caps_spend":
        return reg.get("trust_share_spend", {})
    return reg.get("trust_share", {})


