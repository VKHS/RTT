#!/usr/bin/env python3
"""Serve one label-free ogbn-arxiv call from a completed RTT suite.

This is a research inference utility. It does not turn the numerical checker
into a formally validated floating-point verifier. A member that did not pass
selection is never silently deployed as the registered policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_rtt_iclr2027_arrhenius as rtt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--component", choices=rtt.ALL_COMPONENTS, default="clean")
    parser.add_argument("--pool", choices=["dev19", "final19", "transfer20"], default="final19")
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--member", default="registered",
                        help="registered, or a named calibrated member for explicit exploratory inference")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--graph-npz", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.call_index < 0 or args.threads <= 0:
        parser.error("--call-index must be nonnegative and --threads must be positive")
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output must be empty; existing predictions are not overwritten")
    suite = rtt.Suite(args.suite)
    cfg, metadata = suite.config()
    plan_path = suite.manifest / "pipeline.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    # Preserve all dataset arguments from the suite; explicit path overrides are
    # accepted only if the runner's full content hash still matches the checkpoint.
    dataset_options = dict(
        dataset_kind=metadata["dataset_kind"], dataset_name="ogbn-arxiv",
        dataset_root=plan.get("dataset_root", "./datasets/ogb_cache"),
        rtt_graph_npz=plan.get("graph_npz", "./datasets/graph/rtt_ogbn_arxiv_graph.npz"),
        graph_pt=plan.get("graph_pt"), external_feature_pt=None,
        feature_normalization="standard", use_node_year=False,
        synthetic_nodes=1600, synthetic_features=16, synthetic_classes=6, synthetic_seed=7,
    )
    dataset_options.update(metadata.get("dataset_arguments", {}))
    if args.dataset_root is not None:
        dataset_options["dataset_root"] = args.dataset_root
    if args.graph_npz is not None:
        dataset_options["rtt_graph_npz"] = args.graph_npz
    da = argparse.Namespace(**dataset_options)
    if plan:
        from rtt_pipeline import verify_source
        verify_source(suite.root, plan)
    rtt.configure_torch_numerics(deterministic=True, threads=args.threads)
    device = rtt.pick_device(require_cuda=not args.cpu, force_cpu=args.cpu)
    start = time.perf_counter()
    loaded = rtt.load_system(suite, da, cfg, device, "native", need_heads=True)
    setup_seconds = time.perf_counter() - start
    if args.member == "registered":
        member, selection = rtt.selected_member(suite)
        if selection == "SELECTED":
            policy, _ = rtt.registered_policy(suite, cfg)
        else:
            policy = rtt.PolicySpec(name="incumbent_only_unselected", kind="rtt",
                                   H_plus=0.0, H_row=cfg.H_row, open_depths=[])
    else:
        member = rtt.member_by_name(args.member)
        radii = rtt.load_radii(suite, "native__" + member.name)
        if radii is None:
            parser.error("The explicitly requested member has not been calibrated")
        policy = rtt.policy_for_member(cfg, member, radii, name="exploratory_" + member.name)
        selection = "EXPLICIT_EXPLORATORY_MEMBER"
    call = loaded.env.generator.make("predict", args.pool, args.component, args.call_index)
    context = loaded.env.build(call, need_labels=False)
    with torch.no_grad():
        logits, executed, certificate, timing = rtt.serve(loaded.system, loaded.checker, context, policy)
        log_prob = torch.log_softmax(logits.double(), dim=1)
    assert context.y_q is None
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", node_ids=call.query,
                        logits=logits.cpu().numpy(), log_prob=log_prob.cpu().numpy(),
                        probabilities=log_prob.exp().cpu().numpy(),
                        predictions=log_prob.argmax(1).cpu().numpy())
    report = {
        "call_id": call.call_id, "component": call.component, "query_count": call.nq,
        "selection_status": selection, "policy": policy.name, "release_mode": certificate.mode,
        "numerical_check_passed": certificate.passed, "certificate_status": rtt.CERTIFICATE_STATUS,
        "damage_bound": certificate.damage_bound, "row_bound_max": certificate.row_bound_max,
        "first_fail": certificate.first_fail, "first_fail_r2": certificate.first_fail_r2,
        "timing_seconds": timing, "cold_model_and_cache_setup_seconds": setup_seconds,
        "labels_supplied_to_policy": False, "separate_full_incumbent_endpoint": context.inc_logits is not None,
        "device": str(device), "dataset": loaded.g.fingerprint(),
    }
    rtt.write_json(out / "call.json", report)
    print(rtt.to_json(report))


if __name__ == "__main__":
    main()
