"""Offline fitting; all supervised targets are isolated from online admission."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .adaptation import AdaptationModules, ReferenceTailEngine, true_tail_gradient
from .losses import supervised_loss
from .runtime import progress, seed_for


def _frozen_forward(model, batch):
    context = dict(batch.context)
    state = model.encode(batch.x, context)
    for depth in range(model.T):
        state = model.transition(state, depth, context)
    return state, context, batch.select(model.logits(state, context))


def train_incumbent(model, factory, protocol, config):
    settings, seed = config["training"], config["protocol"]["seed"]
    history, best, best_loss, stale = [], None, float("inf"), 0
    if settings["train_incumbent"]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        for epoch in range(settings["epochs"]):
            model.train()
            losses = []
            for step in range(settings["steps_per_epoch"]):
                batch = (factory.make(protocol["pools"]["inc_train"]) if factory.bundle.task == "node" else
                         factory.sample(protocol["pools"]["inc_train"], settings["batch_size"],
                                        seed_for(seed, "incumbent", epoch, step)))
                optimizer.zero_grad(set_to_none=True)
                logits = batch.select(model(batch.x, dict(batch.context)))
                loss = supervised_loss(logits, batch.labels, batch.multilabel, batch.mask)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite incumbent fitting loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                model.project_parameters()
                losses.append(float(loss.detach()))
            model.eval()
            validation = (factory.make(protocol["pools"]["inc_valid"]) if factory.bundle.task == "node" else
                          factory.sample(protocol["pools"]["inc_valid"], 4 * settings["batch_size"],
                                         seed_for(seed, "incumbent_validation")))
            with torch.no_grad():
                value = float(supervised_loss(validation.select(model(validation.x, dict(validation.context))),
                                             validation.labels, validation.multilabel, validation.mask))
            history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_loss": value})
            progress("incumbent", epoch + 1, settings["epochs"], validation_loss=f"{value:.6g}")
            if value < best_loss:
                best_loss, stale = value, 0
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
        if best is None:
            raise RuntimeError("No finite incumbent checkpoint was fitted")
        model.load_state_dict(best)
    model.freeze()
    temperature_report = {"fitted": False, "temperature": float(model.temperature)}
    if settings["fit_temperature"]:
        batch = factory.sample(protocol["pools"]["temperature"], min(512, 4 * settings["batch_size"]),
                               seed_for(seed, "temperature"))
        with torch.no_grad():
            logits = batch.select(model(batch.x, dict(batch.context))).detach() * model.temperature
        log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device, dtype=torch.float64))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=30)
        def closure():
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 20.0)
            loss = supervised_loss(logits / temperature, batch.labels, batch.multilabel, batch.mask)
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            model.temperature.copy_(log_temperature.exp().clamp(0.05, 20.0).to(model.temperature))
        temperature_report = {"fitted": True, "temperature": float(model.temperature), "fitting_items": len(batch.item_ids)}
    return {"history": history, "temperature": temperature_report,
            "train_incumbent": settings["train_incumbent"],
            "validation_item_limit": None if factory.bundle.task == "node" else 4 * settings["batch_size"],
            "training_scope": "registered fitting allocation only"}


def fit_proposal(model, modules, factory, protocol, config):
    settings, seed = config["training"], config["protocol"]["seed"]
    if config["adaptation"]["proposal_kind"] == "tent":
        return {"kind": "tent", "offline_proposal_updates": 0,
                "note": "Last-four-block BN-affine entropy updates are reset independently for each call"}
    optimizer = torch.optim.Adam(modules.proposal_parameters(), lr=settings["head_lr"])
    losses = []
    for index in range(settings["proposal_steps"]):
        count = config["protocol"]["query_size"] if factory.bundle.task == "node" else settings["batch_size"]
        batch = factory.sample(protocol["pools"]["fit"], count, seed_for(seed, "proposal", index))
        context = dict(batch.context)
        with torch.no_grad():
            state = model.encode(batch.x, context)
        optimizer.zero_grad(set_to_none=True)
        for depth in range(model.T):
            state = model.transition(state, depth, context)
            if depth in modules.open_depths:
                candidates = modules.depths[str(depth)].candidates(state, context, config["adaptation"]["row_trim"])
                state = state + 0.5 * candidates.sum(dim=0)
        logits = batch.select(model.logits(state, context))
        loss = supervised_loss(logits, batch.labels, batch.multilabel, batch.mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite proposal warm-start loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(modules.proposal_parameters()), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        progress("proposal", index + 1, settings["proposal_steps"], loss=f"{losses[-1]:.6g}")
    return {"kind": "native", "steps": len(losses), "losses": losses,
            "scheme": "supervised_raw_proposal_warm_start"}


def fit_heads(model, modules, factory, protocol, config):
    settings, adaptation, seed = config["training"], config["adaptation"], config["protocol"]["seed"]
    optimizer = torch.optim.Adam(list(modules.head_parameters()) + list(modules.corrector.parameters()), lr=settings["head_lr"])
    engine = ReferenceTailEngine(model, modules, adaptation)
    radii = {depth: adaptation["kappa_train"] for depth in modules.open_depths}
    losses = []
    for index in range(settings["head_steps"]):
        depth = modules.open_depths[index % len(modules.open_depths)]
        count = config["protocol"]["query_size"] if factory.bundle.task == "node" else settings["batch_size"]
        batch = factory.sample(protocol["pools"]["fit"], count, seed_for(seed, "heads_and_corrector", index))
        state, context = engine.serve(batch.unlabeled(), radii, stop_before=depth)
        target = true_tail_gradient(model, batch, state, depth + 1, context)
        module = modules.depths[str(depth)]
        # The field target is computed on the current executed trajectory. For
        # Tent, use exactly the independent exported proposal states used online.
        exported = context.get("rtt_exported_states", {}).get(depth)
        with torch.no_grad():
            candidates = module.candidates(state, context, adaptation["row_trim"], exported)
        field, variance = module.estimate(state.detach(), candidates.detach(), batch.units)
        error = (field - target).flatten(1) * batch.units
        row_error = error.square().mean(1)
        scaled_variance = variance * batch.units ** 2
        field_loss = row_error.mean()
        covariance_loss = 0.5 * (row_error.detach() / scaled_variance + torch.log(scaled_variance)).mean()
        with torch.no_grad():
            final_state, final_context, reference_logits = _frozen_forward(model, batch)
            embedding = batch.select(model.embedding(final_state, final_context)).detach()
        corrected_logits = reference_logits.detach() + modules.corrector(embedding)
        corrector_loss = supervised_loss(corrected_logits, batch.labels, batch.multilabel, batch.mask)
        loss = field_loss + 0.05 * covariance_loss + corrector_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite admission-head fitting loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(modules.head_parameters()) + list(modules.corrector.parameters()), 5.0)
        optimizer.step()
        losses.append({"step": index, "depth": depth, "field_loss": float(field_loss.detach()),
                       "covariance_loss": float(covariance_loss.detach()), "corrector_loss": float(corrector_loss.detach())})
        progress("admission_heads", index + 1, settings["head_steps"], depth=depth, loss=f"{float(loss.detach()):.6g}")
    modules.eval()
    return {"steps": len(losses), "losses": losses, "capacity": modules.capacity,
            "training_calls": "shared heads_and_corrector generator and episode indices",
            "scheme": "staged fields and covariance on executed-prefix trajectories; no implicit KKT differentiation"}


def fit_adaptation(model, factory, protocol, config):
    modules = AdaptationModules(model, config["adaptation"], seed=config["protocol"]["seed"]).to(factory.device)
    if not modules.open_depths:
        raise ValueError("At least one valid open depth is required")
    proposal = fit_proposal(model, modules, factory, protocol, config)
    heads = fit_heads(model, modules, factory, protocol, config)
    for parameter in modules.parameters():
        parameter.requires_grad_(False)
    return modules, {"proposal": proposal, "heads": heads}
