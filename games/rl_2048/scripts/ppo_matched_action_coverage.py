"""One-state, matched finite-pool action-gradient replacement for PPO2048."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from games.rl_2048.environment.game_2048 import State2048
from games.rl_2048.scripts.audit_ppo_update import load_runtime, sha256, write_json
from games.rl_2048.scripts.balanced_sampling_ppo import clipped_policy_loss, _read_pool
from games.rl_2048.scripts.ppo_broad_gradient_audit import _probability_metrics

REPO = Path(__file__).resolve().parents[3]
BANK_DIR = REPO / "artifacts/ppo_2048/ppo_broad_gradient_audit_iter100"
OUT = REPO / "artifacts/ppo_2048/ppo_matched_action_coverage_iter100"
UPDATE_SELECTION = REPO / "artifacts/ppo_2048/ppo_update_audit_iter100/selection.json"
EXPANDED_TRACE = BANK_DIR / "expanded_bank_update_traces.json"


def _canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _flat_grads(grads, params):
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=torch.float64, device="cpu"))
        else:
            pieces.append(grad.detach().reshape(-1).to(dtype=torch.float64, device="cpu"))
    return torch.cat(pieces)


def _flat_param_state(params):
    return torch.cat([p.detach().reshape(-1).to(dtype=torch.float64, device="cpu") for p in params])


def _nested_tensor_hash(obj):
    h = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else:
            h.update(repr(value).encode())
    visit(obj)
    return h.hexdigest()


def _verify_inputs(folder, lock, bank_plan, trace, capture):
    if _canonical_hash({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock["lock_sha256"]:
        raise ValueError("locked plan SHA does not match")
    if bank_plan["lock_sha256"] != lock["inputs"]["bank_lock_sha256"]:
        raise ValueError("matched plan points at a changed bank lock")
    if sha256(BANK_DIR / "capture.pt") != lock["inputs"]["capture_sha256"]:
        raise ValueError("capture hash changed")
    if sha256(EXPANDED_TRACE) != lock["inputs"]["expanded_update_trace_sha256"]:
        raise ValueError("expanded trace hash changed")
    if sha256(BANK_DIR / "heldout_target_outcomes.json") != lock["inputs"]["heldout_outcome_sha256"]:
        raise ValueError("heldout outcome artifact hash changed")
    natural_draws = [{"replicate": x["replicate"], "samples": x["arms"]["natural"]["samples"]} for x in trace["sample_draws"]]
    if _canonical_hash(natural_draws) != lock["draw_manifest_sha256"]:
        raise ValueError("natural draw manifest hash changed")
    if len(natural_draws) != 12 or any(len(x["samples"]) != 96 * 32 for x in natural_draws):
        raise ValueError("expected 12 natural manifests with exactly 32 rows/state")
    for name, digest in lock["inputs"]["training_pools"].items():
        if sha256(BANK_DIR / name) != digest: raise ValueError(f"training pool changed: {name}")
    for name, digest in lock["inputs"]["heldout_pools"].items():
        if sha256(BANK_DIR / name) != digest: raise ValueError(f"heldout pool changed: {name}")
    if capture.get("source_sha256") != lock["inputs"]["checkpoint_sha256"]:
        raise ValueError("capture checkpoint SHA mismatch")
    amendment_path = folder / "amendment_v2.json"
    amendment = json.load(open(amendment_path))
    amendment_hash = amendment.pop("amendment_sha256")
    if _canonical_hash(amendment) != amendment_hash:
        raise ValueError("versioned baseline amendment hash does not match")
    if amendment["prior_locked_plan_sha256"] != lock["lock_sha256"]:
        raise ValueError("amendment does not refer to the preserved original lock")


def _load_sample_rows(draw, boards, pools, mean, std, epsilon):
    rows = []
    counts = [{int(a): 0 for a in b["legal_actions"]} for b in boards]
    action_records = [[[] for _ in range(4)] for _ in boards]
    for state_id, action, trial in draw["samples"]:
        state_id, action, trial = int(state_id), int(action), int(trial)
        if action not in boards[state_id]["legal_actions"]: raise ValueError("sample manifest contains illegal action")
        pool = pools[(state_id, action)]
        if not 0 <= trial < len(pool): raise ValueError("sample trial index outside locked pool")
        raw_gae = float(pool[trial]["gae"])
        norm_gae = (raw_gae - mean) / (std + epsilon)
        rows.append((state_id, action, trial, raw_gae, norm_gae))
        counts[state_id][action] += 1
        action_records[state_id][action].append(norm_gae)
    if any(sum(row.values()) != 32 for row in counts): raise ValueError("natural manifest does not have 32 rows/state")
    return rows, counts, action_records


def _batch_arrays(rows, old_prob):
    state_ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    actions = torch.tensor([r[1] for r in rows], dtype=torch.long)
    raw_adv = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    old_logs = torch.tensor([math.log(old_prob[r[0]][r[1]]) for r in rows], dtype=torch.float32)
    weights = torch.full((len(rows),), 1.0 / (96.0 * 32.0), dtype=torch.float32)
    return state_ids, actions, raw_adv, old_logs, weights


def _reference_loss_for_state(log_probs, state_id, board, pools, old_prob, mean, std, eps, clip_eps):
    terms = []
    for action in board["legal_actions"]:
        prob = float(old_prob[state_id][int(action)])
        records = pools[(state_id, int(action))]
        raw_adv = torch.tensor([float(r["gae"]) for r in records], dtype=torch.float32)
        adv = (raw_adv - mean) / (std + eps)
        old = torch.full((len(records),), math.log(prob), dtype=torch.float32)
        ratio = torch.exp(log_probs[state_id, int(action)].expand_as(old) - old)
        loss = clipped_policy_loss(ratio, adv, clip_eps)
        weight = prob / (96.0 * len(records))
        terms.append(torch.sum(weight * loss))
    return torch.stack(terms).sum()


def _loss_components(model, ppo, states, legal_masks, row_ids, actions, adv, old_logs, weights,
                     focal_id=None, boards=None, pools=None, old_prob=None, mean=0., std=1.,
                     denom=1e-8):
    output = model(ppo._state_batch_tensor(states))
    log_probs = ppo._action_log_probabilities(output.action_logits.masked_fill(legal_masks, -torch.inf), legal_masks)
    selected = log_probs[row_ids, actions]
    ratios = torch.exp(selected - old_logs)
    normalized_adv = (adv - mean) / (std + denom)
    per_loss = clipped_policy_loss(ratios, normalized_adv, ppo.config.clip_epsilon)
    weighted_rows = weights * per_loss
    entropy = Categorical(logits=log_probs).entropy().mean()
    policy_baseline = weighted_rows.sum()
    policy = policy_baseline
    if focal_id is not None:
        keep = row_ids != focal_id
        policy = weighted_rows[keep].sum() + _reference_loss_for_state(
            log_probs, focal_id, boards[focal_id], pools, old_prob, mean, std,
            denom, ppo.config.clip_epsilon,
        )
    objective = policy - ppo.config.entropy_coefficient * entropy
    return objective, policy_baseline, policy, entropy, ratios, log_probs


def _score_logodds_projections(model, ppo, boards, states, actor_params):
    model.eval()
    projections, logodds_gradients = [], []
    eps = float(ppo.exploration_probability)
    for board, state in zip(boards, states):
        logits = model(ppo._state_batch_tensor([state])).action_logits[0]
        masked = torch.ones_like(logits, dtype=torch.bool); masked[board["legal_actions"]] = False
        actor = torch.softmax(logits.masked_fill(masked, -torch.inf), -1)
        behavior = (1.0-eps) * actor; behavior = behavior.clone()
        behavior[board["legal_actions"]] += eps / len(board["legal_actions"])
        target, preferred = int(board["critic_action"]), int(board["actor_action"])
        grad_odds = torch.autograd.grad(logits[target] - logits[preferred], actor_params, retain_graph=True)
        flat_odds = _flat_grads(grad_odds, actor_params)
        logodds_gradients.append(flat_odds)
        action_proj = {}
        for action in board["legal_actions"]:
            grad_score = torch.autograd.grad(torch.log(behavior[int(action)]), actor_params, retain_graph=True)
            action_proj[int(action)] = float(torch.dot(_flat_grads(grad_score, actor_params), flat_odds))
        projections.append(action_proj)
    return projections, logodds_gradients


def _metrics_after(model, ppo, boards, old_behavior, old_base, heldout_means):
    behavior, base, target, odds = _probability_metrics(model, ppo, boards)
    value_by_state = [float(sum((behavior[i, int(a)] - old_behavior[i, int(a)]) * heldout_means[i][int(a)]
                                 for a in board["legal_actions"])) for i, board in enumerate(boards)]
    return {"target_behavior_probabilities": target.tolist(),
            "target_base_actor_probabilities": [float(base[i, int(b["critic_action"])]) for i,b in enumerate(boards)],
            "target_preferred_base_logodds": odds.tolist(),
            "mean_logodds_change_from_start": float(np.mean(odds - np.asarray(_metrics_after.old_odds))),
            "mean_behavior_target_probability_change_from_start": float(np.mean(target - np.asarray(_metrics_after.old_target))),
            "heldout_return_weighted_policy_change_by_state": value_by_state,
            "heldout_return_weighted_policy_change_bank_mean": float(np.mean(value_by_state)),
            "mean_behavior_kl_from_start": float(np.mean(np.sum(np.where(old_behavior > 0,
                old_behavior * (np.log(np.maximum(old_behavior,1e-300)) - np.log(np.maximum(behavior,1e-300))),0),axis=1)))}, behavior, base, target, odds


def _restore(model, optimizer, capture, lr, actor_params):
    model.load_state_dict(capture["model_state_dict"])
    expected_model_hash=_nested_tensor_hash(capture["model_state_dict"])
    if _nested_tensor_hash(model.state_dict()) != expected_model_hash:
        raise ValueError("model/parameter/buffer restore differs from pristine captured model")
    optimizer.load_state_dict(copy.deepcopy(capture["optimizer_state_dict"]))
    for group in optimizer.param_groups: group["lr"] = float(lr)
    model.train()
    return _nested_tensor_hash(optimizer.state_dict())


def _step(model, optimizer, ppo, actor_params, objective):
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    grad_pre = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in actor_params if p.grad is not None))
    if ppo.config.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(actor_params, ppo.config.gradient_clip_norm)
    grad_post = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in actor_params if p.grad is not None))
    optimizer.step()
    return grad_pre, grad_post


def _optimizer_counter(optimizer, actor_params):
    for p in actor_params:
        state=optimizer.state.get(p,{})
        step=state.get("step")
        if step is not None: return int(step.item()) if torch.is_tensor(step) else int(step)
    return None


def _baseline(args):
    folder=args.folder.resolve(); lock=json.load(open(folder/"locked_plan.json"))
    bank_plan=json.load(open(BANK_DIR/"lockedplan.json")); trace=json.load(open(EXPANDED_TRACE)); capture=torch.load(BANK_DIR/"capture.pt",map_location="cpu",weights_only=False)
    _verify_inputs(folder,lock,bank_plan,trace,capture)
    prior=json.load(open(UPDATE_SELECTION)); torch.set_num_threads(int(prior["threads"]))
    _,model,optimizer,ppo,_=load_runtime(Path(prior["source_checkpoint"]),prior["seed"],prior["current_d4_order_acknowledged"])
    boards=bank_plan["existing_boards"]+bank_plan["new_boards"]
    states=[State2048(grid=np.asarray(b["board"],dtype=np.int64)) for b in boards]
    masks=torch.ones((96,4),dtype=torch.bool)
    for i,b in enumerate(boards): masks[i,b["legal_actions"]]=False
    actor_params=[]
    for name,param in model.named_parameters():
        param.requires_grad_("critic" not in name and "value" not in name)
        if param.requires_grad:actor_params.append(param)
    mean,std=float(capture["advantage_mean"]),float(capture["advantage_std"])
    denom=float(prior["ppo_config"].get("denominator_epsilon",1e-8)); lr=float(lock["learning_rate"])
    old_behavior,old_base,old_target,old_odds=_probability_metrics(model,ppo,boards)
    heldout_means=[{int(a):float(np.mean([r["discounted_return"] for r in _read_pool(BANK_DIR,"heldout",i,int(a))])) for a in b["legal_actions"]} for i,b in enumerate(boards)]
    _metrics_after.old_odds=old_odds.copy(); _metrics_after.old_target=old_target.copy()
    pools={(i,int(a)):_read_pool(BANK_DIR,"training",i,int(a)) for i,b in enumerate(boards) for a in b["legal_actions"]}
    draw_rows=trace["sample_draws"]
    baselines=[]; optimizer_hashes=[]; replicate_checks=[]
    pristine_model_hash=_nested_tensor_hash(capture["model_state_dict"])
    pristine_optimizer_hash=_nested_tensor_hash(capture["optimizer_state_dict"])
    effective_optimizer_state=copy.deepcopy(capture["optimizer_state_dict"])
    for group in effective_optimizer_state["param_groups"]: group["lr"]=lr
    effective_optimizer_hash=_nested_tensor_hash(effective_optimizer_state)
    def run_draw(repdraw):
        rep=int(repdraw["replicate"]); rows,counts,action_records=_load_sample_rows(repdraw["arms"]["natural"],boards,pools,mean,std,denom)
        row_ids,actions,adv,old_logs,weights=_batch_arrays(rows, [{int(a):float(b["behavior_probabilities"][int(a)]) for a in b["legal_actions"]} for b in boards])
        opt_hash=_restore(model,optimizer,capture,lr,actor_params)
        if opt_hash != effective_optimizer_hash: raise ValueError("baseline Adam restore differs from pristine captured optimizer plus locked LR")
        if _optimizer_counter(optimizer,actor_params) != 4896: raise ValueError("baseline Adam counter did not restore to 4896")
        params_before=_flat_param_state(actor_params)
        objective,policy_loss,_,entropy,ratios,_=_loss_components(model,ppo,states,masks,row_ids,actions,adv,old_logs,weights,mean=mean,std=std,denom=denom)
        ratio_values=ratios.detach().cpu().numpy()
        grad_pre,grad_post=_step(model,optimizer,ppo,actor_params,objective)
        params_after=_flat_param_state(actor_params)
        if _optimizer_counter(optimizer,actor_params) != 4897: raise ValueError("Adam step counter did not advance once from captured state")
        if _nested_tensor_hash(capture["optimizer_state_dict"]) != pristine_optimizer_hash or _nested_tensor_hash(capture["model_state_dict"]) != pristine_model_hash:
            raise ValueError("baseline update mutated source capture tensors")
        model.eval()
        actual,behavior,base,target,odds=_metrics_after(model,ppo,boards,old_behavior,old_base,heldout_means)
        clip_fraction=float(np.mean(np.abs(ratio_values-1)>ppo.config.clip_epsilon))
        return {"replicate":rep,"policy_loss":float(policy_loss.detach()),"entropy":float(entropy.detach()),"ratio_mean":float(ratio_values.mean()),"ratio_max_abs_from_one":float(np.max(np.abs(ratio_values-1))),"clip_fraction":clip_fraction,"gradient_norm_before_clip":grad_pre,"gradient_norm_after_clip":grad_post,"actor_parameter_displacement_l2":float((params_after-params_before).norm()),"adam_step_before":4896,"adam_step_after":_optimizer_counter(optimizer,actor_params),"optimizer_state_hash_before":opt_hash,"mean_target_logodds_change_from_start":float(np.mean(odds-old_odds)),"mean_logodds_change_by_state":(odds-old_odds).tolist(),"mean_target_behavior_probability_change_by_state":(target-old_target).tolist(),"target_base_actor_probabilities_96":[float(base[i,int(b["critic_action"])]) for i,b in enumerate(boards)],"target_base_actor_probability_change_by_state":[float(base[i,int(b["critic_action"])]-old_base[i,int(b["critic_action"])] ) for i,b in enumerate(boards)],"heldout_return_weighted_policy_change_by_state":actual["heldout_return_weighted_policy_change_by_state"],"heldout_return_weighted_policy_change_bank_mean":actual["heldout_return_weighted_policy_change_bank_mean"],"behavior_kl_from_start":actual["mean_behavior_kl_from_start"],"target_behavior_probabilities_96":target.tolist(),"target_preferred_logodds_96":odds.tolist()}
    forward={int(d["replicate"]):run_draw(d) for d in draw_rows}
    reverse={int(d["replicate"]):run_draw(d) for d in reversed(draw_rows)}
    tolerances=lock["checks_locked_before_execution"]["baseline_tolerances"]
    probability_errors=[];odds_errors=[];metric_errors=[]
    for rep in range(12):
        a,b=forward[rep],reverse[rep]
        probability_errors.append(float(np.max(np.abs(np.asarray(a["target_behavior_probabilities_96"])-np.asarray(b["target_behavior_probabilities_96"])))) )
        odds_errors.append(float(np.max(np.abs(np.asarray(a["target_preferred_logodds_96"])-np.asarray(b["target_preferred_logodds_96"])))) )
        metric_errors.append(max(abs(a[k]-b[k]) for k in ("policy_loss","entropy","ratio_mean","clip_fraction","gradient_norm_before_clip","gradient_norm_after_clip","actor_parameter_displacement_l2")))
    parity={"replicates":12,"reference":"Regenerated locked natural draws with pristine captured model/Adam deep-copy restored per replicate; saved expanded trace endpoints are not a valid reference because their optimizer steps carry forward.","max_target_behavior_probability_error_forward_vs_reverse":max(probability_errors),"max_abs_logodds_error_forward_vs_reverse":max(odds_errors),"max_abs_step_metric_error_forward_vs_reverse":max(metric_errors),"tolerances":tolerances,"passed":max(probability_errors)<=tolerances["max_target_behavior_probability_error"] and max(odds_errors)<=tolerances["max_target_preferred_logodds_error"] and max(metric_errors)<=tolerances["scalar_step_metrics_atol"] and all(x["adam_step_after"]==4897 and x["optimizer_state_hash_before"]==effective_optimizer_hash for x in forward.values())}
    baselines=[forward[i] for i in range(12)]
    replicate_checks=[{"replicate":i,"forward_vs_reverse_target_probability_error":probability_errors[i],"forward_vs_reverse_logodds_error":odds_errors[i],"forward_displacement_l2":forward[i]["actor_parameter_displacement_l2"],"reverse_displacement_l2":reverse[i]["actor_parameter_displacement_l2"],"adam_step_before_after":[forward[i]["adam_step_before"],forward[i]["adam_step_after"]]} for i in range(12)]
    if not parity["passed"]:
        parity["per_replicate_checks"]=replicate_checks
        write_json(folder/"baseline_parity_check.json",parity)
        raise ValueError(f"corrected independent-restore baseline integrity gate failed: {parity}")
    write_json(folder/"baseline_parity_check.json",{**parity,"per_replicate_checks":replicate_checks,"pristine_optimizer_state_hash":pristine_optimizer_hash,"pristine_model_state_hash":pristine_model_hash,"source_capture_unchanged":True})
    write_json(folder/"baseline_step_traces.json",{"scope":"12 ordinary natural 96-state batch draws; independent deep-copy captured params/Adam per draw; ascending/descending replay order check","bank_lock_sha256":bank_plan["lock_sha256"],"matched_plan_sha256":lock["lock_sha256"],"amendment_sha256":json.load(open(folder/"amendment_v2.json"))["amendment_sha256"],"draw_manifest_sha256":lock["draw_manifest_sha256"],"baseline_parity":parity,"optimizer_state_hashes_before":[x["optimizer_state_hash_before"] for x in baselines],"traces":baselines})
    print(f"Corrected baseline integrity gate passed; max target probability order error {parity['max_target_behavior_probability_error_forward_vs_reverse']:.3g}, max logodds order error {parity['max_abs_logodds_error_forward_vs_reverse']:.3g}",flush=True)


def _direct_gradient_validation(model,optimizer,ppo,actor_params,capture,lr,boards,states,masks,pools,draw_rows,old_prob,mean,std,denom):
    focal_ids=[0,1,3,24,48,68,95]
    reps=[0,6,11]
    errors=[]
    for rep in reps:
        draw=draw_rows[rep]["arms"]["natural"]
        rows,_,_=_load_sample_rows(draw,boards,pools,mean,std,denom)
        row_ids,actions,adv,old_logs,weights=_batch_arrays(rows,old_prob)
        normalized_adv=(adv-mean)/(std+denom)
        for focal in focal_ids:
            _restore(model,optimizer,capture,lr,actor_params)
            _,base_policy,_,entropy,_,logs=_loss_components(model,ppo,states,masks,row_ids,actions,adv,old_logs,weights,mean=mean,std=std,denom=denom)
            indices=row_ids==focal
            sampled= (weights[indices]*clipped_policy_loss(torch.exp(logs[row_ids[indices],actions[indices]]-old_logs[indices]),normalized_adv[indices],ppo.config.clip_epsilon)).sum()
            ref=_reference_loss_for_state(logs,focal,boards[focal],pools,old_prob,mean,std,denom,ppo.config.clip_epsilon)
            ref_action_losses=[]
            for action in boards[focal]["legal_actions"]:
                action=int(action); prob=float(old_prob[focal][action]); records=pools[(focal,action)]
                action_raw_adv=torch.tensor([float(r["gae"]) for r in records],dtype=torch.float32)
                action_adv=(action_raw_adv-mean)/(std+denom)
                action_old=torch.full((len(records),),math.log(prob),dtype=torch.float32)
                action_ratio=torch.exp(logs[focal,action].expand_as(action_old)-action_old)
                ref_action_losses.append((prob/(96.0*len(records)))*clipped_policy_loss(action_ratio,action_adv,ppo.config.clip_epsilon).sum())
            direct=(weights[~indices]*clipped_policy_loss(torch.exp(logs[row_ids[~indices],actions[~indices]]-old_logs[~indices]),normalized_adv[~indices],ppo.config.clip_epsilon)).sum()+ref-ppo.config.entropy_coefficient*entropy
            base=base_policy-ppo.config.entropy_coefficient*entropy
            params=actor_params
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gs=torch.autograd.grad(sampled,params,retain_graph=True,allow_unused=True)
            gr=torch.autograd.grad(ref,params,retain_graph=True,allow_unused=True)
            ref_action_grads=[]
            for j, action_loss in enumerate(ref_action_losses):
                ref_action_grads.append(_flat_grads(torch.autograd.grad(action_loss,params,retain_graph=True,allow_unused=True),params))
            gd=torch.autograd.grad(direct,params,retain_graph=False,allow_unused=True)
            composed=_flat_grads(gb,params)-_flat_grads(gs,params)+_flat_grads(gr,params)
            direct_flat=_flat_grads(gd,params)
            action_sum=torch.stack(ref_action_grads).sum(dim=0)
            errors.append({"replicate":rep,"focal_state":focal,"absolute_gradient_l2_error":float((composed-direct_flat).norm()),"reference_action_gradient_sum_error":float((action_sum-_flat_grads(gr,params)).norm())})
    maximum=max(x["absolute_gradient_l2_error"] for x in errors)
    action_max=max(x["reference_action_gradient_sum_error"] for x in errors)
    if max(maximum,action_max)>1e-5: raise ValueError(f"replacement gradient identity failed; replacement={maximum}, action sum={action_max}")
    return {"focal_ids":[0,1,3,24,48,68,95],"replicates":[0,6,11],"n_checks":len(errors),"max_absolute_gradient_l2_error":maximum,"max_reference_action_gradient_sum_l2_error":action_max,"tolerance":1e-5,"passed":True,"checks":errors}


def _interventions(args):
    folder=args.folder.resolve(); lock=json.load(open(folder/"locked_plan.json")); bank_plan=json.load(open(BANK_DIR/"lockedplan.json")); trace=json.load(open(EXPANDED_TRACE)); capture=torch.load(BANK_DIR/"capture.pt",map_location="cpu",weights_only=False)
    _verify_inputs(folder,lock,bank_plan,trace,capture)
    baseline_path=folder/"baseline_step_traces.json"
    if not baseline_path.exists(): raise ValueError("run the baseline stage and pass its parity gate first")
    baseline=json.load(open(baseline_path))
    if not baseline["baseline_parity"]["passed"]: raise ValueError("baseline parity did not pass")
    prior=json.load(open(UPDATE_SELECTION)); torch.set_num_threads(int(prior["threads"]))
    _,model,optimizer,ppo,_=load_runtime(Path(prior["source_checkpoint"]),prior["seed"],prior["current_d4_order_acknowledged"])
    boards=bank_plan["existing_boards"]+bank_plan["new_boards"]; states=[State2048(grid=np.asarray(b["board"],dtype=np.int64)) for b in boards]
    masks=torch.ones((96,4),dtype=torch.bool)
    for i,b in enumerate(boards):masks[i,b["legal_actions"]]=False
    actor_params=[]
    for name,p in model.named_parameters():
        p.requires_grad_("critic" not in name and "value" not in name)
        if p.requires_grad:actor_params.append(p)
    mean,std=float(capture["advantage_mean"]),float(capture["advantage_std"]); denom=float(prior["ppo_config"].get("denominator_epsilon",1e-8)); lr=float(lock["learning_rate"])
    old_prob=[{int(a):float(b["behavior_probabilities"][int(a)]) for a in b["legal_actions"]} for b in boards]
    old_behavior,old_base,old_target,old_odds=_probability_metrics(model,ppo,boards)
    _metrics_after.old_odds=old_odds.copy(); _metrics_after.old_target=old_target.copy()
    heldout_means=[{int(a):float(np.mean([r["discounted_return"] for r in _read_pool(BANK_DIR,"heldout",i,int(a))])) for a in b["legal_actions"]} for i,b in enumerate(boards)]
    pools={(i,int(a)):_read_pool(BANK_DIR,"training",i,int(a)) for i,b in enumerate(boards) for a in b["legal_actions"]}
    # The lock commits all necessary hashes; recompute per-action base log-odds score projections once.
    _restore(model,optimizer,capture,lr,actor_params)
    score_projections,logodds_gradients=_score_logodds_projections(model,ppo,boards,states,actor_params)
    check=_direct_gradient_validation(model,optimizer,ppo,actor_params,capture,lr,boards,states,masks,pools,trace["sample_draws"],old_prob,mean,std,denom)
    samples=[]
    for repdraw in trace["sample_draws"]:
        rows,counts,action_records=_load_sample_rows(repdraw["arms"]["natural"],boards,pools,mean,std,denom)
        samples.append((rows,counts,action_records,*_batch_arrays(rows,old_prob)))
    baselines_by_rep={int(x["replicate"]):x for x in baseline["traces"]}
    def run_order_arm(rep, focal):
        rows,counts,action_records,row_ids,actions,adv,old_logs,weights=samples[rep]
        restored_hash=_restore(model,optimizer,capture,lr,actor_params)
        if restored_hash != baseline["optimizer_state_hashes_before"][rep]: raise ValueError("order check Adam restore differs from baseline")
        objective,policy_base,policy_cf,entropy,ratio,logs=_loss_components(model,ppo,states,masks,row_ids,actions,adv,old_logs,weights,
            focal_id=(focal if focal >= 0 else None),boards=boards,pools=pools,old_prob=old_prob,mean=mean,std=std,denom=denom)
        policy=policy_cf if focal >= 0 else policy_base
        grad_pre,grad_post=_step(model,optimizer,ppo,actor_params,objective)
        if _optimizer_counter(optimizer,actor_params)!=4897: raise ValueError("order-check Adam counter did not advance exactly once")
        model.eval()
        actual,behavior,base,target,odds=_metrics_after(model,ppo,boards,old_behavior,old_base,heldout_means)
        return {"policy_loss":float(policy.detach()),"entropy":float(entropy.detach()),"gradient_norm_before_clip":grad_pre,"gradient_norm_after_clip":grad_post,"target_probabilities":target.tolist(),"logodds":odds.tolist(),"heldout_return_weighted_change_by_state":actual["heldout_return_weighted_policy_change_by_state"]}
    arm_orders={}
    for order_name,order in (("baseline_then_replacement",[-1,0]),("replacement_then_baseline",[0,-1])):
        arm_orders[order_name]={"baseline":None,"replacement":None}
        for focal in order:
            arm="baseline" if focal<0 else "replacement"
            arm_orders[order_name][arm]=run_order_arm(0,focal)
    order_diffs={}
    for arm in ("baseline","replacement"):
        a=arm_orders["baseline_then_replacement"][arm];b=arm_orders["replacement_then_baseline"][arm]
        order_diffs[arm]={"max_target_probability_error":float(np.max(np.abs(np.asarray(a["target_probabilities"])-np.asarray(b["target_probabilities"])))),"max_logodds_error":float(np.max(np.abs(np.asarray(a["logodds"])-np.asarray(b["logodds"])))),"max_scalar_error":max(abs(a[k]-b[k]) for k in ("policy_loss","entropy","gradient_norm_before_clip","gradient_norm_after_clip")),"max_heldout_value_change_error":float(np.max(np.abs(np.asarray(a["heldout_return_weighted_change_by_state"])-np.asarray(b["heldout_return_weighted_change_by_state"]))))}
    tolerances=lock["checks_locked_before_execution"]["baseline_tolerances"]
    order_pass=all(x["max_target_probability_error"]<=tolerances["max_target_behavior_probability_error"] and x["max_logodds_error"]<=tolerances["max_target_preferred_logodds_error"] and x["max_scalar_error"]<=tolerances["scalar_step_metrics_atol"] and x["max_heldout_value_change_error"]<=1e-10 for x in order_diffs.values())
    write_json(folder/"baseline_intervention_order_check.json",{"replicate":0,"focal_state":0,"orders":{"baseline_then_replacement":["baseline","replacement"],"replacement_then_baseline":["replacement","baseline"]},"max_differences_by_arm":order_diffs,"passed":order_pass,"tolerances":tolerances,"qa_only_not_in_primary_outcomes":True})
    if not order_pass: raise ValueError("baseline/focal replacement execution order changed one-step outcomes")
    if args.gate_only:
        print("Baseline/focal execution-order gate passed; no intervention outcomes were collected.",flush=True)
        return
    result_rows=[]; optimizer_restore_errors=[]; critic_errors=[]
    for rep in range(12):
        rows,counts,action_records,row_ids,actions,adv,old_logs,weights=samples[rep]
        # Recreate the identical baseline gradient only to decompose the other-95-state and entropy projections.
        opt_hash=_restore(model,optimizer,capture,lr,actor_params)
        if opt_hash != baseline["optimizer_state_hashes_before"][rep]: raise ValueError("captured optimizer did not restore to baseline hash")
        objective,bpolicy,_,entropy,ratios,logs=_loss_components(model,ppo,states,masks,row_ids,actions,adv,old_logs,weights,mean=mean,std=std,denom=denom)
        policy_gradient=torch.autograd.grad(bpolicy,actor_params,retain_graph=True,allow_unused=True)
        entropy_loss=-ppo.config.entropy_coefficient*entropy
        entropy_gradient=torch.autograd.grad(entropy_loss,actor_params,retain_graph=True,allow_unused=True)
        policy_asc=-_flat_grads(policy_gradient,actor_params).numpy()
        entropy_asc=-_flat_grads(entropy_gradient,actor_params).numpy()
        for focal in range(96):
            # Reset model/Adam before every focal intervention; clone isolation is checked by hashing capture state.
            reset_hash=_restore(model,optimizer,capture,lr,actor_params)
            if reset_hash != opt_hash: raise ValueError("Adam state aliasing/carryover detected")
            objective,policy_base,policy_cf,entropy,ratio,log_probs=_loss_components(model,ppo,states,masks,row_ids,actions,adv,old_logs,weights,
                focal_id=focal,boards=boards,pools=pools,old_prob=old_prob,mean=mean,std=std,denom=denom)
            # Recompute baseline components only for this focal's scalar decomposition; do not use heldout data in gradients.
            focal_mask=row_ids==focal
            normalized_adv=(adv-mean)/(std+denom)
            focal_loss=(weights[focal_mask]*clipped_policy_loss(torch.exp(log_probs[row_ids[focal_mask],actions[focal_mask]]-old_logs[focal_mask]),normalized_adv[focal_mask],ppo.config.clip_epsilon)).sum()
            sample_proj_by_action={}; ref_proj_by_action={}; advantage_summary={}
            for a in boards[focal]["legal_actions"]:
                a=int(a); sampled_vals=action_records[focal][a]; n=len(sampled_vals)
                sample_proj_by_action[str(a)]=float(sum(sampled_vals)/(96.0*32.0)*score_projections[focal][a])
                ref_vals=[(float(x["gae"])-mean)/(std+denom) for x in pools[(focal,a)]]
                ref_proj_by_action[str(a)]=float(old_prob[focal][a]*np.mean(ref_vals)/96.0*score_projections[focal][a])
                advantage_summary[str(a)]={"sample_count":n,"sample_mean_normalized_advantage":float(np.mean(sampled_vals)) if n else None,"sample_sd_normalized_advantage":float(np.std(sampled_vals,ddof=1)) if n>1 else None,"sample_positive_fraction":float(np.mean(np.asarray(sampled_vals)>0)) if n else None,"reference_count":len(ref_vals),"reference_mean_normalized_advantage":float(np.mean(ref_vals)),"reference_sd_normalized_advantage":float(np.std(ref_vals,ddof=1))}
            sample_proj=float(sum(sample_proj_by_action.values())); ref_proj=float(sum(ref_proj_by_action.values()))
            base_policy_projection=float(np.dot(policy_asc,logodds_gradients[focal].numpy()))
            other95_projection=base_policy_projection-sample_proj
            entropy_projection=float(np.dot(entropy_asc,logodds_gradients[focal].numpy()))
            # The focal loss is read only to assert that the selected rows match the recorded manifest.
            del focal_loss
            grad_pre,grad_post=_step(model,optimizer,ppo,actor_params,objective)
            if _optimizer_counter(optimizer,actor_params)!=4897:raise ValueError("Adam did not perform exactly one update")
            model.eval()
            actual,behavior,base,target,odds=_metrics_after(model,ppo,boards,old_behavior,old_base,heldout_means)
            baseline_rep=baselines_by_rep[rep]
            base_odds=np.asarray(baseline_rep["target_preferred_logodds_96"]); base_target=np.asarray(baseline_rep["target_behavior_probabilities_96"])
            paired_focal_logodds=float(odds[focal]-base_odds[focal]); paired_all_mean_logodds=float(np.mean(odds-base_odds))
            paired_focal_target_behavior=float(target[focal]-base_target[focal])
            paired_all_mean_target_behavior=float(np.mean(target-base_target))
            base_value=np.asarray(baseline_rep["heldout_return_weighted_policy_change_by_state"])
            cf_value=np.asarray(actual["heldout_return_weighted_policy_change_by_state"])
            board=boards[focal]; target_action=int(board["critic_action"]); preferred=int(board["actor_action"])
            action_counts=counts[focal]
            result_rows.append({"replicate":rep,"focal_state":focal,"source_episode_id":board["source_episode_id"],"critic_target_action":target_action,"actor_preferred_action":preferred,
                "critic_target_count":action_counts[target_action],"actor_preferred_count":action_counts[preferred],"critic_target_missing":action_counts[target_action]==0,
                "sample_action_counts":{str(a):int(action_counts[int(a)]) for a in board["legal_actions"]},"per_action_advantage_summary":advantage_summary,
                "per_action_ratio_one_sample_ascent_logodds_contributions":sample_proj_by_action,"per_action_ratio_one_reference_ascent_logodds_contributions":ref_proj_by_action,
                "sampled_focal_policy_projection":sample_proj,"finite_pool_reference_focal_policy_projection":ref_proj,"sampled_full96_policy_projection":base_policy_projection,"other95_policy_projection_approx":other95_projection,"entropy_bonus_projection_separate":entropy_projection,
                "baseline_target_logodds_change":float(base_odds[focal]-old_odds[focal]),"replacement_target_logodds_change":float(odds[focal]-old_odds[focal]),"paired_replacement_minus_baseline_target_logodds_change":paired_focal_logodds,
                "baseline_mean_all96_logodds_change":float(np.mean(base_odds-old_odds)),"replacement_mean_all96_logodds_change":float(np.mean(odds-old_odds)),"paired_replacement_minus_baseline_mean_all96_logodds_change":paired_all_mean_logodds,
                "baseline_target_base_actor_probability_change":float(baseline_rep["target_base_actor_probability_change_by_state"][focal]),"replacement_target_base_actor_probability_change":float(base[focal,target_action]-old_base[focal,target_action]),"paired_replacement_minus_baseline_target_base_probability":float(base[focal,target_action]-baseline_rep["target_base_actor_probabilities_96"][focal]),
                "baseline_target_behavior_probability_change":float(base_target[focal]-old_target[focal]),"replacement_target_behavior_probability_change":float(target[focal]-old_target[focal]),"paired_replacement_minus_baseline_target_behavior_probability":paired_focal_target_behavior,
                "baseline_focal_heldout_return_weighted_change":float(base_value[focal]),"replacement_focal_heldout_return_weighted_change":float(cf_value[focal]),"paired_replacement_minus_baseline_focal_heldout_return_weighted_change":float(cf_value[focal]-base_value[focal]),
                "baseline_all96_heldout_return_weighted_change":float(np.mean(base_value)),"replacement_all96_heldout_return_weighted_change":float(np.mean(cf_value)),"paired_replacement_minus_baseline_all96_heldout_return_weighted_change":float(np.mean(cf_value-base_value)),
                "target_minus_preferred_heldout_return_estimate":float(heldout_means[focal][target_action]-heldout_means[focal][preferred]),"max_abs_ratio_from_one_at_pre_step":float(np.max(np.abs(ratio.detach().cpu().numpy()-1))),
                "replacement_gradient_norm_before_clip":grad_pre,"replacement_gradient_norm_after_clip":grad_post,"adam_step":_optimizer_counter(optimizer,actor_params)})
            if rep%3==2 and focal%24==23:print(f"completed matched interventions: rep {rep+1}/12, focal {focal+1}/96",flush=True)
    out={"scope":"one focal board per independent replacement; 12 existing natural draws; all-action finite training-pool expectation; one Adam step","matched_plan_sha256":lock["lock_sha256"],"bank_lock_sha256":bank_plan["lock_sha256"],"draw_manifest_sha256":lock["draw_manifest_sha256"],"direct_gradient_validation":check,"pairing":"each intervention uses same natural 96-state rows as its paired baseline and restores captured params/Adam; entropy unchanged; clip after combined gradient","pool_and_draw_hashes_unchanged":True,"rows":result_rows}
    write_json(folder/"matched_intervention_results.json",out)
    print(f"Saved {len(result_rows)} focal-state interventions; direct-gradient check max error {check['max_absolute_gradient_l2_error']:.3g}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    for name in ("baseline","interventions"):
        sp=sub.add_parser(name);sp.add_argument("--folder",type=Path,default=OUT)
        if name=="interventions": sp.add_argument("--gate-only",action="store_true",help="run the locked baseline/focal restore-order check, then stop")
    args=parser.parse_args()
    if args.command=="baseline":_baseline(args)
    else:_interventions(args)

if __name__=="__main__":main()
