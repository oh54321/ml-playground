import json, numpy as np
from pathlib import Path
from collections import Counter
base=Path('artifacts/ppo_2048/ppo_matched_action_coverage_iter100')
raw=json.load(open(base/'matched_intervention_results.json'))
boards=json.load(open('artifacts/ppo_2048/ppo_broad_gradient_audit_iter100/lockedplan.json'))
allboards=boards['existing_boards']+boards['new_boards']
rows=raw['rows']; nrep=12;nstate=96
mat=lambda key: np.array([[next(r[key] for r in rows if r['replicate']==rep and r['focal_state']==s) for s in range(nstate)] for rep in range(nrep)],dtype=float)
y=mat('paired_replacement_minus_baseline_target_logodds_change')
hv=mat('paired_replacement_minus_baseline_focal_heldout_return_weighted_change')
prob_b=mat('paired_replacement_minus_baseline_target_base_probability')
prob_mix=mat('paired_replacement_minus_baseline_target_behavior_probability')
repmean=y.mean(axis=1);point=float(y.mean())
episodes=[b['source_episode_id'] for b in allboards]; uniq=sorted(set(episodes)); epind={e:i for i,e in enumerate(uniq)};counts=np.bincount([epind[e] for e in episodes],minlength=len(uniq));
rng=np.random.default_rng(20482171); B=20000
bootrep=repmean[rng.integers(0,nrep,size=(B,nrep))].mean(axis=1)
rng=np.random.default_rng(20482171); br=[]
for _ in range(B):
    rs=rng.integers(0,nrep,size=nrep); es=rng.integers(0,len(uniq),size=len(uniq)); w=np.bincount(es,minlength=len(uniq)); sw=np.array([w[epind[e]] for e in episodes],dtype=float)
    br.append(float(np.mean([np.sum(y[r]*sw)/sw.sum() for r in rs])))

targ_count=np.array([r['critic_target_count'] for r in rows]).reshape(nrep,nstate)
targ_missing= targ_count==0
base_target_p=np.array([float(allboards[i]['actor_probabilities'][int(allboards[i]['critic_action'])]) for i in range(nstate)])
strata={'<1%':base_target_p<.01,'1%-<10%':(base_target_p>=.01)&(base_target_p<.10),'>=10%':base_target_p>=.10}
subset=json.load(open('artifacts/ppo_2048/ppo_broad_gradient_audit_iter100/heldout_target_outcomes.json'))
pos=[int(r['id']) for r in subset['per_board'] if r.get('simultaneous_interval_excludes_zero') and r['mean_discounted_return_difference_target_minus_preferred']>0]
# derive historical exact pair ids from parent research record
hist=[0,1,2,3,5]
metrics={}
for label,idx in {'all96':np.arange(96),'max_t_positive8':np.array(pos),'historical5':np.array(hist)}.items():
    z=y[:,idx]
    metrics[label]={'n_states':len(idx),'mean_paired_logodds_change':float(z.mean()),'mean_paired_target_base_probability_change':float(prob_b[:,idx].mean()),'mean_paired_target_behavior_probability_change':float(prob_mix[:,idx].mean()),'positive_mean_state_count':int((z.mean(axis=0)>0).sum()),'negative_mean_state_count':int((z.mean(axis=0)<0).sum()),'mean_per_replicate':z.mean(axis=1).tolist(),'own_sample_projection_positive_count':int(np.mean(np.array([r['sampled_focal_policy_projection'] for r in rows]).reshape(nrep,nstate)[:,idx],axis=0).gt(0).sum()) if False else None}
# action score signs
sample_proj=np.array([r['sampled_focal_policy_projection'] for r in rows]).reshape(nrep,nstate)
ref_proj=np.array([r['finite_pool_reference_focal_policy_projection'] for r in rows]).reshape(nrep,nstate)
# other-state adverse around own projection signs
other_proj=np.array([r['other95_policy_projection_approx'] for r in rows]).reshape(nrep,nstate)
summary={'locked_plan_sha256':json.load(open(base/'locked_plan.json'))['lock_sha256'],'amendment_sha256':json.load(open(base/'amendment_v2.json'))['amendment_sha256'],'n_states':96,'n_source_episodes':len(uniq),'paired_replicates':12,'n_counterfactual_updates':len(rows),'direct_gradient_validation':raw['direct_gradient_validation'],'paired_target_probability_effects':{'base_actor_target_probability_mean':float(prob_b.mean()),'base_actor_target_probability_positive_count':int((prob_b>0).sum()),'base_actor_target_probability_negative_count':int((prob_b<0).sum()),'behavior_target_probability_mean':float(prob_mix.mean()),'behavior_target_probability_positive_count':int((prob_mix>0).sum()),'behavior_target_probability_negative_count':int((prob_mix<0).sum())},'primary':{'estimand':'Equal-state mean over 96 states and 12 shared natural draws of focal-state post-step minus pre-step base-actor target/preferred log odds under replacement, minus paired natural baseline.','point_mean':point,'replicate_means':repmean.tolist(),'bootstrap_replicate_only_95pct_interval':np.quantile(bootrep,[.025,.975]).tolist(),'bootstrap_episode_cluster_plus_paired_replicate_95pct_interval':np.quantile(br,[.025,.975]).tolist(),'bootstrap_draws':B,'seed':20482171},'coverage':{'target_samples_missing':int(targ_missing.sum()),'target_sample_slots':int(targ_count.sum()),'total_draws':int(targ_count.size),'fraction_missing':float(targ_missing.mean()),'states_missing_in_at_least_one_draw':int(targ_missing.any(axis=0).sum()),'states_missing_in_all_12_draws':int(targ_missing.all(axis=0).sum()),'by_actor_target_probability_stratum':{}},'projections':{'sample_projection_positive_count':int((sample_proj>0).sum()),'reference_projection_positive_count':int((ref_proj>0).sum()),'reference_positive_sample_nonpositive_count':int(((ref_proj>0)&(sample_proj<=0)).sum()),'reference_negative_sample_positive_count':int(((ref_proj<0)&(sample_proj>0)).sum()),'other95_projection_opposes_positive_reference_count':int(((ref_proj>0)&(other_proj<0)).sum()),'target_sample_missing_and_reference_positive_count':int((targ_missing&(ref_proj>0)).sum()),'sample_projection_mean':float(sample_proj.mean()),'reference_projection_mean':float(ref_proj.mean()),'other95_projection_mean':float(other_proj.mean())},'heldout_alignment_proxy':{'mean_focal_replacement_minus_baseline':float(hv.mean()),'mean_all96_change_within_interventions':float(np.mean([r['paired_replacement_minus_baseline_all96_heldout_return_weighted_change'] for r in rows])),'description':'Fixed-pool local alignment proxy; not post-update game performance.'},'subsets':metrics}
for name,mask in strata.items():
 i=np.where(mask)[0]; tt=targ_count[:,i]; yp=y[:,i]
 summary['coverage']['by_actor_target_probability_stratum'][name]={'n_states':len(i),'mean_paired_target_base_probability_change':float(prob_b[:,i].mean()),'mean_paired_target_behavior_probability_change':float(prob_mix[:,i].mean()),'target_samples_missing':int((tt==0).sum()),'sample_slots':int(tt.size),'missing_fraction':float((tt==0).mean()),'mean_target_count_per_draw':float(tt.mean()),'mean_paired_logodds_change':float(yp.mean()),'replicate_means':yp.mean(axis=1).tolist()}
summary['subsets']['max_t_positive8']={'ids':pos,**metrics['max_t_positive8']}
summary['subsets']['historical5']={'ids':hist,**metrics['historical5']}
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
