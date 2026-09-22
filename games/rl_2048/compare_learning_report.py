"""Summarize comparison experiments and render ordinary PNG plots.

--prepare needs NumPy; --render needs Pillow. Neither phase changes training.
"""
import argparse
import json
import math
from pathlib import Path


LABELS = ['separate', 'shared', 'small']
COLORS = [(31,119,180), (214,39,40), (44,160,44)]


def read(folder, name):
    return json.loads((folder / name).read_text())


def prepare(folder):
    import numpy as np
    rng = np.random.default_rng(90351)
    meta = read(folder, 'metadata.json')
    output = dict(models={}, paired={}, windows={})
    samples = {}
    for label in LABELS:
        output['models'][label] = {}
        for mode in ('sampled', 'greedy'):
            data = read(folder, f'{label}_{mode}.json')
            games = data['games']
            scores = np.array([g['score'] for g in games])
            samples[label, mode] = scores
            indices = rng.integers(0, len(games), size=(2000, len(games)))
            summary = {k:v for k,v in data.items() if k != 'games'}
            summary.update(p10=float(np.quantile(scores,.1)), below_2000=float(np.mean(scores<2000)),
                tile_2048=float(np.mean([g['max_tile']>=2048 for g in games])),
                tile_1024=float(np.mean([g['max_tile']>=1024 for g in games])),
                mean_moves=float(np.mean([g['moves'] for g in games])),
                entropy_game_mean=float(np.mean([g['entropy'] for g in games])),
                empty_cells_game_mean=float(np.mean([g['empty_cells'] for g in games])),
                max_in_corner_game_mean=float(np.mean([g['max_in_corner'] for g in games])),
                forced_fraction_game_mean=float(np.mean([g['forced_fraction'] for g in games])),
                avoidable_terminal_fraction=float(np.mean([g['avoidable_terminal'] for g in games])),
                mean_terminal_value=float(np.mean([g['terminal_value'] for g in games])),
                risk_opportunities=sum(g['risk_opportunities'] for g in games),
                avoidable_mass_at_risk_opportunities=sum(g['avoidable_mass_sum'] for g in games)/max(1,sum(g['risk_opportunities'] for g in games)))
            calibration = {}
            for stratum in ('all','open','crowded','early_tile','late_tile','last_20_moves'):
                rows = [g['calibration'].get(stratum,{}) for g in games]
                n = np.array([r.get('n',0) for r in rows])
                pred = np.array([r.get('predicted_sum',0) for r in rows])
                ret = np.array([r.get('return_sum',0) for r in rows])
                errors = np.array([r.get('error_square_sum',0) for r in rows])
                bias_samples = (pred-ret)[indices].sum(1)/n[indices].sum(1)
                calibration[stratum] = dict(states=int(n.sum()), prediction=float(pred.sum()/n.sum()),
                    actual_remaining=float(ret.sum()/n.sum()), bias=float((pred-ret).sum()/n.sum()),
                    bias_cluster_ci95=np.quantile(bias_samples,[.025,.975]).tolist(),
                    mc_rmse=float(np.sqrt(errors.sum()/n.sum())))
            summary['calibration'] = calibration
            output['models'][label][mode] = summary
        output['windows'][label] = []
        for lo, hi in [(1,100),(101,200),(401,500),(501,600),(601,670),(801,900),(901,950),(1201,1300)]:
            rows=[r for r in meta[label]['training'] if lo<=r['iteration']<=hi and 'policy_kl_between_iterations' in r]
            evals=[e['mean'] for e in meta[label]['evaluations'] if lo<=e['iteration']<=hi]
            if not rows or not evals:
                continue
            row=dict(start=lo,end=hi,n=len(rows),score_mean=float(np.mean(evals)),epochs=float(np.mean([len(r['explained_variance']) for r in rows])))
            for key in ('learning_rate','policy_kl_between_iterations','parameter_update','value_policy_ratio','gradient_cosine'):
                row[key]=float(np.mean([r[key] for r in rows if key in r]))
            for key in ('explained_variance','clip_fraction','negative_entropy'):
                row[key]=float(np.mean([r[key][-1] for r in rows]))
            output['windows'][label].append(row)
    for mode in ('greedy','sampled'):
        for a,b in [('separate','shared'),('small','separate')]:
            delta=samples[a,mode]-samples[b,mode]
            bootstrap=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
            output['paired'][f'{a}_minus_{b}_{mode}']=dict(mean=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),n=len(delta))
    (folder/'aggregate.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output,indent=2))


def render(folder):
    from games.rl_2048.plot_training_diagnostics import save_line_plot
    meta=read(folder,'metadata.json')
    # All runs evaluated every ten iterations; truncated series stop at their
    # last observation instead of inventing values beyond the captured history.
    common=range(10,2001,10)
    score_series=[]
    for label,color in zip(LABELS,COLORS):
        scores=[e['mean'] for e in meta[label]['evaluations']]
        smoothed=[sum(scores[max(0,i-4):i+1])/len(scores[max(0,i-4):i+1]) for i in range(len(scores))]
        score_series.append((label,smoothed,color))
    save_line_plot(folder/'score_by_iteration.png',common,score_series,
        'Evaluation score: trailing 5 evaluations (same fixed 100 seeds)',y_label='Mean game score (points)',y_min=0)
    # Environment-step matched comparison, on an evenly spaced interpolation
    # grid. Interpolation is only within each run's observed evaluation range.
    step_grid=list(range(327680,32768001,327680))
    step_series=[]
    for label,color in zip(LABELS,COLORS):
        config=meta[label]['ppo_config']
        steps_per_iteration=config['num_parallel_games']*config['steps_per_game_per_rollout']
        observations=meta[label]['evaluations']
        xs=[e['iteration']*steps_per_iteration for e in observations]
        ys=[e['mean'] for e in observations]
        values=[]
        j=0
        for x in step_grid:
            if x>xs[-1]:break
            while j+1<len(xs) and xs[j+1]<x:j+=1
            if j+1==len(xs):values.append(ys[j])
            else:values.append(ys[j]+(ys[j+1]-ys[j])*(x-xs[j])/(xs[j+1]-xs[j]))
        smooth=[sum(values[max(0,i-4):i+1])/len(values[max(0,i-4):i+1]) for i in range(len(values))]
        step_series.append((label,smooth,color))
    save_line_plot(folder/'score_by_environment_steps.png',[v/1e6 for v in step_grid],step_series,
        'Score vs collected data: interpolated, trailing 5 grid points',x_label='Environment transitions (millions)',y_label='Mean game score (points)',y_min=0)
    for key,title,ylabel,logscale in [
        ('policy_kl_between_iterations','Policy movement: 50-iteration means','KL(old || new)',False),
        ('learning_rate','Learning-rate schedules: 50-iteration means','log10 learning rate',True),
        ('epochs','Adaptive epoch counts: 50-iteration means','Epochs per rollout',False),
        ('explained_variance','Training-target explained variance: 50-iteration means','End-of-epoch EV (not Monte Carlo calibration)',False),
    ]:
        series=[]
        for label,color in zip(LABELS,COLORS):
            records=[r for r in meta[label]['training'] if 'policy_kl_between_iterations' in r]
            vals=[]
            for end in range(50,len(records)+1,50):
                rows=records[end-50:end]
                if key=='epochs':v=sum(len(r['explained_variance']) for r in rows)/len(rows)
                elif key=='explained_variance':v=sum(r[key][-1] for r in rows)/len(rows)
                else:v=sum(r[key] for r in rows)/len(rows)
                vals.append(math.log10(max(v,1e-12)) if logscale else v)
            series.append((label,vals,color))
        save_line_plot(folder/f'{key}.png',list(range(50,2001,50)),series,title,y_label=ylabel)
    for mode in ('greedy', 'sampled'):
        series=[]
        for label,color in zip(LABELS,COLORS):
            data=read(folder,f'{label}_{mode}.json')
            scores=sorted(g['score'] for g in data['games'])
            series.append((label,scores,color))
        percentiles=[100*i/(len(scores)-1) for i in range(len(scores))]
        save_line_plot(folder/f'fresh_{mode}_quantiles.png',percentiles,series,
            f'Fresh {mode} evaluation: 96 games per checkpoint',x_label='Game-score percentile (%)',y_label='Game score (points)',y_min=0)
    if all((folder/f'{label}_updates.json').exists() for label in LABELS):
        for key,title,ylabel in [
            ('full_kl','Captured rollout: full policy KL after each epoch','KL(old || new)'),
            ('greedy_action_change_fraction','Captured rollout: decisions changed after each epoch','Fraction with changed greedy action'),
            ('explained_variance','Captured rollout: fitting frozen bootstrap targets','Explained variance'),
        ]:
            series=[]
            for label,color in zip(LABELS,COLORS):
                rows=read(folder,f'{label}_updates.json')['epochs']
                series.append((label,[r[key] for r in rows],color))
            save_line_plot(folder/f'captured_{key}.png',list(range(9)),series,title,
                x_label='Completed PPO update epochs',y_label=ylabel,y_min=0)
    if (folder/'cases.json').exists():
        from PIL import Image, ImageDraw
        from games.rl_2048.plot_training_diagnostics import _font
        cases=read(folder,'cases.json')
        for i,c in enumerate(cases['cases']):
            image=Image.new('RGB',(640,680),'#faf8ef'); draw=ImageDraw.Draw(image)
            draw.text((20,12),f"Case {i}: {c['source']} / {c['selection']}",fill='#333333',font=_font(20))
            for y,row in enumerate(c['board']):
                for x,v in enumerate(row):
                    box=(20+x*150,65+y*150,160+x*150,205+y*150)
                    color='#cdc1b4' if v==0 else '#eee4da' if v<8 else '#f2b179' if v<128 else '#edc850'
                    draw.rounded_rectangle(box,radius=7,fill=color)
                    draw.text((90+x*150,135+y*150),str(v) if v else '',anchor='mm',font=_font(30),fill='#333333')
            image.save(folder/f'case_{i}.png')
    print('Rendered comparison plots in',folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.folder)
    if args.render:render(args.folder)
