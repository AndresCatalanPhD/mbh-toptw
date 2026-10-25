"""
run_campaign.py -- same campaign as the Colab notebook, for a local PC with gurobipy + WLS licence.

    python run_campaign.py --inst ../instances --out ../results [--campaigns benchmark scalability random] [--threads 8]

Finished runs are appended to <out>/runs.csv; re-running skips them. Stop with Ctrl-C at any time.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbh_experiments import Campaign

ap = argparse.ArgumentParser()
ap.add_argument('--inst', default='../instances'); ap.add_argument('--out', default='../results')
ap.add_argument('--campaigns', nargs='*', default=['benchmark', 'scalability', 'random'])
ap.add_argument('--methods', nargs='*', default=['CG-TS', 'CG', 'MILP'])
ap.add_argument('--tl_milp_bench', type=float, default=3600); ap.add_argument('--tl_cg_bench', type=float, default=3600)
ap.add_argument('--tl_milp_rand', type=float, default=1800); ap.add_argument('--tl_cg_rand', type=float, default=1800)
ap.add_argument('--sp_time', type=float, default=60); ap.add_argument('--final_mip', type=float, default=600)
ap.add_argument('--threads', type=int, default=0); ap.add_argument('--filter', nargs='*', default=None)
a = ap.parse_args()

import gurobipy as gp
ENV = gp.Env(); ENV.setParam('OutputFlag', 0)
RES = Campaign(a.out, env=ENV, threads=a.threads, host=f'pc-{os.cpu_count()}cpu')
order_bench = [f'pr{k:02d}' for k in (11, 17, 1, 7, 12, 2, 18, 8, 13, 3, 19, 9, 14, 4, 20, 10, 15, 5, 16, 6)]
jobs = []
if 'benchmark' in a.campaigns:
    jobs += [('benchmark', f'{a.inst}/benchmark_het/{n}', m, a.tl_milp_bench if m == 'MILP' else a.tl_cg_bench) for n in order_bench for m in a.methods]
if 'scalability' in a.campaigns:
    jobs += [('scalability', f'{a.inst}/scalability/{n}', m, a.tl_milp_bench if m == 'MILP' else a.tl_cg_bench) for n in sorted(os.listdir(f'{a.inst}/scalability')) for m in a.methods]
    if os.path.isdir(f'{a.inst}/scalability_hard'):
        jobs += [('scalability_hard', f'{a.inst}/scalability_hard/{n}', m, a.tl_milp_bench if m == 'MILP' else a.tl_cg_bench) for n in sorted(os.listdir(f'{a.inst}/scalability_hard')) for m in a.methods]
if 'random' in a.campaigns:
    names = sorted(os.listdir(f'{a.inst}/random'))
    if a.filter: names = [n for n in names if n.split('_')[0] in a.filter]
    names.sort(key=lambda n: (int(n.split('_')[1].split('a')[1].split('v')[0]), n))
    jobs += [('random', f'{a.inst}/random/{n}', m, a.tl_milp_rand if m == 'MILP' else a.tl_cg_rand) for n in names for m in a.methods]
if 'real' in a.campaigns and os.path.isdir(f'{a.inst}/real'):
    jobs += [('real', f'{a.inst}/real/{n}', m, a.tl_cg_bench) for n in sorted(os.listdir(f'{a.inst}/real')) if not n.endswith('.json') for m in a.methods]
print(f'{len(jobs)} jobs; already done: {len(RES.clean())}')
t0 = time.time()
for k, (camp, path, m, tl) in enumerate(jobs, 1):
    row = RES.run(camp, path, m, tl, sp_time=a.sp_time, final_mip_time=a.final_mip)
    if row is None: continue
    print(f"[{k}/{len(jobs)} {(time.time()-t0)/3600:.2f}h] {camp} {os.path.basename(path)} {m}: value={row.get('value')} bound={row.get('bound')} time={row.get('time')} {row.get('status')}", flush=True)
