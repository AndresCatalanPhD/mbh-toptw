"""test_ts_more.py -- CG-TS against brute force on 30 sub-instances with 8-10 areas (closed routes frequent), plus the
real instance (brute-force optimum and CG-TS on the test LP backend). Run from code/."""
import sys, os, random, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mbh_core import *
from test_model_structure import sub_instance
import mbh_instances
base = read_instance(os.path.join(HERE, '..', 'instances_src', 'pr01')); base2 = read_instance(os.path.join(HERE, '..', 'instances', 'benchmark_het', 'pr07'))
rng = random.Random(11); n = 0; hit = 0; closed_opt = 0; t0 = time.time()
for trial in range(30):
    src = base if trial % 2 == 0 else base2
    nA = rng.choice([8, 9, 10]); areas = sorted(rng.sample(range(src.V), nA)); bases = sorted(rng.sample(range(src.D), 2))
    inst = sub_instance(src, areas, bases, [0, 1])
    opt, sol = brute_force_optimum(inst, max_len=5)
    closed_opt += any(s == e and rt for (s, e, rt) in sol)
    rmp = SimplexRMP(inst); ts = [TabuPricer(inst, r, seed=100 + r) for r in range(inst.R)]
    res = column_generation(inst, rmp, ts, None, verbose=False, final_mip_time=30)
    ok, val = verify_solution(inst, res.routes); n += 1; hit += abs(val - opt) < 1e-6
    assert ok and val <= opt + 1e-6, (val, opt)
print(f'{n} instances (V 8-10, 2 bases, 2 aircraft): CG-TS == brute-force optimum in {hit}/{n}; optimum contains a closed route in {closed_opt}/{n}; {time.time()-t0:.1f}s')
inst = mbh_instances.build_real_instance()
t0 = time.time(); opt, sol = brute_force_optimum(inst, max_len=6); print('real instance brute force:', opt, 'of', inst.score.sum(), f'({time.time()-t0:.1f}s)')
for (s, e, rt) in sol: print('   ', s, '->', rt, '->', e, 'closed' if s == e and rt else '')
rmp = SimplexRMP(inst); ts = [TabuPricer(inst, r, seed=r) for r in range(inst.R)]
res = column_generation(inst, rmp, ts, None, verbose=False, final_mip_time=30)
print('real instance CG-TS:', res.integer_value, 'LP', round(res.lp_bound, 2), 'verify', verify_solution(inst, res.routes))
