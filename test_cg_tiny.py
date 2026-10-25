import numpy as np, random, time, sys, copy
sys.path.insert(0,'.')
from mbh_core import *

class EnumPricer:
    """Exact pricing by enumeration (tiny instances) with the GurobiPricer interface."""
    class _M:  # mimics .m.Params.TimeLimit
        class Params: TimeLimit=60
    m=_M(); time_limit=60
    def __init__(self, inst, r, max_len=6):
        self.inst, self.r = inst, r; self.routes = enumerate_routes(inst, r, max_len)
    def solve(self, pi, lam_r, min_rc=1e-4, pool_size=10):
        gain = self.inst.score - pi
        vals = [(float(gain[rt].sum()) - lam_r if rt else -lam_r, rt, s, e) for (s,e,rt,dur) in self.routes]
        vals.sort(key=lambda x:-x[0]); best = vals[0][0]
        cols = [v for v in vals[:pool_size] if v[0] > min_rc]
        return True, max(0.0,best), cols

def sub_instance(inst, areas, bases, aircraft):
    """Tiny sub-instance: keep given areas (indices) and bases; aircraft restricted; start bases remapped."""
    air=[]
    for k in aircraft:
        a=inst.air[k]; st = bases.index(a.start) if a.start in bases else 0
        air.append(Aircraft(a.T,a.v,a.w,st))
    s=Instance('tiny', len(areas), len(bases), len(air), air, inst.ax[areas], inst.ay[areas], inst.size[areas], inst.score[areas],
               inst.o[areas], inst.c[areas], inst.bx[bases], inst.by[bases])
    return s.build()

base = read_instance('../instances_src/pr01')
rng = random.Random(7); tot=0; agree_exact=0; ts_within=0
for trial in range(12):
    nA = rng.choice([6,7,8]); areas = sorted(rng.sample(range(base.V), nA)); bases=[0,1]
    inst = sub_instance(base, areas, bases, [0,1])
    opt, sol = brute_force_optimum(inst, max_len=6)
    # CG exact (enum pricing) + TS warm start, SimplexRMP backend
    rmp = SimplexRMP(inst); ts=[TabuPricer(inst,r,seed=r) for r in range(inst.R)]; ex=[EnumPricer(inst,r) for r in range(inst.R)]
    res = column_generation(inst, rmp, ts, ex, verbose=False, final_mip_time=30)
    ok, val = verify_solution(inst, res.routes)
    # CG-TS only
    rmp2 = SimplexRMP(inst); ts2=[TabuPricer(inst,r,seed=10+r) for r in range(inst.R)]
    res2 = column_generation(inst, rmp2, ts2, None, verbose=False, final_mip_time=30)
    ok2, val2 = verify_solution(inst, res2.routes)
    tot+=1
    flag_ub = res.lagr_bound >= opt-1e-6; flag_lb = res.integer_value <= opt+1e-6 and abs(val-res.integer_value)<1e-6 and ok
    agree_exact += (abs(res.integer_value-opt)<1e-6)
    ts_within += (abs(val2-opt)<1e-6)
    print(f"tiny#{trial:2d} V={nA} opt={opt:5.1f} | CG: LP={res.lp_bound:7.3f} UB={res.lagr_bound:7.3f} int={res.integer_value:5.1f} proved={res.proved} it={res.iterations} feasOK={ok} | CG-TS: int={res2.integer_value:5.1f} feasOK={ok2} | UBvalid={flag_ub} LBvalid={flag_lb}")
    assert flag_ub and flag_lb and ok2
print(f"\n{tot} tiny instances: CG integer == brute-force optimum in {agree_exact}/{tot}; CG-TS == optimum in {ts_within}/{tot}; all bounds valid")
