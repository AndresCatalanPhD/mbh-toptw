"""
test_model_structure.py -- checks, WITHOUT a solver, that every truly feasible solution (route schedules verified by
schedule()) satisfies all constraints of the compact MILP (solve_milp) and of the exact pricing model (GurobiPricer),
and that the model objective equals the collected score. Uses fake_gurobipy, which records the models symbolically.

    python test_model_structure.py            (from code/)
"""
import sys, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'fake_gurobipy')); sys.path.insert(0, HERE)
import gurobipy as fg                     # the fake one
import numpy as np
from mbh_core import *

def sub_instance(inst, areas, bases, aircraft):
    air = []
    for k in aircraft:
        a = inst.air[k]; st = bases.index(a.start) if a.start in bases else 0
        air.append(Aircraft(a.T, a.v, a.w, st))
    s = Instance('tiny', len(areas), len(bases), len(air), air, inst.ax[areas], inst.ay[areas], inst.size[areas], inst.score[areas],
                 inst.o[areas], inst.c[areas], inst.bx[bases], inst.by[bases])
    return s.build()

def times_along(inst, r, route, s, e):
    """departure time, service start per area, arrival time at the end base (same schedule as schedule())"""
    ok, dur, dmax, cum = schedule(inst, r, route, s, e)
    assert ok
    tt, tau = inst.tt[r], inst.tau[r]
    d = dmax; t = d; prev = inst.V + s; starts = {}
    for k in route:
        arr = t + tau[prev] + tt[prev, k]; t = max(arr, inst.o[k]); starts[k] = t; prev = k
    end = t + tau[prev] + tt[prev, inst.V + e]
    assert abs((end - d) - dur) < 1e-6, (end - d, dur)
    return d, starts, end

def assign_route(m, prefix_x, prefix_s, prefix_e, inst, r, route, s, e, assign):
    V = inst.V
    X = m.vardicts[prefix_x]; S = m.vardicts[prefix_s]; E = m.vardicts.get(prefix_e)
    for i in range(V): assign[S[i]] = float(inst.o[i])          # unvisited areas sit at the window opening
    if not route: return
    d, starts, end = times_along(inst, r, route, s, e)
    nodes = [V + s] + list(route) + [V + e]
    for a, b in zip(nodes[:-1], nodes[1:]): assign[X[a, b]] = 1.0
    for k, t in starts.items(): assign[S[k]] = float(t)
    assign[S[V + s]] = float(d)
    if E is not None: assign[E[e]] = float(end)
    else: assign[S[V + e]] = float(end)        # shared variable (old formulation): overwrites the departure if e == s

def run():
    base = read_instance(os.path.join(HERE, '..', 'instances_src', 'pr01'))
    rng = random.Random(3)
    tot = closed = 0; viol_closed = viol_open = 0; obj_bad = 0
    ptot = pclosed = pviol_closed = pviol_open = 0
    for trial in range(6):
        nA = rng.choice([4, 5]); areas = sorted(rng.sample(range(base.V), nA)); bases = [0, 1]
        inst = sub_instance(base, areas, bases, [0, 1])
        # ---- MILP ----
        fg.MODELS.clear(); solve_milp(inst, time_limit=1, env=None); m = fg.MODELS[-1]
        R = [enumerate_routes(inst, r, max_len=4) for r in range(inst.R)]
        combos = [(a, b) for a in R[0] for b in R[1] if not (set(a[2]) & set(b[2]))]
        rng.shuffle(combos); combos = combos[:150]
        for (s0, e0, rt0, _), (s1, e1, rt1, _) in combos:
            assign = {}
            assign_route(m, 'x0', 's0', 'e0', inst, 0, rt0, s0, e0, assign)
            assign_route(m, 'x1', 's1', 'e1', inst, 1, rt1, s1, e1, assign)
            bad, obj = m.check(assign)
            is_closed = (rt0 and s0 == e0) or (rt1 and s1 == e1)
            tot += 1; closed += bool(is_closed)
            if bad:
                if is_closed: viol_closed += 1
                else: viol_open += 1
            if abs(obj - (route_score(inst, rt0) + route_score(inst, rt1))) > 1e-6: obj_bad += 1
        # ---- pricer (aircraft 0) ----
        pr = GurobiPricer(inst, 0, env=None, time_limit=1); pm = pr.m
        for (s, e, rt, _) in R[0]:
            if not rt: continue
            assign = {}; assign_route(pm, 'x', 's', 'e', inst, 0, rt, s, e, assign)
            bad, obj = pm.check(assign)
            ptot += 1; pclosed += (s == e)
            if bad:
                if s == e: pviol_closed += 1
                else: pviol_open += 1
        print(f'tiny#{trial} V={nA}: MILP feasible solutions tested={len(combos)}  | pricer routes tested={sum(1 for x in R[0] if x[2])}')
    print(f'\nMILP : {tot} feasible solutions ({closed} with a closed route): violated by model -> closed {viol_closed}, open {viol_open}; objective mismatches {obj_bad}')
    print(f'PRICER: {ptot} feasible routes ({pclosed} closed): violated by model -> closed {pviol_closed}, open {pviol_open}')
    ok = (viol_closed == viol_open == obj_bad == pviol_closed == pviol_open == 0)
    print('MODEL STRUCTURE OK' if ok else 'MODEL STRUCTURE **FAILS**')
    return ok

if __name__ == '__main__':
    sys.exit(0 if run() else 1)
