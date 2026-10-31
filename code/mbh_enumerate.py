"""mbh_enumerate.py -- complete enumeration of the optimal plans of a small instance (Chilean case study, Section 7).

A plan gives every aircraft a set of areas, possibly empty, and a landing base, such that some visiting order of the set
is a feasible route from the departure base of the aircraft to that landing base. Plans that differ only in the visiting
order within a route are counted once; the flight time of a route is the shortest over its feasible visiting orders.
An aircraft with an empty set stays at its departure base (the empty route is counted once, not once per base).
"""
from __future__ import annotations
from functools import lru_cache
from mbh_core import enumerate_routes, route_score


def route_options(inst, max_len: int = 8):
    """per aircraft: list of (area bitmask, score, shortest duration, start, end, route), one per (set, landing base)"""
    out = []
    for r in range(inst.R):
        g = {}
        for (s, e, rt, dur) in enumerate_routes(inst, r, max_len):
            if not rt and e != s:       # an empty route is the aircraft staying at its base: counted once
                continue
            m = 0
            for i in rt:
                m |= 1 << i
            if (m, e) not in g or dur < g[(m, e)][2]:
                g[(m, e)] = (m, float(route_score(inst, rt)), float(dur), s, e, list(rt))
        out.append(sorted(g.values(), key=lambda t: (t[0], t[4])))
    return out


def optimal_plans(inst, max_len: int = 8, tol: float = 1e-9):
    """(optimum, plans); a plan is a list, one entry per aircraft, of (mask, score, duration, start, end, route)"""
    O = route_options(inst, max_len)
    R = inst.R

    @lru_cache(maxsize=None)
    def best(r, used):
        if r == R:
            return 0.0
        return max(sc + best(r + 1, used | m) for (m, sc, *_x) in O[r] if not m & used)

    opt = best(0, 0)
    plans = []

    def rec(r, used, acc, chosen):
        if r == R:
            plans.append(list(chosen))
            return
        for o in O[r]:
            m, sc = o[0], o[1]
            if m & used:
                continue
            if acc + sc + best(r + 1, used | m) >= opt - tol:
                chosen.append(o)
                rec(r + 1, used | m, acc + sc, chosen)
                chosen.pop()

    rec(0, 0, 0.0, [])
    return opt, plans


def best_with_k_aircraft(inst, k: int, max_len: int = 8):
    """highest score attainable when at most k aircraft fly"""
    O = route_options(inst, max_len)
    R = inst.R

    @lru_cache(maxsize=None)
    def best(r, used, left):
        if r == R:
            return 0.0
        v = best(r + 1, used, left)                     # aircraft r stays on the ground
        if left > 0:
            for (m, sc, dur, *_x) in O[r]:
                if dur > 0 and not (m & used):
                    v = max(v, sc + best(r + 1, used | m, left - 1))
        return v

    return best(0, 0, k)


def summary(inst, max_len: int = 8):
    """optimum, number of optimal plans, plans by number of flying aircraft, best score with one aircraft fewer than the
    smallest optimal fleet, range of total flight time among optimal plans, and the optimal plan of least flight time"""
    opt, plans = optimal_plans(inst, max_len)
    n_fly = [sum(1 for o in pl if o[2] > 0) for pl in plans]
    flight = [sum(o[2] for o in pl) for pl in plans]
    k_min = min(n_fly)
    j = min(range(len(plans)), key=lambda q: (flight[q], n_fly[q]))
    return dict(optimum=opt, total_score=float(inst.score.sum()), n_plans=len(plans),
                plans_by_flying_aircraft={k: n_fly.count(k) for k in sorted(set(n_fly))},
                best_with_one_fewer=best_with_k_aircraft(inst, k_min - 1, max_len) if k_min > 1 else 0.0,
                flight_min=min(flight), flight_max=max(flight), min_flight_plan=plans[j]), plans


def describe(inst, plan):
    names = [a['sector'] for a in inst.meta['areas']]
    bases = [b['name'] for b in inst.meta['bases']]
    lines = []
    for r, (m, sc, dur, s, e, rt) in enumerate(plan):
        cls = inst.meta['aircraft'][r]['cls']
        if dur == 0:
            lines.append(f'{cls:16s} {bases[s]}: on the ground')
        else:
            lines.append(f'{cls:16s} {bases[s]} -> ' + ' -> '.join(names[i] for i in rt) + f' -> {bases[e]}   score {sc:.0f}, {dur:.0f} min')
    return '\n'.join(lines)
