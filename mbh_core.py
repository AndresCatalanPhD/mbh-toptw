"""
mbh_core.py  --  Multi-Base Heterogeneous Team Orienteering Problem with Time Windows (MBH-TOPTW)

Faithful Python port of the Java/CPLEX implementation (Canut-de-Bon, 2019-2026), plus:
  * a route-schedule evaluator with exact minimum-duration computation under time windows,
  * a Tabu Search pricing heuristic (new),
  * column generation with TS pricing certified by an exact MIP pricing (valid Lagrangian bound),
  * a pure-Python LP backend (dense simplex) used only for testing without a commercial solver.

Units (identical to the Java code):
  coordinates in km (Euclidean), speed in km/h, time in MINUTES,
  travel time  t_ij^r  = dist(i,j) * 60 / v_r
  service time tau_i^r = size_i * SIZE_SCALE * 60 / (w_r * v_r)      (SIZE_SCALE = 100: sizes are in 100 km^2)
  a route is feasible if there is a departure time d >= 0 such that every area is served inside its
  window [o_i, c_i] (waiting allowed) and the return to the end base occurs no later than d + T_r.

Node indexing inside this module: areas 0..V-1, bases V..V+D-1 (a base serves both as departure and arrival).
"""
from __future__ import annotations
import math, time, random, os, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

SIZE_SCALE = 100.0
TIME_SCALE = 60.0
INF = float('inf')

# =====================================================================================
#  Instance
# =====================================================================================
@dataclass
class Aircraft:
    T: float            # endurance (min)
    v: float            # speed (km/h)
    w: float            # detection range / sweep width (km)
    start: int          # index of departure base in 0..D-1, or -1 if free

@dataclass
class Instance:
    name: str
    V: int; D: int; R: int
    air: List[Aircraft]
    ax: np.ndarray; ay: np.ndarray; size: np.ndarray; score: np.ndarray; o: np.ndarray; c: np.ndarray
    bx: np.ndarray; by: np.ndarray
    # derived
    tt: List[np.ndarray] = field(default_factory=list)   # tt[r][i,j] minutes over all nodes (V+D)
    tau: List[np.ndarray] = field(default_factory=list)  # tau[r][i] minutes, i in 0..V+D-1 (0 at bases)

    @property
    def N(self): return self.V + self.D
    def is_base(self, i): return i >= self.V

    def build(self):
        X = np.concatenate([self.ax, self.bx]); Y = np.concatenate([self.ay, self.by])
        dist = np.sqrt((X[:, None]-X[None, :])**2 + (Y[:, None]-Y[None, :])**2)
        self.tt, self.tau = [], []
        for a in self.air:
            self.tt.append(dist * TIME_SCALE / a.v)
            t = np.zeros(self.N); t[:self.V] = self.size * SIZE_SCALE * TIME_SCALE / (a.w * a.v)
            self.tau.append(t)
        return self

def read_instance(path: str, name: Optional[str] = None) -> Instance:
    """Reads the adapted-Cordeau format (see Description.txt). Base numbers in the file are V+1..V+D."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    R, V, D = map(int, lines[0][:3])
    air = []
    for k in range(R):
        T, v, w = map(float, lines[1+k][:3]); nb = int(float(lines[1+k][3]))
        start = (nb - V - 1) if nb > 0 else -1
        air.append(Aircraft(T, v, w, start))
    A = np.array([[float(x) for x in ln[:7]] for ln in lines[1+R:1+R+V]])
    B = np.array([[float(x) for x in ln[:3]] for ln in lines[1+R+V:1+R+V+D]])
    inst = Instance(name or os.path.basename(path), V, D, R, air, A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5], A[:, 6], B[:, 1], B[:, 2])
    return inst.build()

def write_instance(inst: Instance, path: str):
    with open(path, 'w') as f:
        f.write(f"{inst.R} {inst.V} {inst.D}\n")
        for a in inst.air:
            f.write(f"{a.T:g} {a.v:g} {a.w:g} {inst.V+1+a.start if a.start >= 0 else 0}\n")
        for i in range(inst.V):
            f.write(f"{i+1} {inst.ax[i]:.3f} {inst.ay[i]:.3f} {inst.size[i]:g} {inst.score[i]:g} {inst.o[i]:g} {inst.c[i]:g}\n")
        for k in range(inst.D):
            f.write(f"{inst.V+1+k} {inst.bx[k]:.3f} {inst.by[k]:.3f} 0 1 0 1000\n")

# =====================================================================================
#  Route evaluation (exact minimum duration with time windows and free departure time)
# =====================================================================================
def schedule(inst: Instance, r: int, route: List[int], start_base: int, end_base: int):
    """route = list of area indices (may be empty).
    Returns (feasible, duration, d_max, cum_end) where d_max is the latest feasible departure time.
    Formulation: arrival at k-th stop = max(d + cum_k, max_{m<k}(o_m + cum_k - cum_m)); start of service >= o_k,
    service must start <= c_k; duration = end - d, minimised by d = d_max = min_k (c_k - cum_k) (>= 0 required)."""
    tt, tau = inst.tt[r], inst.tau[r]
    if not route:
        return True, 0.0, 0.0, 0.0
    T = inst.air[r].T
    nodes = [inst.V + start_base] + list(route) + [inst.V + end_base]
    cum = 0.0; H = 0.0          # H = max_{m<k} (o_m - cum_m), base has o=0,cum=0 -> H starts at 0 (d>=0)
    dmax = INF
    prev = nodes[0]
    for k in nodes[1:]:
        cum += tau[prev] + tt[prev, k]
        if k < inst.V:
            g = inst.c[k] - cum
            if g < H - 1e-9:        # some earlier window forces arrival after c_k
                return False, INF, -1.0, cum
            dmax = min(dmax, g)
            H = max(H, inst.o[k] - cum)
        prev = k
    if dmax < -1e-9:
        return False, INF, dmax, cum
    dmax = max(0.0, min(dmax, 1e12))
    dur = float(cum + max(0.0, H - dmax))      # end = max(d+cum, max_m(o_m)+cum-cum_m) with d = dmax
    # d = dmax minimises duration (duration non-increasing in d). If dmax is +inf (no windows) d=0 is fine.
    return bool(dur <= T + 1e-6), dur, float(dmax), float(cum)

def route_score(inst: Instance, route: List[int]) -> float:
    return float(inst.score[route].sum()) if route else 0.0

# =====================================================================================
#  Column (path) representation for the master problem
# =====================================================================================
@dataclass
class Column:
    r: int
    route: List[int]
    start: int
    end: int
    score: float
    dur: float
    def key(self): return (self.r, self.start, tuple(self.route), self.end)

# =====================================================================================
#  LP / MIP backends for the restricted master problem
# =====================================================================================
class SimplexRMP:
    """Dense primal simplex (Bland) with warm start for the RMP:
       max P z  s.t.  A z <= 1 (areas),  sum_r z <= 1 (aircraft), z >= 0.
    Slacks occupy indices 0..m-1 so that column indices are stable as columns are appended;
    the basis is kept between calls (adding columns keeps the old basis primal feasible).
    Used ONLY for testing without Gurobi. Also provides a tiny exact MIP by DFS over columns."""
    def __init__(self, inst: Instance):
        self.inst = inst; self.cols: List[Column] = []; self.keys = set()
        self.m = inst.V + inst.R; self.basis = list(range(self.m))
        self.Acols = []   # dense column vectors (length m) of path columns
        self.P = []
    def add_column(self, col: Column) -> bool:
        if col.key() in self.keys: return False
        self.keys.add(col.key()); self.cols.append(col)
        v = np.zeros(self.m)
        for i in col.route: v[i] = 1.0
        v[self.inst.V + col.r] = 1.0
        self.Acols.append(v); self.P.append(col.score); return True
    def solve_lp(self):
        m = self.m; n = len(self.cols)
        T = np.hstack([np.eye(m), np.array(self.Acols).T]) if n else np.eye(m)
        c = np.concatenate([np.zeros(m), np.array(self.P)]); b = np.ones(m)
        basis = self.basis; inb = np.zeros(m+n, bool); inb[basis] = True
        degen = 0
        for _ in range(50000):
            B = T[:, basis]
            Binv = np.linalg.inv(B)
            y = c[basis] @ Binv
            red = c - y @ T; red[inb] = -1.0
            cand = np.nonzero(red > 1e-9)[0]
            if len(cand) == 0: break
            enter = int(cand[0]) if degen > 50 else int(cand[np.argmax(red[cand])])   # Dantzig, Bland if stalling
            d = Binv @ T[:, enter]; xB = Binv @ b
            pos = np.nonzero(d > 1e-12)[0]
            if len(pos) == 0: raise RuntimeError('unbounded RMP (impossible)')
            ratios = xB[pos] / d[pos]; k = int(np.argmin(ratios)); leave = int(pos[k])
            degen = degen + 1 if ratios[k] < 1e-12 else 0
            inb[basis[leave]] = False; basis[leave] = enter; inb[enter] = True
        B = T[:, basis]; xB = np.linalg.solve(B, b); y = np.linalg.solve(B.T, c[basis])
        x = np.zeros(m+n); x[basis] = xB; self.basis = basis
        pi = np.maximum(y[:self.inst.V], 0.0); lam = np.maximum(y[self.inst.V:], 0.0)
        return float(c @ x), x[m:], pi, lam
    def solve_mip(self, time_limit=60.0):
        """Exact over the current pool by DFS with bound (pools are small in tests)."""
        cols = self.cols; V, R = self.inst.V, self.inst.R
        byr = [[j for j, c in enumerate(cols) if c.r == r] for r in range(R)]
        for r in range(R): byr[r].sort(key=lambda j: -cols[j].score)
        best = [0.0, [None]*R]; used = np.zeros(V, bool); t0 = time.time()
        def rec(r, val, chosen):
            if time.time() - t0 > time_limit: return
            if val + sum((cols[byr[q][0]].score if byr[q] else 0) for q in range(r, R)) <= best[0] + 1e-9: return
            if r == R:
                if val > best[0] + 1e-9: best[0] = val; best[1] = chosen[:]
                return
            for j in byr[r] + [None]:
                if j is None: rec(r+1, val, chosen + [None]); continue
                rt = cols[j].route
                if any(used[i] for i in rt): continue
                for i in rt: used[i] = True
                rec(r+1, val + cols[j].score, chosen + [j])
                for i in rt: used[i] = False
        rec(0, 0.0, [])
        self.mip_complete = (time.time() - t0 <= time_limit)
        return best[0], [cols[j] for j in best[1] if j is not None]

class GurobiRMP:
    """Restricted master problem on gurobipy: columns are added incrementally; LP duals; final MIP."""
    def __init__(self, inst: Instance, env=None, threads=0):
        import gurobipy as gp
        from gurobipy import GRB
        self.gp, self.GRB = gp, GRB
        self.inst = inst; self.cols: List[Column] = []; self.keys = set()
        self.m = gp.Model('RMP', env=env) if env is not None else gp.Model('RMP')
        self.m.Params.OutputFlag = 0
        if threads: self.m.Params.Threads = threads
        self.m.ModelSense = GRB.MAXIMIZE
        self.row_area = [self.m.addConstr(gp.LinExpr() <= 1, name=f'a{i}') for i in range(inst.V)]
        self.row_air = [self.m.addConstr(gp.LinExpr() <= 1, name=f'r{r}') for r in range(inst.R)]
        self.vars = []
        self.m.update()
    def add_column(self, col: Column) -> bool:
        if col.key() in self.keys: return False
        self.keys.add(col.key()); self.cols.append(col)
        gp = self.gp
        constrs = [self.row_area[i] for i in col.route] + [self.row_air[col.r]]
        coeffs = [1.0]*len(constrs)
        z = self.m.addVar(lb=0, ub=1, obj=float(col.score), vtype='C', name=f'z{len(self.cols)-1}', column=gp.Column(coeffs, constrs))
        self.vars.append(z); return True
    def solve_lp(self):
        for z in self.vars: z.VType = 'C'
        self.m.update(); self.m.optimize()
        x = np.array([z.X for z in self.vars])
        pi = np.array([max(0.0, c.Pi) for c in self.row_area]); lam = np.array([max(0.0, c.Pi) for c in self.row_air])
        return float(self.m.ObjVal), x, pi, lam
    def solve_mip(self, time_limit=600.0):
        for z in self.vars: z.VType = 'B'
        self.m.Params.TimeLimit = time_limit; self.m.update(); self.m.optimize()
        chosen = [c for c, z in zip(self.cols, self.vars) if z.X > 0.5]
        return float(self.m.ObjVal), chosen

# =====================================================================================
#  Pricing by exact enumeration (tiny instances only; used as test oracle)
# =====================================================================================
def enumerate_routes(inst: Instance, r: int, max_len: int = 6):
    """All feasible (start,end,route) for aircraft r by DFS; tiny instances only."""
    a = inst.air[r]; starts = [a.start] if a.start >= 0 else list(range(inst.D))
    out = []
    def dfs(route, start):
        for e in range(inst.D):
            ok, dur, _, _ = schedule(inst, r, route, start, e)
            if ok: out.append((start, e, list(route), dur))
        if len(route) >= max_len: return
        for i in range(inst.V):
            if i in route: continue
            # prune: partial route must be feasible to at least one end base
            if any(schedule(inst, r, route+[i], start, e)[0] for e in range(inst.D)):
                dfs(route+[i], start)
    for s in starts: dfs([], s)
    return out

def brute_force_optimum(inst: Instance, max_len: int = 6):
    """Exact optimum of tiny instances: one feasible route per aircraft, areas disjoint."""
    routes = [enumerate_routes(inst, r, max_len) for r in range(inst.R)]
    best = [0.0, None]
    used = np.zeros(inst.V, bool)
    def rec(r, val, chosen):
        if r == inst.R:
            if val > best[0] + 1e-9: best[0] = val; best[1] = chosen[:]
            return
        for (s, e, rt, dur) in routes[r]:
            if any(used[i] for i in rt): continue
            for i in rt: used[i] = True
            rec(r+1, val + route_score(inst, rt), chosen + [(s, e, rt)])
            for i in rt: used[i] = False
    rec(0, 0.0, [])
    return best[0], best[1]

# =====================================================================================
#  Tabu Search pricing heuristic
# =====================================================================================
class TabuPricer:
    """Maximises  f(route) = sum_{j in route} (score_j - pi_j) - lambda_r  over feasible routes of aircraft r.
    Neighbourhoods: insertion (vectorised O(1) feasibility via prefix/suffix slacks), removal, swap (replace),
    2-opt segment reversal and or-opt relocation (re-evaluated with schedule()). Tabu attributes on areas."""
    def __init__(self, inst: Instance, r: int, max_iter=None, tenure=None, stall=None, seed=0, time_limit=INF):
        self.inst, self.r = inst, r
        self.a = inst.air[r]
        self.tt, self.tau = inst.tt[r], inst.tau[r]
        n = inst.V
        self.max_iter = max_iter if max_iter is not None else 60 + 8*int(math.sqrt(n))
        self.base_iter = self.max_iter
        self.tenure = tenure if tenure is not None else max(3, int(round(math.sqrt(n)/2)))
        self.stall = stall if stall is not None else max(15, self.max_iter//5)
        self.rng = random.Random(seed); self.time_limit = time_limit
        self.starts = [self.a.start] if self.a.start >= 0 else list(range(inst.D))

    # ---- scheduling helpers --------------------------------------------------------
    def _eval(self, route, s, e):
        return schedule(self.inst, self.r, route, s, e)

    def _best_end(self, route, s):
        """Best end base = the one giving minimal duration. Since bases have no window, the optimal end base is the
        one nearest (in time) to the last area; the prefix schedule is computed once."""
        inst = self.inst
        if not route:
            return True, 0.0, s
        last = route[-1]
        e = int(np.argmin(self.tt[last, inst.V:inst.V + inst.D]))
        ok, dur, _, _ = self._eval(route, s, e)
        return (ok, dur, e) if ok else (False, INF, e)

    def _insertions(self, route, s, e, cand, gain):
        """Vectorised evaluation of inserting each candidate area (array cand) at every position.
        Returns list of (delta_f, area, pos) feasible, where pos = index in route where area is inserted."""
        inst, tt, tau = self.inst, self.tt, self.tau
        V = inst.V; T = self.a.T
        nodes = [V + s] + list(route) + [V + e]; L = len(nodes)
        # prefix quantities along the current route (no-wait cumulative arrival offsets)
        cum = np.zeros(L); g = np.full(L, INF); oc = np.full(L, -INF)   # oc_k = o_k - cum_k (areas), -inf at bases
        for k in range(1, L):
            cum[k] = cum[k-1] + tau[nodes[k-1]] + tt[nodes[k-1], nodes[k]]
            if nodes[k] < V: g[k] = inst.c[nodes[k]] - cum[k]; oc[k] = inst.o[nodes[k]] - cum[k]
        oc[0] = 0.0  # base: o=0, cum=0  (encodes d >= 0)
        # prefix max of oc (H_p = max_{m<=p}), prefix min of g, suffix min of g, suffix max of oc
        Hp = np.maximum.accumulate(oc); Gp = np.minimum.accumulate(g)
        Gs = np.minimum.accumulate(g[::-1])[::-1]; Os = np.maximum.accumulate(oc[::-1])[::-1]
        out = []
        o_c, c_c, tau_c = inst.o[cand], inst.c[cand], tau[cand]
        t_near = tt[np.ix_(cand, range(V, V + inst.D))].min(axis=1)   # time to the nearest base from each candidate
        for p in range(L-1):                       # insert between nodes[p] and nodes[p+1]
            i, j = nodes[p], nodes[p+1]
            cum_a = cum[p] + tau[i] + tt[i, cand]              # arrival offset at candidate
            # shift of the suffix; when inserting in last position the end base is re-chosen (nearest base, as _best_end does)
            t_out = t_near if p == L-2 else tt[cand, j]
            delta = tt[i, cand] + tau_c + t_out - tt[i, j]
            g_a = c_c - cum_a; oc_a = o_c - cum_a
            Hpp = Hp[p]
            feas = (g_a >= Hpp - 1e-9)                          # window of a reachable given earlier waits
            Gsuf = Gs[p+1] - delta                              # suffix slack after shift
            feas &= (Gsuf >= np.maximum(Hpp, oc_a) - 1e-9)
            dmax = np.minimum(np.minimum(Gp[p], g_a), Gsuf)
            feas &= (dmax >= -1e-9)
            Hall = np.maximum(np.maximum(Hpp, oc_a), Os[p+1] - delta)
            cum_end = cum[-1] + delta
            dur = cum_end + np.maximum(0.0, Hall - np.minimum(np.maximum(dmax, 0.0), 1e12))
            feas &= (dur <= T + 1e-6)
            for q in np.nonzero(feas)[0]:
                out.append((float(gain[q]), int(cand[q]), p))
        return out

    # ---- main search --------------------------------------------------------------
    def search(self, pi: np.ndarray, lam_r: float, init_route: Optional[List[int]] = None, init_start: Optional[int] = None,
               pool_size: int = 5, min_rc: float = 1e-4, restarts: int = 3):
        """Runs `restarts` tabu searches (warm start, empty start, perturbed best) and merges the column pools.
        Returns (best_f, best_route, (start,end), pool) where pool = list of (f, route, start, end) with f > min_rc."""
        self._gain = self.inst.score - pi; self._lam = lam_r; self._pool = {}; self._min_rc = min_rc
        self._t0 = time.time()
        best = (-INF, [], (self.starts[0], 0))
        inits = [(init_route, init_start), ([], None)]
        for k in range(restarts):
            if k < len(inits): ir, iss = inits[k]
            else:
                ir = list(best[1]); iss = best[2][0]
                for _ in range(min(2, len(ir))):
                    if ir: ir.pop(self.rng.randrange(len(ir)))
            f, rt, se = self._run(ir, iss)
            if f > best[0] + 1e-9: best = (f, rt, se)
            if time.time() - self._t0 > self.time_limit: break
        cols = sorted(self._pool.values(), key=lambda x: -x[0])[:pool_size]
        return best[0], best[1], best[2], cols

    def _consider(self, rt, ss, ee, val):
        if val > self._min_rc:
            k = (ss, tuple(rt), ee)
            if k not in self._pool: self._pool[k] = (val, list(rt), ss, ee)

    def _run(self, init_route, init_start):
        """One tabu search trajectory. Move selection per iteration:
           1. best admissible insertion if it improves f (areas with positive gain only);
           2. otherwise the intra-route move (2-opt / or-opt) that most reduces the route duration (makes room);
           3. otherwise the least-damaging removal (or a change of departure base), accepting a worsening step.
           Tabu: an inserted area cannot be removed, and a removed area cannot be re-inserted, for `tenure` iterations
           (aspiration if the move yields a new best). Diversification after `stall` non-improving iterations."""
        inst = self.inst; V = inst.V; gain_all = self._gain; lam_r = self._lam
        s = init_start if init_start is not None else self.rng.choice(self.starts)
        route = list(init_route) if init_route else []
        ok, dur, e = self._best_end(route, s)
        if not ok: route = []; ok, dur, e = self._best_end(route, s)
        f = lambda rt: float(gain_all[rt].sum()) - lam_r if rt else -lam_r
        cur_f = f(route); best_f, best_route, best_se = cur_f, list(route), (s, e)
        self._consider(route, s, e, cur_f)
        tabu_in = {}; tabu_out = {}
        it = 0; last_imp = 0
        attractive = np.nonzero(gain_all > 1e-9)[0]
        while it < self.max_iter and time.time() - self._t0 < self.time_limit:
            it += 1
            in_route = np.zeros(V, bool); in_route[route] = True
            cand = attractive[~in_route[attractive]]
            if len(cand):
                keep = [tabu_out.get(int(a), -1) < it or (cur_f + gain_all[a] > best_f + 1e-9) for a in cand]
                cand = cand[keep]
            applied = False
            # 1) insertions
            if len(cand):
                ins = self._insertions(route, s, e, cand, gain_all[cand])
                if ins:
                    self.rng.shuffle(ins); gn, a, p = max(ins, key=lambda m: m[0])
                    if gn > 1e-9:
                        route = route[:p] + [a] + route[p:]; ok, dur, e = self._best_end(route, s)
                        tabu_in[a] = it + self.tenure; applied = True
            # 2) intra-route moves that reduce duration (make room for further insertions)
            if not applied and len(route) >= 2:
                L = len(route); best_mv = (dur - 1e-6, None)
                for k1 in range(L-1):
                    for k2 in range(k1+1, L):
                        rt = route[:k1] + route[k1:k2+1][::-1] + route[k2+1:]
                        okk, d2, ee = self._best_end(rt, s)
                        if okk and d2 < best_mv[0]: best_mv = (d2, (rt, ee))
                for k in range(L):
                    a = route[k]; rest = route[:k] + route[k+1:]
                    for p in range(len(rest)+1):
                        if p == k: continue
                        rt = rest[:p] + [a] + rest[p:]
                        okk, d2, ee = self._best_end(rt, s)
                        if okk and d2 < best_mv[0]: best_mv = (d2, (rt, ee))
                if best_mv[1] is not None:
                    route, e = best_mv[1]; dur = best_mv[0]; applied = True
            # 3) worsening step: removal (least loss) or base change
            if not applied:
                moves = []
                for k in range(len(route)):
                    a = route[k]
                    if tabu_in.get(a, -1) >= it and not (cur_f - gain_all[a] > best_f + 1e-9): continue
                    moves.append((cur_f - gain_all[a], 'rem', k))
                if len(self.starts) > 1:
                    for ss in self.starts:
                        if ss != s:
                            okk, d2, ee = self._best_end(route, ss)
                            if okk: moves.append((cur_f - 1e-6, 'base', (ss, ee)))
                if not moves: break
                self.rng.shuffle(moves); mv = max(moves, key=lambda m: m[0])
                if mv[1] == 'rem':
                    k = mv[2]; a = route[k]; route = route[:k] + route[k+1:]; tabu_out[a] = it + self.tenure
                    ok, dur, e = self._best_end(route, s)
                else:
                    s, e = mv[2]; ok, dur, _ = self._best_end(route, s)
            cur_f = f(route)
            self._consider(route, s, e, cur_f)
            if cur_f > best_f + 1e-9:
                best_f, best_route, best_se = cur_f, list(route), (s, e); last_imp = it
            elif it - last_imp > self.stall:
                route = list(best_route); s, e = best_se
                for _ in range(min(2, len(route))):
                    if route: route.pop(self.rng.randrange(len(route)))
                ok, dur, e = self._best_end(route, s); cur_f = f(route); last_imp = it
                tabu_in.clear(); tabu_out.clear()
        return best_f, best_route, best_se

# =====================================================================================
#  Exact pricing with Gurobi (per aircraft)
# =====================================================================================
class GurobiPricer:
    """Elementary shortest path with resource (time windows + endurance), MTZ-style, one aircraft.
    Objective: max sum (score_j - pi_j) x_ij - lambda_r.  Solution pool harvested for extra columns."""
    def __init__(self, inst: Instance, r: int, env=None, threads=0, time_limit=60.0, pool=10):
        import gurobipy as gp
        from gurobipy import GRB
        self.gp, self.GRB, self.inst, self.r = gp, GRB, inst, r
        a = inst.air[r]; V, D = inst.V, inst.D
        tt, tau = inst.tt[r], inst.tau[r]
        m = gp.Model(f'SP{r}', env=env) if env is not None else gp.Model(f'SP{r}')
        m.Params.OutputFlag = 0; m.Params.TimeLimit = time_limit
        m.Params.PoolSolutions = pool; m.Params.PoolSearchMode = 0
        if threads: m.Params.Threads = threads
        self.time_limit = time_limit
        starts = [a.start] if a.start >= 0 else list(range(D))
        self.starts = starts
        S = [V + s for s in starts]; E = [V + e for e in range(D)]
        A = list(range(V))
        # arcs: S->A, A->A, A->E
        arcs = [(i, j) for i in S for j in A] + [(i, j) for i in A for j in A if i != j] + [(i, j) for i in A for j in E]
        self.x = m.addVars(arcs, vtype=GRB.BINARY, name='x')
        # time variables: s[i] = start of service at area i; s[V+d] = DEPARTURE time from base d; e[d] = ARRIVAL time at
        # base d (separate variable, so that a route may return to its departure base). A departure later than the last
        # window closing is useless, hence the upper bound c_max, which also keeps every big-M small.
        c_max = float(inst.c.max())
        self.s = m.addVars(range(V + D), lb=0.0, ub=[GRB.INFINITY] * V + [c_max] * D, name='s')
        self.e = m.addVars(range(D), lb=0.0, name='e')
        # one departure, one arrival, flow conservation
        m.addConstr(gp.quicksum(self.x[i, j] for i in S for j in A) == 1)
        m.addConstr(gp.quicksum(self.x[i, j] for i in A for j in E) == 1)
        for h in A:
            m.addConstr(gp.quicksum(self.x[i, h] for i in S + A if i != h) == gp.quicksum(self.x[h, j] for j in A + E if j != h))
            m.addConstr(gp.quicksum(self.x[i, h] for i in S + A if i != h) <= 1)          # elementary: visit h at most once
        # endurance as a knapsack on travel + service time (valid: waiting >= 0); tightens the LP relaxation (paper eq. 7)
        m.addConstr(gp.quicksum(float(tt[i, j] + tau[i]) * self.x[i, j] for (i, j) in arcs) <= float(a.T))
        # time propagation with tight big-M; service time at bases is 0
        o = np.concatenate([inst.o, np.zeros(D)]); c = np.concatenate([inst.c, np.full(D, c_max)])
        for (i, j) in arcs:
            M = float(max(0.0, c[i] + tau[i] + tt[i, j] - o[j]) + 1.0)
            tj = self.e[j - V] if j >= V else self.s[j]
            m.addConstr(self.s[i] + float(tau[i] + tt[i, j]) - tj <= M * (1 - self.x[i, j]))
        for i in A:
            m.addConstr(self.s[i] >= float(inst.o[i])); m.addConstr(self.s[i] <= float(inst.c[i]))
        for e in E:
            for st in S:
                m.addConstr(self.e[e - V] <= float(a.T) + self.s[st])
        m.ModelSense = GRB.MAXIMIZE
        self.m = m; self.A, self.S, self.E = A, S, E

    def solve(self, pi: np.ndarray, lam_r: float, min_rc: float = 1e-4, pool_size: int = 10):
        gp, GRB, inst = self.gp, self.GRB, self.inst
        gain = inst.score - pi
        obj = gp.quicksum(float(gain[j]) * self.x[i, j] for (i, j) in self.x.keys() if j < inst.V) - float(lam_r)
        self.m.setObjective(obj, GRB.MAXIMIZE)
        self.m.optimize()
        st = self.m.Status
        if st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            return True, 0.0, []                     # no feasible route for this aircraft: nothing to price
        try: bound = float(self.m.ObjBound)
        except Exception: bound = INF
        if bound != bound or bound > 1e100: bound = INF     # NaN / +inf guard (no bound available yet)
        proved = (st == GRB.OPTIMAL)
        cols = []
        if self.m.SolCount:
            for k in range(min(self.m.SolCount, pool_size)):
                self.m.Params.SolutionNumber = k
                val = self.m.PoolObjVal
                if val <= min_rc: continue
                xv = {key: v.Xn for key, v in self.x.items() if v.Xn > 0.5}
                route, s, e = self._extract(xv)
                if route is not None: cols.append((val, route, s, e))
        return proved, float(bound), cols

    def _extract(self, xv):
        inst = self.inst; V = inst.V
        succ = {i: j for (i, j) in xv}
        for st in self.S:
            if st in succ:
                route = []; cur = succ[st]
                while cur < V:
                    route.append(cur); cur = succ.get(cur, None)
                    if cur is None: return None, None, None
                return route, st - V, cur - V
        return None, None, None

# =====================================================================================
#  Column generation driver
# =====================================================================================
@dataclass
class CGResult:
    method: str; lp_bound: float; lagr_bound: float; integer_value: float; gap: float
    time_total: float; time_lp: float; iterations: int; n_columns: int; proved: bool
    routes: list; log: list
    hit_time_limit: bool = False     # True when the CG loop was cut by the time limit (not by convergence)

def initial_columns(inst: Instance, rmp):
    """One column per (aircraft, area) if feasible, choosing the best end base (Java: single-area routes)."""
    n = 0
    for r in range(inst.R):
        a = inst.air[r]; starts = [a.start] if a.start >= 0 else list(range(inst.D))
        for s in starts:
            for i in range(inst.V):
                best = None
                for e in range(inst.D):
                    ok, dur, _, _ = schedule(inst, r, [i], s, e)
                    if ok and (best is None or dur < best[0]): best = (dur, e)
                if best is not None:
                    n += rmp.add_column(Column(r, [i], s, best[1], float(inst.score[i]), best[0]))
    return n

def column_generation(inst: Instance, rmp, pricers_ts: Optional[list], pricers_exact: Optional[list],
                      time_limit=3600.0, max_iter=10000, min_rc=1e-4, verbose=True, final_mip_time=600.0,
                      ts_init_from_best=True, post_improve=True, sp_time_final=600.0):
    """Generic CG loop.  Each iteration: solve RMP LP, then for each aircraft price with TS (if given);
    aircraft for which TS finds no improving column are priced exactly (if exact pricers given).
    Terminates when no improving column exists (proved if all exact pricers ended OPTIMAL) or on time limit.
    Returns CGResult with LP bound, Lagrangian bound (valid even on early stop) and the integer value of the RMP MIP."""
    t0 = time.time(); log = []; t_lp = 0.0
    n0 = initial_columns(inst, rmp)
    if verbose: print(f'  initial columns: {n0}')
    it = 0; lp_val = 0.0; lagr = INF; proved = False; best_routes = {}; second_chance = False; n_second = 0
    converged = False; cut_by_time = False
    pi = np.zeros(inst.V); lam = np.zeros(inst.R)
    while it < max_iter and time.time() - t0 < time_limit:
        it += 1
        t1 = time.time(); lp_val, x, pi, lam = rmp.solve_lp(); t_lp += time.time() - t1
        # remember the best current route per aircraft to warm-start TS
        if ts_init_from_best:
            for j, c in enumerate(rmp.cols):
                if x[j] > 0.5: best_routes[c.r] = (c.route, c.start)
        new = 0; rc_report = []
        # ---- pass 1: Tabu Search pricing for every aircraft; add every new column it finds ----
        if pricers_ts is not None:
            for r in range(inst.R):
                ir, iss = best_routes.get(r, (None, None))
                bf, brt, bse, cols = pricers_ts[r].search(pi, lam[r], init_route=ir, init_start=iss, min_rc=min_rc,
                                                          restarts=(6 if second_chance else 3), pool_size=8)
                rc_report.append(bf)
                for (val, route, s, e) in cols:
                    ok, dur, _, _ = schedule(inst, r, route, s, e)
                    assert ok, 'TS returned an infeasible route'
                    new += rmp.add_column(Column(r, list(route), s, e, route_score(inst, route), dur))
        # ---- pass 2: exact pricing whenever TS added no NEW column (certification or rescue). Its best bound gives a
        #      valid Lagrangian upper bound even when it finds new columns. ----
        all_proved = False; slack = INF
        if new == 0 and pricers_exact is not None:
            all_proved = True; slack = 0.0; rc_report = []
            remaining = time_limit - (time.time() - t0)
            per_sp = max(1.0, min(sp_time_final, remaining / max(1, inst.R)))
            for r in range(inst.R):
                remaining = time_limit - (time.time() - t0)
                if remaining <= 0:
                    all_proved = False; slack = INF; cut_by_time = True; break
                pricers_exact[r].m.Params.TimeLimit = max(1.0, min(per_sp, remaining))
                pr, bound, cols = pricers_exact[r].solve(pi, lam[r], min_rc=min_rc)
                rc_report.append(bound)
                all_proved = all_proved and pr
                slack = slack + max(0.0, bound) if (slack < INF and bound < INF) else INF
                for (val, route, s, e) in cols:
                    ok, dur, _, _ = schedule(inst, r, route, s, e)
                    assert ok, 'exact pricer returned an infeasible route'
                    new += rmp.add_column(Column(r, list(route), s, e, route_score(inst, route), dur))
        lagr_it = lp_val + slack if (pricers_exact is not None and slack < INF) else INF
        if lagr_it < lagr: lagr = lagr_it
        log.append(dict(it=it, t=round(time.time()-t0, 1), lp=round(lp_val, 3), lagr=(None if lagr == INF else round(lagr, 3)),
                        ncols=len(rmp.cols), new=new, max_rc=round(max(rc_report) if rc_report else 0, 4)))
        if verbose and (it % 5 == 1 or new == 0):
            print(f"  it {it:4d}  t={time.time()-t0:7.1f}s  LP={lp_val:9.3f}  lagr={'-' if lagr==INF else f'{lagr:9.3f}'}  cols={len(rmp.cols):5d}  new={new}")
        if new == 0:
            if pricers_exact is None and n_second < 2:
                # heuristic pricing found nothing: intensify (more restarts/iterations) before giving up
                second_chance = True; n_second += 1
                for p in pricers_ts: p.max_iter = p.base_iter * 3
                continue
            proved = all_proved if pricers_exact is not None else False
            converged = not cut_by_time
            break
        if second_chance:
            second_chance = False
            for p in pricers_ts: p.max_iter = p.base_iter
    if pricers_ts is not None:
        for p in pricers_ts: p.max_iter = p.base_iter
    if pricers_exact is not None and proved:
        lagr = min(lagr, lp_val)      # when pricing is proved, the RMP LP value is the LP bound itself
    hit_time_limit = not converged
    t1 = time.time(); int_val, chosen = rmp.solve_mip(time_limit=final_mip_time); t_mip = time.time() - t1
    # post-optimisation: sequential re-pricing of each route given the others (TS with used areas forbidden)
    if pricers_ts is not None and post_improve:
        chosen, int_val = improve_integer(inst, chosen, pricers_ts)
    ub = lagr if lagr < INF else float('nan')
    gap = (ub - int_val) / ub * 100 if ub == ub and ub > 0 else float('nan')
    method = 'CG' if pricers_exact is not None else 'CG-TS'
    routes = [dict(r=c.r, start=c.start, end=c.end, route=[int(i) for i in c.route], score=float(c.score), dur=round(float(c.dur), 2)) for c in chosen]
    return CGResult(method, lp_val, ub, int_val, gap, time.time()-t0, t_lp, it, len(rmp.cols), proved, routes, log, hit_time_limit)

def improve_integer(inst: Instance, chosen: List[Column], pricers_ts: list, rounds: int = 3):
    """Given one route per aircraft (list of Column), re-optimise each route with TS while the other routes are fixed
    (areas used by others get a prohibitive dual). Accepts only improvements; returns (columns, total score)."""
    cur = {c.r: c for c in chosen}
    for _ in range(rounds):
        improved = False
        for r in range(inst.R):
            used = set()
            for q, c in cur.items():
                if q != r: used.update(c.route)
            pi = np.zeros(inst.V); pi[list(used)] = 1e9
            c0 = cur.get(r)
            f, rt, (s, e), _ = pricers_ts[r].search(pi, 0.0, init_route=(c0.route if c0 else None), init_start=(c0.start if c0 else None),
                                                   restarts=4, pool_size=1, min_rc=INF)
            if rt and (c0 is None or f > c0.score + 1e-9):
                ok, dur, _, _ = schedule(inst, r, rt, s, e); assert ok
                cur[r] = Column(r, list(rt), s, e, route_score(inst, rt), dur); improved = True
        if not improved: break
    cols = [cur[r] for r in sorted(cur)]
    return cols, float(sum(c.score for c in cols))

# =====================================================================================
#  Compact MILP with Gurobi
# =====================================================================================
def solve_milp(inst: Instance, time_limit=3600.0, env=None, threads=0, verbose=False, mip_gap=1e-4):
    """Compact MILP (same as the paper's model, with tight big-M per arc). Returns dict with value, bound, gap, time, routes."""
    import gurobipy as gp
    from gurobipy import GRB
    V, D, R = inst.V, inst.D, inst.R
    m = gp.Model('MBH-TOPTW', env=env) if env is not None else gp.Model('MBH-TOPTW')
    m.Params.OutputFlag = 1 if verbose else 0; m.Params.TimeLimit = time_limit; m.Params.MIPGap = mip_gap
    if threads: m.Params.Threads = threads
    A = list(range(V)); E = [V + e for e in range(D)]
    c_max = float(inst.c.max())        # no useful departure after the last window closes; also keeps the big-M small
    o = np.concatenate([inst.o, np.zeros(D)]); c = np.concatenate([inst.c, np.full(D, c_max)])
    x = {}; s = {}; inc = {h: [] for h in A}
    t_build = time.time()
    for r in range(R):
        a = inst.air[r]; S = [V + a.start] if a.start >= 0 else [V + d for d in range(D)]
        tt, tau = inst.tt[r], inst.tau[r]
        arcs = [(i, j) for i in S for j in A] + [(i, j) for i in A for j in A if i != j] + [(i, j) for i in A for j in E]
        xr = m.addVars(arcs, vtype=GRB.BINARY, name=f'x{r}')
        # sr[i]: service start at area i; sr[V+d]: departure time from base d; er[d]: arrival time at base d (separate
        # variable, so that an aircraft may return to its departure base)
        sr = m.addVars(range(V + D), lb=0.0, ub=[GRB.INFINITY] * V + [c_max] * D, name=f's{r}')
        er = m.addVars(range(D), lb=0.0, name=f'e{r}')
        x[r], s[r] = xr, sr
        m.addConstr(gp.quicksum(xr[i, j] for i in S for j in A) <= 1)
        m.addConstr(gp.quicksum(xr[i, j] for i in A for j in E) <= 1)
        for h in A:
            ins = [xr[i, h] for i in S + A if i != h]; inc[h].extend(ins)
            m.addConstr(gp.quicksum(ins) == gp.quicksum(xr[h, j] for j in A + E if j != h))
        m.addConstr(gp.quicksum(float(tt[i, j] + tau[i]) * xr[i, j] for (i, j) in arcs) <= float(a.T))   # paper eq. (7)
        for (i, j) in arcs:
            M = float(max(0.0, c[i] + tau[i] + tt[i, j] - o[j]) + 1.0)
            tj = er[j - V] if j >= V else sr[j]
            m.addConstr(sr[i] + float(tau[i] + tt[i, j]) - tj <= M * (1 - xr[i, j]))
        for i in A:
            m.addConstr(sr[i] >= float(inst.o[i])); m.addConstr(sr[i] <= float(inst.c[i]))
        for e in E:
            for st in S:
                m.addConstr(er[e - V] <= float(a.T) + sr[st])
    for h in A:
        m.addConstr(gp.quicksum(inc[h]) <= 1)
    m.setObjective(gp.quicksum(float(inst.score[j]) * x[r][i, j] for r in range(R) for (i, j) in x[r].keys() if j < V), GRB.MAXIMIZE)
    t_build = time.time() - t_build
    m.optimize()
    val = m.ObjVal if m.SolCount else 0.0; bound = m.ObjBound
    gap = (bound - val) / bound * 100 if bound > 0 else 0.0
    routes = []
    if m.SolCount:
        for r in range(R):
            succ = {i: j for (i, j), v in x[r].items() if v.X > 0.5}
            S = [V + inst.air[r].start] if inst.air[r].start >= 0 else [V + d for d in range(D)]
            for st in S:
                if st in succ:
                    rt = []; cur = succ[st]
                    while cur < V: rt.append(cur); cur = succ[cur]
                    ok, dur, _, _ = schedule(inst, r, rt, st - V, cur - V)
                    routes.append(dict(r=int(r), start=int(st-V), end=int(cur-V), route=[int(i) for i in rt], score=float(route_score(inst, rt)),
                                       dur=round(float(dur), 2), feasible=bool(ok)))
    status = {GRB.OPTIMAL: 'optimal', GRB.TIME_LIMIT: 'time_limit'}.get(m.Status, str(m.Status))
    out = dict(method='MILP', value=float(val), bound=float(bound), gap=float(gap), time=float(m.Runtime), time_build=t_build,
               status=status, nodes=int(m.NodeCount), vars=m.NumVars, constrs=m.NumConstrs, routes=routes)
    m.dispose()
    return out

# =====================================================================================
#  Convenience runners
# =====================================================================================
def run_cg(inst: Instance, mode='CG', time_limit=3600.0, env=None, threads=0, sp_time=60.0, verbose=True,
           ts_iter=None, seed=0, final_mip_time=600.0, sp_time_final=600.0):
    """mode='CG' (TS + exact certification) or 'CG-TS' (TS only). Without gurobipy falls back to the test simplex
    backend and TS-only pricing (for local testing)."""
    try:
        import gurobipy  # noqa
        have_grb = True
    except Exception:
        have_grb = False
    ts = [TabuPricer(inst, r, max_iter=ts_iter, seed=seed + r) for r in range(inst.R)]
    if have_grb:
        rmp = GurobiRMP(inst, env=env, threads=threads)
        ex = [GurobiPricer(inst, r, env=env, threads=threads, time_limit=sp_time) for r in range(inst.R)] if mode == 'CG' else None
    else:
        rmp = SimplexRMP(inst); ex = None
        if mode == 'CG': print('  [warning] gurobipy not available: CG runs as CG-TS on the test backend')
    res = column_generation(inst, rmp, ts, ex, time_limit=time_limit, verbose=verbose, final_mip_time=final_mip_time,
                            sp_time_final=max(sp_time, sp_time_final))
    if have_grb:
        for p in (ex or []): p.m.dispose()
        rmp.m.dispose()
    return res

def verify_solution(inst: Instance, routes: list) -> Tuple[bool, float]:
    """Independent check: feasibility of every route and disjointness of areas. Returns (ok, total_score)."""
    used = set(); total = 0.0; ok = True
    for rt in routes:
        f, dur, _, _ = schedule(inst, rt['r'], rt['route'], rt['start'], rt['end'])
        ok &= f
        for i in rt['route']:
            if i in used: ok = False
            used.add(i)
        total += route_score(inst, rt['route'])
    return ok, total
