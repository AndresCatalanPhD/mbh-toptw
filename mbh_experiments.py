"""
mbh_experiments.py -- campaign driver, checkpointing, statistics and LaTeX export for the MBH-TOPTW study.

A run = (campaign, instance, method). Every finished run is appended immediately to <results_dir>/runs.csv
(one row) and its routes to <results_dir>/routes/<campaign>__<instance>__<method>.json, so a Colab session can be
interrupted and resumed: already-finished runs are skipped.
"""
from __future__ import annotations
import os, json, time, math, traceback, shutil
import numpy as np, pandas as pd
from mbh_core import read_instance, solve_milp, run_cg, verify_solution, INF

COLUMNS = ['campaign', 'instance', 'method', 'R', 'V', 'D', 'value', 'bound', 'gap', 'time', 'status', 'proved', 'iterations',
           'n_columns', 'lp_bound', 'time_limit', 'feasible', 'nodes', 'n_vars', 'n_constrs', 'host', 'timestamp', 'model']

# Version of the optimisation models/heuristics. A row written by an earlier version is ignored (and the run is redone).
#   1: initial release (MILP and exact pricing shared the base time variable: routes returning to their departure base
#      were infeasible for Gurobi -> MILP values and CG bounds were those of a restricted problem)
#   2: separate arrival-time variables, bounded departure time (small big-M), TS end-base re-selection on insertion
MODEL_VERSION = 2

class Campaign:
    def __init__(self, results_dir: str, env=None, threads: int = 0, host: str = ''):
        self.dir = results_dir; os.makedirs(os.path.join(results_dir, 'routes'), exist_ok=True)
        self.csv = os.path.join(results_dir, 'runs.csv'); self.env = env; self.threads = threads; self.host = host
        if os.path.exists(self.csv):
            self.df = pd.read_csv(self.csv)
            missing = [c for c in COLUMNS if c not in self.df.columns]
            if missing:      # file written by an earlier version: keep a copy, add the new columns, rewrite the header
                bak = self.csv + '.v1.bak'
                if not os.path.exists(bak): shutil.copy(self.csv, bak)
                for c in missing: self.df[c] = np.nan
                self.df = self.df.reindex(columns=COLUMNS); self.df.to_csv(self.csv, index=False)
        else:
            self.df = pd.DataFrame(columns=COLUMNS); self.df.to_csv(self.csv, index=False)
    def _valid(self, d):
        """rows usable by the current code: no error, written by the current model version, and (CG) a terminal status"""
        ok = ~d.status.astype(str).str.startswith('error')
        ok &= (pd.to_numeric(d.model, errors='coerce') == MODEL_VERSION)
        ok &= (d.method != 'CG') | self._cg_ok(d)
        return ok
    def done(self, campaign, instance, method):
        """finished without error by the current version AND its routes file exists (a run interrupted while saving is redone)"""
        d = self.df
        hit = bool((self._valid(d) & (d.campaign == campaign) & (d.instance == instance) & (d.method == method)).any())
        return hit and os.path.exists(self._routes_path(campaign, instance, method))
    def clean(self):
        """valid rows; if a run was repeated, the last one counts"""
        d = self.df[self._valid(self.df)].copy()
        return d.drop_duplicates(['campaign', 'instance', 'method'], keep='last').copy()
    def stale(self):
        """rows of earlier versions or with errors (kept in runs.csv for traceability, never used in tables)"""
        return self.df[~self._valid(self.df)].copy()
    @staticmethod
    def _cg_ok(d):
        """CG rows that count as finished: converged (pricing proved), stopped (time limit), unproved (valid Lagrangian
        bound, pricing not proved), no_bound (no valid bound could be computed)."""
        return d.status.astype(str).isin(['converged', 'stopped', 'unproved', 'no_bound'])
    @staticmethod
    def _json_default(o):
        import numpy as _np
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_np.bool_,)): return bool(o)
        if isinstance(o, (_np.ndarray,)): return o.tolist()
        return str(o)
    def _routes_path(self, campaign, instance, method):
        return os.path.join(self.dir, 'routes', f"{campaign}__{instance}__{method}.json")
    def _append(self, row: dict, routes):
        row = {k: row.get(k, None) for k in COLUMNS}
        with open(self._routes_path(row['campaign'], row['instance'], row['method']), 'w') as f:   # routes first: a failure leaves no CSV row
            json.dump(routes, f, default=self._json_default)
        pd.DataFrame([row]).to_csv(self.csv, mode='a', header=False, index=False)
        self.df = pd.read_csv(self.csv)
    def run(self, campaign: str, path: str, method: str, time_limit: float, sp_time: float = 60.0, final_mip_time: float = 600.0,
            verbose: bool = False, force: bool = False):
        name = os.path.basename(path)
        if not force and self.done(campaign, name, method):
            return None
        inst = read_instance(path, name)
        t0 = time.time(); row = dict(campaign=campaign, instance=name, method=method, R=inst.R, V=inst.V, D=inst.D, time_limit=time_limit,
                                     host=self.host, timestamp=time.strftime('%Y-%m-%d %H:%M:%S'), model=MODEL_VERSION)
        try:
            if method == 'MILP':
                res = solve_milp(inst, time_limit=time_limit, env=self.env, threads=self.threads, verbose=verbose)
                ok, val = verify_solution(inst, res['routes'])
                row.update(value=res['value'], bound=res['bound'], gap=res['gap'], time=res['time'], status=res['status'], proved=(res['status'] == 'optimal'),
                           feasible=ok and abs(val - res['value']) < 1e-6, nodes=res['nodes'], n_vars=res['vars'], n_constrs=res['constrs'])
                routes = res['routes']
            else:
                mode = 'CG' if method == 'CG' else 'CG-TS'
                res = run_cg(inst, mode=mode, time_limit=time_limit, env=self.env, threads=self.threads, sp_time=sp_time, verbose=verbose,
                             final_mip_time=final_mip_time)
                ok, val = verify_solution(inst, res.routes)
                has_bound = (res.lagr_bound == res.lagr_bound) and res.lagr_bound < INF
                if mode == 'CG':   # converged: pricing proved; stopped: time limit; unproved: valid Lagrangian bound, pricing not proved
                    status = 'converged' if res.proved else ('stopped' if res.hit_time_limit else ('unproved' if has_bound else 'no_bound'))
                else:              # stopped: time limit; heuristic_stop: TS found no further column after two intensifications
                    status = 'stopped' if res.hit_time_limit else 'heuristic_stop'
                if has_bound and res.integer_value > res.lagr_bound + 1e-6:     # must never happen: a feasible value above a valid bound
                    status = f'error:value {res.integer_value} > bound {res.lagr_bound:.3f}'
                row.update(value=res.integer_value, bound=(res.lagr_bound if has_bound else None),
                           gap=(res.gap if res.gap == res.gap else None), time=res.time_total,
                           status=status,
                           proved=res.proved, iterations=res.iterations, n_columns=res.n_columns, lp_bound=res.lp_bound,
                           feasible=ok and abs(val - res.integer_value) < 1e-6)
                routes = res.routes
        except Exception as e:
            row.update(status='error:' + str(e)[:80]); routes = {'error': traceback.format_exc()}
        self._append(row, routes)
        return row

# ------------------------------------------------------------------------------------------
#  statistics (scipy if available; exact/normal fallbacks otherwise)
# ------------------------------------------------------------------------------------------
def wilcoxon_paired(a, b, alternative='greater'):
    d = np.asarray(a, float) - np.asarray(b, float); d = d[np.abs(d) > 1e-12]; n = len(d)
    if n == 0: return 0.0, 1.0, 0
    try:
        from scipy import stats
        r = stats.wilcoxon(d, alternative=alternative); return float(r.statistic), float(r.pvalue), n
    except Exception:
        pass
    ranks = pd.Series(np.abs(d)).rank().values; Wp = ranks[d > 0].sum(); Wm = ranks[d < 0].sum()
    mu = n*(n+1)/4; sd = math.sqrt(n*(n+1)*(2*n+1)/24); z = (min(Wp, Wm) - mu) / sd
    p2 = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    if alternative == 'two-sided': return float(Wp), float(min(1, p2)), n
    return float(Wp), float(p2/2 if (Wp > Wm) == (alternative == 'greater') else 1-p2/2), n

def friedman_nemenyi(M: np.ndarray, names, higher_is_better=True):
    n, k = M.shape
    R = np.apply_along_axis(lambda r: pd.Series(-r if higher_is_better else r).rank().values, 1, M).mean(0)
    chi2 = 12*n/(k*(k+1))*(np.sum(R**2) - k*(k+1)**2/4)
    try:
        from scipy import stats; p = float(stats.chi2.sf(chi2, k-1))
    except Exception:
        p = float(math.exp(-chi2/2)) if k == 3 else float('nan')
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}.get(k, 2.728)
    CD = q*math.sqrt(k*(k+1)/(6*n))
    return dict(chi2=float(chi2), p=p, avg_rank={nm: round(float(r), 3) for nm, r in zip(names, R)}, CD=float(CD), n=int(n))

# ------------------------------------------------------------------------------------------
#  tables
# ------------------------------------------------------------------------------------------
def to_latex(df: pd.DataFrame, path: str, caption: str, label: str, fmt=None, colfmt=None, note=None, wide=True):
    fmt = fmt or {}
    cols = list(df.columns); env = 'table*' if wide else 'table'
    width = r'\textwidth' if wide else r'\columnwidth'
    colfmt = colfmt or ('l' + 'r'*(len(cols)-1))
    L = [f'\\begin{{{env}}}[!ht]', r'\centering', f'\\caption{{{caption}\\label{{{label}}}}}', r'\small',
         f'\\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep\\fill}}{colfmt}@{{\\extracolsep\\fill}}}}', r'\toprule',
         ' & '.join(r'\textbf{' + str(c).replace('_', r'\_').replace('%', r'\%') + '}' for c in cols) + r' \\', r'\midrule']
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and v != v): cells.append('--')
            elif c in fmt: cells.append(fmt[c].format(v))
            elif isinstance(v, float): cells.append(f'{v:.2f}')
            else: cells.append(str(v).replace('_', r'\_').replace('%', r'\%'))
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}']
    if note: L += [r'\begin{tablenotes}', r'\item {\it Note:} ' + note, r'\end{tablenotes}']
    L += [f'\\end{{{env}}}']
    with open(path, 'w') as f: f.write('\n'.join(L) + '\n')

def benchmark_table(df: pd.DataFrame, campaign='benchmark'):
    """One row per instance with MILP / CG / CG-TS columns, gaps as in the paper."""
    d = df[(df.campaign == campaign) & ~df.status.astype(str).str.startswith('error')]
    rows = []
    for inst, g in d.groupby('instance', sort=True):
        r = dict(instance=inst)
        m = g[g.method == 'MILP'].tail(1); c = g[g.method == 'CG'].tail(1); t = g[g.method == 'CG-TS'].tail(1)
        ub_candidates = []
        if len(m) and pd.notna(m.value.iloc[0]):
            r.update(milp_value=float(m.value.iloc[0]), milp_time=float(m.time.iloc[0]), milp_gap=float(m.gap.iloc[0]), milp_bound=float(m.bound.iloc[0]),
                     milp_status=str(m.status.iloc[0]))
        if len(m) and pd.notna(m.bound.iloc[0]): ub_candidates.append(float(m.bound.iloc[0]))
        if len(c):
            r.update(cg_value=float(c.value.iloc[0]), cg_time=float(c.time.iloc[0]), cg_ub=(float(c.bound.iloc[0]) if pd.notna(c.bound.iloc[0]) else np.nan),
                     cg_lp=float(c.lp_bound.iloc[0]), cg_proved=bool(c.proved.iloc[0]))
            if pd.notna(c.bound.iloc[0]): ub_candidates.append(float(c.bound.iloc[0]))
        if len(t): r.update(ts_value=float(t.value.iloc[0]), ts_time=float(t.time.iloc[0]))
        best_ub = min(ub_candidates) if ub_candidates else np.nan
        vals = [r.get(k) for k in ('milp_value', 'cg_value', 'ts_value') if r.get(k) is not None]
        best_known = max(vals) if vals else np.nan
        r['best_ub'] = best_ub; r['best_known'] = best_known
        for k, lab in (('cg_value', 'cg_gap'), ('ts_value', 'ts_gap')):
            r[lab] = (best_ub - r[k]) / best_ub * 100 if (k in r and best_ub == best_ub and best_ub > 0) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def random_table(df: pd.DataFrame, campaign='random'):
    """Scenario-level means over replications: time (mean, sd) per method and gaps to the best upper bound."""
    d = df[(df.campaign == campaign) & ~df.status.astype(str).str.startswith('error')].copy()
    d['scenario'] = d.instance.str.extract(r'^(rnd\d\d)_')[0]
    # best UB per instance = min over MILP bound and CG lagrangian bound
    ub = d[d.method.isin(['MILP', 'CG'])].groupby('instance').bound.min()
    d['best_ub'] = d.instance.map(ub)
    d['gap_ub'] = (d.best_ub - d.value) / d.best_ub * 100
    rows = []
    for sc, g in d.groupby('scenario', sort=True):
        r = dict(scenario=sc, R=int(g.R.iloc[0]), V=int(g.V.iloc[0]), D=int(g.D.iloc[0]), n=int(g[g.method == 'MILP'].instance.nunique()))
        for mth, pre in (('MILP', 'milp'), ('CG', 'cg'), ('CG-TS', 'ts')):
            h = g[g.method == mth]
            if len(h):
                r[f'{pre}_time'] = h.time.mean(); r[f'{pre}_sd'] = h.time.std(ddof=1) if len(h) > 1 else 0.0
                r[f'{pre}_gap'] = (h.gap.mean() if mth == 'MILP' else h.gap_ub.mean())
                r[f'{pre}_value'] = h.value.mean()
        rows.append(r)
    return pd.DataFrame(rows), d

def paired_stats(d: pd.DataFrame, col='time'):
    """Wilcoxon (one-sided) and Friedman over instances having all three methods."""
    P = d.pivot_table(index='instance', columns='method', values=col, aggfunc='last').dropna()
    out = {'n': int(len(P))}
    if len(P) < 5: return out
    out['wilcoxon_CG_faster_than_MILP'] = wilcoxon_paired(P['MILP'], P['CG'], 'greater')[1]
    out['wilcoxon_TS_faster_than_CG'] = wilcoxon_paired(P['CG'], P['CG-TS'], 'greater')[1]
    out['friedman'] = friedman_nemenyi(P[['MILP', 'CG', 'CG-TS']].values, ['MILP', 'CG', 'CG-TS'], higher_is_better=(col != 'time'))
    return out
