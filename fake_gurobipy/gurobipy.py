"""
Minimal stand-in for gurobipy used ONLY to test the *structure* of the MILP and pricing models without a solver:
it records variables, constraints and the objective as linear expressions, and can evaluate a given assignment.
optimize() does not solve anything (Status = TIME_LIMIT, SolCount = 0).
"""
import itertools

class GRB:
    BINARY = 'B'; CONTINUOUS = 'C'; INTEGER = 'I'
    MAXIMIZE = -1; MINIMIZE = 1
    OPTIMAL = 2; INFEASIBLE = 3; INF_OR_UNBD = 4; TIME_LIMIT = 9
    INFINITY = 1e100

MODELS = []

class LinExpr:
    def __init__(self, coeffs=None, const=0.0):
        self.c = dict(coeffs or {}); self.k = float(const)
    @staticmethod
    def of(x):
        if isinstance(x, LinExpr): return x
        if isinstance(x, Var): return LinExpr({x: 1.0})
        return LinExpr({}, float(x))
    def _comb(self, other, sgn):
        o = LinExpr.of(other); out = LinExpr(self.c, self.k + sgn * o.k)
        for v, a in o.c.items(): out.c[v] = out.c.get(v, 0.0) + sgn * a
        return out
    def __add__(self, o): return self._comb(o, 1.0)
    def __radd__(self, o): return self._comb(o, 1.0)
    def __sub__(self, o): return self._comb(o, -1.0)
    def __rsub__(self, o): return LinExpr.of(o)._comb(self, -1.0)
    def __neg__(self): return LinExpr({v: -a for v, a in self.c.items()}, -self.k)
    def __mul__(self, s):
        s = float(s); return LinExpr({v: a * s for v, a in self.c.items()}, self.k * s)
    __rmul__ = __mul__
    def __le__(self, o): return TempConstr(self - o, '<=')
    def __ge__(self, o): return TempConstr(self - o, '>=')
    def __eq__(self, o): return TempConstr(self - o, '==')
    __hash__ = object.__hash__
    def value(self, val):
        return sum(a * val(v) for v, a in self.c.items()) + self.k

class Var:
    _ids = itertools.count()
    def __init__(self, m, lb, ub, obj, vtype, name):
        self.m, self.lb, self.ub, self.obj, self.VType, self.VarName = m, lb, ub, obj, vtype, name
        self.id = next(Var._ids); self.X = 0.0; self.Xn = 0.0
    def __add__(self, o): return LinExpr.of(self) + o
    def __radd__(self, o): return LinExpr.of(self) + o
    def __sub__(self, o): return LinExpr.of(self) - o
    def __rsub__(self, o): return LinExpr.of(o) - LinExpr.of(self)
    def __mul__(self, s): return LinExpr.of(self) * s
    __rmul__ = __mul__
    def __neg__(self): return -LinExpr.of(self)
    def __le__(self, o): return LinExpr.of(self) <= o
    def __ge__(self, o): return LinExpr.of(self) >= o
    def __eq__(self, o): return LinExpr.of(self) == o
    def __hash__(self): return self.id
    def __repr__(self): return self.VarName

class TempConstr:
    def __init__(self, expr, sense): self.expr, self.sense = expr, sense

class Constr:
    def __init__(self, expr, sense, name): self.expr, self.sense, self.name = expr, sense, name; self.Pi = 0.0
    def violation(self, val, tol=1e-6):
        v = self.expr.value(val)
        if self.sense == '<=': return max(0.0, v - tol)
        if self.sense == '>=': return max(0.0, -v - tol)
        return max(0.0, abs(v) - tol)

class Column:
    def __init__(self, coeffs, constrs): self.coeffs, self.constrs = coeffs, constrs

class _Params:
    def __setattr__(self, k, v): object.__setattr__(self, k, v)

class tupledict(dict):
    def sum(self): return quicksum(self.values())

class Model:
    def __init__(self, name='', env=None):
        self.name = name; self.Params = _Params(); self.vars = []; self.constrs = []; self.vardicts = {}
        self.ModelSense = GRB.MINIMIZE; self.obj = LinExpr(); self.Status = GRB.TIME_LIMIT; self.SolCount = 0
        self.ObjBound = float('inf'); self.ObjVal = 0.0; self.NodeCount = 0; self.Runtime = 0.0; MODELS.append(self)
    def addVar(self, lb=0.0, ub=float('inf'), obj=0.0, vtype='C', name='', column=None):
        v = Var(self, lb, ub, obj, vtype, name); self.vars.append(v)
        if column is not None:
            for a, c in zip(column.coeffs, column.constrs): c.expr.c[v] = c.expr.c.get(v, 0.0) + a
        return v
    def addVars(self, keys, lb=0.0, ub=float('inf'), obj=0.0, vtype='C', name=''):
        keys = list(range(keys)) if isinstance(keys, int) else list(keys)
        lbs = list(lb) if isinstance(lb, (list, tuple)) else [lb] * len(keys)
        ubs = list(ub) if isinstance(ub, (list, tuple)) else [ub] * len(keys)
        d = tupledict((k, self.addVar(lbs[n], ubs[n], obj, vtype, f'{name}[{k}]')) for n, k in enumerate(keys))
        self.vardicts[name] = d; return d
    def addConstr(self, tc, name=''):
        c = Constr(tc.expr, tc.sense, name); self.constrs.append(c); return c
    def setObjective(self, expr, sense=None):
        self.obj = LinExpr.of(expr)
        if sense is not None: self.ModelSense = sense
    def update(self): pass
    def optimize(self): self.Status = GRB.TIME_LIMIT; self.SolCount = 0
    def dispose(self): pass
    @property
    def NumVars(self): return len(self.vars)
    @property
    def NumConstrs(self): return len(self.constrs)
    # ---- evaluation helpers ----
    def check(self, assign, tol=1e-6):
        """assign: dict Var -> value (missing vars = lb). Returns list of (constraint, violation)."""
        val = lambda v: assign.get(v, v.lb if v.lb > -1e300 else 0.0)
        bad = []
        for v in self.vars:
            x = val(v)
            if x < v.lb - tol or x > v.ub + tol: bad.append((f'bound {v}', abs(x)))
        for c in self.constrs:
            viol = c.violation(val, tol)
            if viol > 0: bad.append((c, viol))
        return bad, self.obj.value(val)

def quicksum(it):
    out = LinExpr()
    for t in it: out = out + t
    return out

class Env:
    def __init__(self, *a, **k): pass
    def setParam(self, *a, **k): pass
    def dispose(self): pass

class gurobi:
    @staticmethod
    def version(): return (0, 0, 0)
