"""
mbh_instances.py -- instance generation for the MBH-TOPTW computational study.

1. Heterogeneous adapted-Cordeau benchmark (pr01..pr20):
   the original adaptation uses one aircraft type per instance (T, v, w). We assign aircraft k (0-based) the profile
   PROFILES[k mod 3] = multipliers on (T, v, w):
       A: (1.00, 1.00, 1.0)   baseline
       B: (0.80, 1.25, 0.6)   fast, short-legged, narrow sensor
       C: (1.25, 0.80, 1.6)   slow, long-legged, wide sensor
   Travel times and service times therefore differ by aircraft (service time ~ 1/(w*v): B is 1.33x slower, C 0.78x).
2. Scalability families: pr02 truncated to its first n areas, n in {20,...,90} (3 aircraft), and pr06 truncated to
   n in {40,...,280} (7 aircraft), the family on which the compact MILP stops closing.
3. Random instances (Table 4 bounds), heterogeneous by construction, fixed seeds.
4. Real instance of Section 6.4 (8 Chilean MPAs, 4 bases, P-3 and C-295) from the authors' data sheet: build_real_instance.
"""
from __future__ import annotations
import os, math, json
import numpy as np
from mbh_core import Instance, Aircraft, read_instance, write_instance

PROFILES = [(1.00, 1.00, 1.0), (0.80, 1.25, 0.6), (1.25, 0.80, 1.6)]
PROFILE_NAMES = ['A', 'B', 'C']

def make_heterogeneous(inst: Instance) -> Instance:
    air = []
    for k, a in enumerate(inst.air):
        fT, fv, fw = PROFILES[k % 3]
        air.append(Aircraft(float(round(a.T * fT)), float(round(a.v * fv)), float(round(a.w * fw, 1)), a.start))
    out = Instance(inst.name, inst.V, inst.D, inst.R, air, inst.ax, inst.ay, inst.size, inst.score, inst.o, inst.c, inst.bx, inst.by)
    return out.build()

def truncate_areas(inst: Instance, n: int, name: str) -> Instance:
    out = Instance(name, n, inst.D, inst.R, [Aircraft(a.T, a.v, a.w, a.start) for a in inst.air],
                   inst.ax[:n], inst.ay[:n], inst.size[:n], inst.score[:n], inst.o[:n], inst.c[:n], inst.bx, inst.by)
    return out.build()

# Table-4 bounds
RB = dict(T=(400, 800), v=(170, 500), w=(1.0, 3.0), xy=(-100, 100), size=(1, 25), score=(1, 25), o=(0, 500), width=(90, 180))

def random_instance(R: int, V: int, D: int, seed: int, name: str) -> Instance:
    rng = np.random.default_rng(seed)
    air = [Aircraft(float(rng.integers(RB['T'][0], RB['T'][1] + 1)), float(rng.integers(RB['v'][0], RB['v'][1] + 1)),
                    float(round(rng.uniform(*RB['w']), 1)), int(rng.integers(0, D))) for _ in range(R)]
    ax = rng.uniform(*RB['xy'], V).round(3); ay = rng.uniform(*RB['xy'], V).round(3)
    size = rng.integers(RB['size'][0], RB['size'][1] + 1, V).astype(float)
    score = rng.integers(RB['score'][0], RB['score'][1] + 1, V).astype(float)
    o = rng.integers(RB['o'][0], RB['o'][1] + 1, V).astype(float)
    c = o + rng.integers(RB['width'][0], RB['width'][1] + 1, V)
    bx = rng.uniform(*RB['xy'], D).round(3); by = rng.uniform(*RB['xy'], D).round(3)
    return Instance(name, V, D, R, air, ax, ay, size, score, o, c.astype(float), bx, by).build()

# scenarios of Table 1 / Table 5 (aircraft, areas, bases)
SCENARIOS = [(2, 48, 4), (3, 96, 4), (4, 144, 4), (5, 192, 4), (6, 240, 4), (7, 288, 4), (2, 72, 6), (3, 144, 6), (4, 216, 6), (5, 288, 6),
             (1, 48, 4), (2, 96, 4), (3, 144, 4), (4, 192, 4), (5, 240, 4), (6, 288, 4), (1, 72, 6), (2, 144, 6), (3, 216, 6), (4, 288, 6)]

SCAL_HARD_SIZES = (40, 80, 120, 160, 200, 240, 280)

def generate_all(src_dir: str, out_dir: str, n_rep: int = 5, scal_sizes=(20, 30, 40, 50, 60, 70, 80, 90), scal_hard_sizes=SCAL_HARD_SIZES):
    """Writes benchmark_het/, scalability/, random/ under out_dir. Returns a manifest dict."""
    man = {'benchmark_het': [], 'scalability': [], 'scalability_hard': [], 'random': []}
    d1 = os.path.join(out_dir, 'benchmark_het'); os.makedirs(d1, exist_ok=True)
    for k in range(1, 21):
        nm = f'pr{k:02d}'; inst = make_heterogeneous(read_instance(os.path.join(src_dir, nm), nm))
        write_instance(inst, os.path.join(d1, nm)); man['benchmark_het'].append(nm)
    d2 = os.path.join(out_dir, 'scalability'); os.makedirs(d2, exist_ok=True)
    base = make_heterogeneous(read_instance(os.path.join(src_dir, 'pr02'), 'pr02'))
    for n in scal_sizes:
        nm = f'pr02_n{n:03d}'; write_instance(truncate_areas(base, n, nm), os.path.join(d2, nm)); man['scalability'].append(nm)
    d2h = os.path.join(out_dir, 'scalability_hard'); os.makedirs(d2h, exist_ok=True)
    base6 = make_heterogeneous(read_instance(os.path.join(src_dir, 'pr06'), 'pr06'))
    for n in scal_hard_sizes:
        nm = f'pr06_n{n:03d}'; write_instance(truncate_areas(base6, n, nm), os.path.join(d2h, nm)); man['scalability_hard'].append(nm)
    d3 = os.path.join(out_dir, 'random'); os.makedirs(d3, exist_ok=True)
    for si, (R, V, D) in enumerate(SCENARIOS, start=1):
        for rep in range(1, n_rep + 1):
            nm = f'rnd{si:02d}_{R}a{V}v{D}b_r{rep}'; seed = 1000 * si + rep
            write_instance(random_instance(R, V, D, seed, nm), os.path.join(d3, nm)); man['random'].append(nm)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f: json.dump(man, f, indent=1)
    return man

# -------------------------------------------------------------------------------------------------
#  Real instance (Section 6.4): a single operational day over the Chilean MPAs of the authors' data sheet.
#
#  * Areas: the 8 MPAs of the sheet. The three oceanic parks and the austral park are divided into patrol SECTORS
#    (operational practice for areas of 10^5-10^6 km^2); each sector is a node with its own score and window.
#  * Fleet: three aircraft classes parameterised on public specifications (P-3ACH Orion, C-295 Persuader,
#    AS365 Dauphin class rotary-wing). Maximum route time = min(endurance, 8 h crew-duty limit).
#  * Surveillance time: Koopman random search with lateral detection range rho (paper eq. 22):
#        tau = -A ln(1-Pmin) / (2 rho v).   File stores size_eff = A(-ln(1-Pmin))/200 with w = rho (see README).
#  * Pmin per node: 0.8 for sectors up to PMIN_AREA_KM2, 0.6 for larger ones (a single pass cannot reasonably
#    achieve 0.8 over 10^5 km^2). Score s_i = round(25 * (I_i/5) * Pmin_i), I_i = strategic importance 1..5.
#  * Time windows: start of the surveillance pass between civil sunrise and sunset minus the shortest pass,
#    computed from latitude/longitude for the planning date (default 21 June, shortest day), common clock UTC-4.
#  * Bases: the four of the sheet (Iquique, Concon, Maquehue/Temuco, Punta Arenas) plus Mataveri as a forward
#    operating location for the Easter Island sectors. Each aircraft has a given departure base; arrival base free.
# -------------------------------------------------------------------------------------------------
PMIN_AREA_KM2 = 150000.0
# (sector name, parent MPA, area km2, lat, lon, importance 1..5)
REAL_SECTORS = [
    ('Rapa Nui NE',       'Pascua (Rapa Nui / Motu Motiro Hiva)', 712694/4, -25.6, -107.8, 5),
    ('Rapa Nui NW',       'Pascua (Rapa Nui / Motu Motiro Hiva)', 712694/4, -25.6, -110.9, 5),
    ('Rapa Nui SE',       'Pascua (Rapa Nui / Motu Motiro Hiva)', 712694/4, -28.6, -107.8, 5),
    ('Rapa Nui SW',       'Pascua (Rapa Nui / Motu Motiro Hiva)', 712694/4, -28.6, -110.9, 5),
    ('Desventuradas E',   'Nazca-Desventuradas',                  300000/2, -26.3,  -79.9, 5),
    ('Nazca ridge W',     'Nazca-Desventuradas',                  300000/2, -26.8,  -82.0, 5),
    ('Robinson Crusoe',   'Juan Fernandez',                       450000/2, -33.6,  -78.8, 5),
    ('Alejandro Selkirk', 'Juan Fernandez',                       450000/2, -33.8,  -80.8, 5),
    ('Francisco Coloane', 'Francisco Coloane',                         15.0, -53.65, -72.30, 4),
    ('Diego Ramirez',     'Cabo de Hornos - Diego Ramirez',       101000/2, -56.5,  -68.7, 5),
    ('Paso Drake E',      'Cabo de Hornos - Diego Ramirez',       101000/2, -56.0,  -67.0, 4),
    ('Isla Chanaral',     'Isla Chanaral',                             28.9, -29.03, -71.57, 3),
    ('Isla Choros',       'Isla Choros',                               38.6, -29.26, -71.54, 3),
    ('La Rinconada',      'La Rinconada',                               3.3, -23.47, -70.50, 3),
]
def _dms(d, m, s): return -(d + m / 60 + s / 3600)
REAL_BASES = [('Iquique',            (_dms(20, 32, 7),  _dms(70, 10, 53))),
              ('Concon (Torquemada)', (_dms(32, 56, 59), _dms(71, 28, 43))),
              ('Maquehue (Temuco)',   (_dms(38, 46, 1),  _dms(72, 38, 14))),
              ('Punta Arenas',        (_dms(53, 0, 9),   _dms(70, 51, 16))),
              ('Mataveri (Rapa Nui)', (-27.165, -109.422))]
CREW_LIMIT_MIN = 480.0
# (class, endurance min, speed km/h, lateral range km)  public specifications; the route limit applies min(endurance, crew limit)
REAL_CLASSES = {'P-3ACH Orion':   dict(T=720.0, v=463.0, rho=150.0),
                'C-295 Persuader': dict(T=480.0, v=480.0, rho=100.0),
                'AS365 Dauphin':   dict(T=240.0, v=250.0, rho=40.0)}
# (class, departure base) for the planning day
REAL_FLEET = [('P-3ACH Orion', 'Concon (Torquemada)'), ('P-3ACH Orion', 'Mataveri (Rapa Nui)'),
              ('C-295 Persuader', 'Iquique'), ('C-295 Persuader', 'Punta Arenas'), ('AS365 Dauphin', 'Punta Arenas')]

def sun_times_utc(lat, lon, doy):
    """NOAA simplified sunrise/sunset in minutes UTC for day-of-year doy (zenith 90.833 deg)."""
    g = 2 * math.pi / 365 * (doy - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    la = math.radians(lat); z = math.radians(90.833)
    cosha = math.cos(z) / (math.cos(la) * math.cos(decl)) - math.tan(la) * math.tan(decl)
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cosha))))
    return 720 - 4 * (lon + ha) - eqt, 720 - 4 * (lon - ha) - eqt

def _aeqd(lat, lon, lat0, lon0, R=6371.0088):
    """Azimuthal equidistant projection (km); distance error below 2% for this instance."""
    lat, lon, lat0, lon0 = map(np.radians, (np.asarray(lat, float), np.asarray(lon, float), lat0, lon0))
    cc = np.clip(np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0), -1, 1); c = np.arccos(cc)
    k = np.where(c > 1e-12, c / np.sin(np.maximum(c, 1e-12)), 1.0)
    x = R * k * np.cos(lat) * np.sin(lon - lon0); y = R * k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
    return x, y

def build_real_instance(doy=172, utc_offset_h=-4.0, pmin_small=0.8, pmin_large=0.6, importance=None, fleet=None, name='real_chile'):
    """Builds the Section-6.4 instance. doy: planning day (172 = 21 June). importance: optional dict sector->1..5 overrides.
    fleet: optional list of (class, base) overrides. Returns Instance with .meta describing every input."""
    secs = [list(t) for t in REAL_SECTORS]
    if importance:
        for t in secs:
            if t[0] in importance: t[5] = importance[t[0]]
    fleet = fleet or REAL_FLEET
    V, D = len(secs), len(REAL_BASES)
    names = [t[0] for t in secs]; A = np.array([t[2] for t in secs], float)
    lat = [t[3] for t in secs] + [b[1][0] for b in REAL_BASES]; lon = [t[4] for t in secs] + [b[1][1] for b in REAL_BASES]
    x, y = _aeqd(lat, lon, float(np.mean(lat)), float(np.mean(lon)))
    P = np.where(A <= PMIN_AREA_KM2, pmin_small, pmin_large)
    I = np.array([t[5] for t in secs], float); score = np.round(25.0 * (I / 5.0) * P)
    size_eff = A * (-np.log(1 - P)) / 200.0
    bnames = [b[0] for b in REAL_BASES]
    air = []
    for cls, base in fleet:
        c = REAL_CLASSES[cls]; air.append(Aircraft(min(c['T'], CREW_LIMIT_MIN), c['v'], c['rho'], bnames.index(base)))
    # time windows: start of pass in [sunrise, sunset - shortest pass among classes], local clock utc_offset_h
    o = np.zeros(V); cl = np.zeros(V); tau_min = np.zeros(V)
    for k in range(V):
        sr, ss = sun_times_utc(lat[k], lon[k], doy); sr += utc_offset_h * 60; ss += utc_offset_h * 60
        taus = [A[k] * (-math.log(1 - P[k])) * 60 / (2 * REAL_CLASSES[c]['rho'] * REAL_CLASSES[c]['v']) for c in REAL_CLASSES]
        tau_min[k] = min(taus); o[k] = round(sr); cl[k] = round(ss - min(taus))
    inst = Instance(name, V, D, len(air), air, x[:V].round(2), y[:V].round(2), size_eff.round(4), score, o, cl, x[V:].round(2), y[V:].round(2)).build()
    inst.meta = dict(
        planning_day_of_year=doy, clock=f'UTC{utc_offset_h:+.0f}', crew_limit_min=CREW_LIMIT_MIN, pmin_rule=f'{pmin_small} if area<={PMIN_AREA_KM2:.0f} km2 else {pmin_large}',
        areas=[dict(id=k + 1, sector=names[k], mpa=secs[k][1], lat=float(lat[k]), lon=float(lon[k]), area_km2=round(float(A[k])), importance=int(I[k]), pmin=float(P[k]),
                    score=int(score[k]), tw_open=int(o[k]), tw_close=int(cl[k]),
                    tau_min={cls: round(float(A[k] * (-math.log(1 - P[k])) * 60 / (2 * REAL_CLASSES[cls]['rho'] * REAL_CLASSES[cls]['v'])), 1) for cls in REAL_CLASSES}) for k in range(V)],
        bases=[dict(id=V + k + 1, name=b[0], lat=b[1][0], lon=b[1][1]) for k, b in enumerate(REAL_BASES)],
        aircraft=[dict(idx=r, cls=cls, base=base, endurance_min=REAL_CLASSES[cls]['T'], route_limit_min=min(REAL_CLASSES[cls]['T'], CREW_LIMIT_MIN),
                       speed_kmh=REAL_CLASSES[cls]['v'], rho_km=REAL_CLASSES[cls]['rho']) for r, (cls, base) in enumerate(fleet)],
        projection='azimuthal equidistant, centre = mean of all points', koopman='tau = -A ln(1-Pmin)/(2 rho v)')
    return inst

def real_instance_tables(inst, out_dir):
    """LaTeX tables for Section 6.4: areas (T6a) and fleet/bases (T6b)."""
    import pandas as pd
    A = pd.DataFrame(inst.meta['areas'])
    def hhmm(m): return f'{int(m)//60:02d}:{int(m)%60:02d}'
    rows = [dict(ID=a.id, Sector=a.sector, MPA=a.mpa, Area=f'{a.area_km2:,.0f}', I=a.importance, Pmin=f'{a.pmin:.1f}', Score=a.score,
                 Window=f'{hhmm(a.tw_open)}--{hhmm(a.tw_close)}', **{f'tau {c.split()[0]}': f"{v:.0f}" for c, v in a.tau_min.items()}) for a in A.itertuples()]
    T = pd.DataFrame(rows)
    L = [r'\begin{table*}[!ht]', r'\centering', r'\caption{Real instance: surveillance sectors, scores, daylight windows (local clock) and Koopman surveillance time in minutes for each aircraft class.\label{tab6}}',
         r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular*}{\textwidth}{@{\extracolsep\fill}' + 'rll' + 'r' * (len(T.columns) - 3) + r'@{\extracolsep\fill}}', r'\toprule',
         ' & '.join(r'\textbf{' + c.replace('tau ', r'$\tau$ ').replace('Pmin', r'$P_{\min}$').replace('Area', r'Area (km$^2$)') + '}' for c in T.columns) + r' \\', r'\midrule']
    for _, r in T.iterrows(): L.append(' & '.join(str(v) for v in r.values) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    with open(os.path.join(out_dir, 'T6a_real_areas.tex'), 'w') as f: f.write('\n'.join(L) + '\n')
    F = pd.DataFrame(inst.meta['aircraft'])
    L = [r'\begin{table}[!ht]', r'\centering', r'\caption{Real instance: fleet for the planning day (public specifications; route time limited by the 8-hour crew-duty rule) and departure bases.\label{tab7}}',
         r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}llrrrr@{\extracolsep\fill}}', r'\toprule',
         r'\textbf{Aircraft} & \textbf{Departure base} & \textbf{Endur.} & \textbf{Limit} & \textbf{Speed} & $\rho$ \\', r' & & (min) & (min) & (km/h) & (km) \\', r'\midrule']
    for _, r in F.iterrows(): L.append(f"{r.cls} & {r.base} & {r.endurance_min:.0f} & {r.route_limit_min:.0f} & {r.speed_kmh:.0f} & {r.rho_km:.0f} \\\\")
    L += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    with open(os.path.join(out_dir, 'T6b_real_fleet.tex'), 'w') as f: f.write('\n'.join(L) + '\n')
    return T, F

if __name__ == '__main__':
    import sys
    src = sys.argv[1]; out = sys.argv[2]
    man = generate_all(src, out)
    print({k: len(v) for k, v in man.items()})
