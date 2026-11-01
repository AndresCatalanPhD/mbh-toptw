"""case_sensitivity.py -- sensitivity analysis of the Chilean case study (Section 7, Table 10 of the paper).

Every scenario is solved exactly by complete enumeration (mbh_enumerate.py); no solver is needed. The enumeration is
exact as long as no feasible route reaches the length cap MAX_LEN, which is checked for every scenario.

    python case_sensitivity.py [out_dir]        # writes sensitivity.csv, sensitivity.json and T10_sensitivity.tex
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbh_instances as MI
import mbh_enumerate as E

MAX_LEN = 9
BASE_FLEET = list(MI.REAL_FLEET)
P3C, P3M, C295I, C295P, HELI = range(5)        # positions in REAL_FLEET
FLEET_LABEL = {P3C: 'P-3ACH at Concon', P3M: 'P-3ACH at Mataveri', C295I: 'C-295 at Iquique', C295P: 'C-295 at Punta Arenas', HELI: 'AS365 Dauphin'}


def without(k):
    return [f for j, f in enumerate(BASE_FLEET) if j != k]


# (group, label, keyword arguments of build_real_instance, crew-duty limit in minutes)
SCENARIOS = [
    ('Base', 'Base case (21 June, crew duty 8 h)', {}, 480.0),
    ('Planning date', 'Daylight of 21 December', dict(doy=355), 480.0),
    ('Crew-duty limit', 'Crew duty 6 h', {}, 360.0),
    ('Crew-duty limit', 'Crew duty 10 h', {}, 600.0),
    ('Minimum detection probability', 'Detection probability 0.7 / 0.5', dict(pmin_small=0.7, pmin_large=0.5), 480.0),
    ('Minimum detection probability', 'Detection probability 0.9 / 0.7', dict(pmin_small=0.9, pmin_large=0.7), 480.0),
    ('Strategic importance', 'Equal importance for all sectors', dict(importance={s[0]: 5 for s in MI.REAL_SECTORS}), 480.0),
    ('Basing', 'Second P-3ACH at Concon, not Mataveri', dict(fleet=[BASE_FLEET[P3C], BASE_FLEET[P3C]] + BASE_FLEET[2:]), 480.0),
] + [('Aircraft unavailable', 'Without ' + FLEET_LABEL[k], dict(fleet=without(k)), 480.0) for k in (P3C, P3M, C295I, C295P, HELI)]


def evaluate(kw, crew):
    MI.CREW_LIMIT_MIN = crew
    try:
        inst = MI.build_real_instance(**kw)
        O = E.route_options(inst, MAX_LEN)
        longest = max(len(o[5]) for opts in O for o in opts)
        assert longest < MAX_LEN, 'a route reaches the length cap: raise MAX_LEN'
        S, plans = E.summary(inst, MAX_LEN)
    finally:
        MI.CREW_LIMIT_MIN = 480.0
    names = [a['sector'] for a in inst.meta['areas']]
    cls = [a['cls'] for a in inst.meta['aircraft']]
    covered = [set(i for o in pl for i in o[5]) for pl in plans]
    never = set(range(inst.V)) - set().union(*covered)
    heli = [r for r, c in enumerate(cls) if c == 'AS365 Dauphin']
    heli_needed = bool(heli) and all(pl[heli[0]][2] > 0 for pl in plans)
    return dict(value=S['optimum'], available=S['total_score'], share=100 * S['optimum'] / S['total_score'],
                min_aircraft=min(S['plans_by_flying_aircraft']), aircraft_available=inst.R, n_plans=S['n_plans'],
                flight_min=S['flight_min'], flight_max=S['flight_max'],
                covered_min=min(len(c) for c in covered), covered_max=max(len(c) for c in covered),
                never_covered=[names[i] for i in sorted(never)], helicopter_needed=heli_needed, longest_route=longest,
                min_flight_plan=E.describe(inst, S['min_flight_plan']))


def latex(rows):
    disp = {'Concon': "Conc\\'on"}
    L = [r'\begin{table}[!ht]', r'\centering',
         r'\caption{Case study: sensitivity of the optimal plans, computed by complete enumeration. Value: optimal score, also as a share of the score available in the scenario; Aircraft: fewest aircraft that fly in an optimal plan, out of those available; Plans: number of optimal plans; Flight: least total flight time of an optimal plan (min); Sectors: sectors covered by the optimal plans.\label{tab10}}',
         r'\footnotesize\linespread{1}\selectfont\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr@{\extracolsep\fill}}', r'\toprule',
         r'\textbf{Scenario} & \textbf{Value} & \textbf{Share (\%)} & \textbf{Aircraft} & \textbf{Plans} & \textbf{Flight} & \textbf{Sectors} \\', r'\midrule']
    group = None
    for g, lab, r in rows:
        if group is not None and g != group:
            L.append(r'\midrule' if group == 'Base' else r'\addlinespace[2pt]')
        group = g
        name = lab
        for a, b in disp.items():
            name = name.replace(a, b)
        sec = f"{r['covered_min']}" if r['covered_min'] == r['covered_max'] else f"{r['covered_min']}--{r['covered_max']}"
        L.append(f"{name} & {r['value']:.0f} / {r['available']:.0f} & {r['share']:.0f} & {r['min_aircraft']} / {r['aircraft_available']} & "
                 f"{r['n_plans']:,} & {r['flight_min']:,.0f} & {sec} \\\\")
    L += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    rows = []
    t0 = time.time()
    for g, lab, kw, crew in SCENARIOS:
        r = evaluate(kw, crew)
        rows.append((g, lab, r))
        print(f"{g:30s} {lab:40s} {r['value']:4.0f}/{r['available']:3.0f} ({r['share']:3.0f}%) aircraft {r['min_aircraft']}/{r['aircraft_available']} "
              f"plans {r['n_plans']:5d} flight {r['flight_min']:6.0f} sectors {r['covered_min']}-{r['covered_max']} never {r['never_covered']} "
              f"heli_needed {r['helicopter_needed']} longest {r['longest_route']}")
    print(f'{time.time() - t0:.1f} s')
    json.dump([dict(group=g, scenario=lab, **r) for g, lab, r in rows], open(os.path.join(out, 'sensitivity.json'), 'w'), indent=1)
    import csv
    with open(os.path.join(out, 'sensitivity.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'scenario', 'value', 'available', 'share', 'min_aircraft', 'aircraft_available', 'n_plans', 'flight_min', 'flight_max',
                    'covered_min', 'covered_max', 'never_covered', 'helicopter_needed', 'longest_route'])
        for g, lab, r in rows:
            w.writerow([g, lab, r['value'], r['available'], round(r['share'], 2), r['min_aircraft'], r['aircraft_available'], r['n_plans'],
                        round(r['flight_min'], 2), round(r['flight_max'], 2), r['covered_min'], r['covered_max'], ';'.join(r['never_covered']),
                        r['helicopter_needed'], r['longest_route']])
    open(os.path.join(out, 'T10_sensitivity.tex'), 'w').write(latex(rows))
    print('written', out)
