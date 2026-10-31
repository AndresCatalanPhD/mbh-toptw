# MBH-TOPTW: Multi-Base Heterogeneous Team Orienteering Problem with Time Windows

Instances, code and raw results of the paper

> Canut-de-Bon, D. A., Catalán, A. H., Yuraszeck, F. J., Smith, A. E. *Multi-Base Heterogeneous Team Orienteering Problem with Time Windows*. Manuscript under review.

The problem routes a heterogeneous team of aircraft, stationed at several bases, over a set of scored surveillance
areas with time windows, so as to maximise the collected score. An aircraft may recover at a base different from the one
it departed from. Three methods are compared: a compact MILP, a column generation method with Tabu Search pricing
and exact certification (CG), and the same column generation with heuristic pricing only (CG-TS).

## Contents

```
instances/benchmark_het/     20 adapted Cordeau instances pr01..pr20 with a heterogeneous fleet (profiles A/B/C)
instances/scalability/       pr02 family, 3 aircraft, 20..90 areas
instances/scalability_hard/  pr06 family, 7 aircraft, 40..280 areas
instances/random/            100 random instances (20 scenarios x 5 replications; seed = 1000*scenario + replication)
instances/real/              one-day surveillance instance over eight Chilean MPAs (file + metadata with every input)
instances_src/               the 20 original (homogeneous) adapted Cordeau files, for reference
code/mbh_core.py             model, route evaluation, MILP, column generation, Tabu Search
code/mbh_instances.py        instance generators (heterogeneous profiles, scalability, random, real instance)
code/mbh_experiments.py      campaign driver with checkpointing, statistics, LaTeX export
code/run_campaign.py         command-line campaign (PC with Gurobi)
code/MBH_Experiments_Colab.ipynb   the notebook used for the experiments (Google Colab + Gurobi WLS licence)
code/MBH_case_study_v2_Colab.ipynb the notebook that rebuilds and solves the case study (version 2)
code/mbh_enumerate.py        complete enumeration of the optimal plans of the case study
code/test_*.py               tests against complete enumeration and a solver-free check of the model structure
results/runs_clean.csv       one row per run (408 runs): value, bound, gap, time, status, iterations, columns
results/routes/              the routes of every run (JSON)
results/real_instance/       the real instance as solved, its metadata and the enumeration of its optimal plans
results/paper_tables/        the LaTeX tables of the paper, generated from runs_clean.csv
results/SUMMARY.txt          summary printed by the notebook
```

## Instance file format

```
R V D                                   number of aircraft, areas, bases
T v w d                                 per aircraft: endurance (min), speed (km/h), detection width (km), departure base (0 = free)
id x y size score open close            per area: coordinates (km), size (hundreds of km2), score, time window (min)
id x y 0 1 0 1000                       per base (numbered after the areas)
```

Travel time of aircraft r between two locations: Euclidean distance / speed. Surveillance time of an area: size * 100 * 60 / (w * v) minutes (exhaustive sweep). In the real instance the sizes stored are effective sizes A(-ln(1-Pmin))/200 and the widths are lateral detection ranges, so that the same rule reproduces the Koopman random-search time.

## Reproducing the experiments

Requirements: Python 3.10+, numpy, pandas, gurobipy 12+ with a licence that is not size-limited (an academic WLS licence is free).

```
cd code
python test_model_structure.py      # solver-free check of the MILP and pricing models
python test_cg_tiny.py              # CG and CG-TS against complete enumeration
python run_campaign.py --inst ../instances --out ../results_new --campaigns benchmark scalability random
```

The notebook `MBH_Experiments_Colab.ipynb` runs the same campaign on Google Colab with checkpointing to Google Drive.

## Changes in version 1.1 (case study)

The case-study instance (`real_chile`) now uses the areas of the creation decrees of the marine parks, rounded to the
nearest thousand km2: Rapa Nui with Motu Motiro Hiva 729,000; Nazca-Desventuradas 300,000; Juan Fernandez 286,000
(marine park 262,000 + multiple-use area 24,000); Diego Ramirez-Paso Drake 144,000. The Paso Drake sector is the southern
half of the austral park and its reference point is the centroid of that half (58.69 S, 70.30 W); in version 1.0 it lay
near Cape Horn, outside the park. The optimum is 196 of 228 (version 1.0: 188 of 218). `mbh_enumerate.py` enumerates its
optimal plans: 8 plans, 4 of them with the helicopter on the ground.
Benchmark, scalability and random campaigns are unchanged.

## Licence

Code: MIT. Instances and results: CC BY 4.0. The adapted Cordeau instances derive from the MDVRPTW instances of Cordeau, Laporte and Mercier (2001).
