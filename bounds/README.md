# bounds: optimization audit

Code behind Section VII-D (Optimization Audit) and the bounds it evaluates: Proposition 6 (Capacity guarantee),
Proposition 7 (A lower bound valid for every layout) and Proposition 8 (Window bound), and the exhaustive check of the
instance of Proposition 3 (Tightness). All scripts take their inputs
and outputs as arguments; `--help` lists every option and its default.

Inputs are the files of the data and hypergraph steps of this release:

- `<TEST_UNIONS>`: the read-once test group unions, `.npz` with `e_ptr`, `e_nodes` (one set per test group;
  16,640 sets on W3-1M).
- `<TEST_SETS>`: the per-sub-query sets of the test split, `.npz` with `e_ptr`, `e_nodes`, `batch_ptr`.
- `<CALLS>`, `<SPLITS>`: `calls.jsonl.gz` and `splits.npz` of the workload.
- `<TRAIN_UNIONS_0>`, `<TRAIN_UNIONS_1>`, `<TRAIN_UNIONS_2>`: the training group unions of the three draws.

Software: Python 3 with numpy, numba and scipy. `small_instances.py` also needs OR-Tools (CP-SAT) and, for B = 2 only,
networkx. The reported small-instance runs used OR-Tools 9.15 and SciPy 1.18 with its bundled HiGHS; the data steps of
`prepare/` used SciPy 1.17.1 (`prepare/README.md`, Section 4).

## Capacity bound (Proposition 6)

`capacity_bound.py` computes L_cap = sum_E ceil(|E| / B) per set.

```sh
python bounds/capacity_bound.py --sets <TEST_UNIONS> --out <OUT.json> --B 12
```

On the W3-1M test group unions it gives 193.1 reads per group (`B12.L_cap_per_set`). It runs in seconds.

## Lagrangian lower bound (Proposition 7)

`lagrangian_bound.py` maximises the bound of Proposition 7 over the pin multipliers gamma and the class prices mu. The
per-record shift kappa of the multipliers enters the inner maximum through a separate upper bound (see the header of the
script), so every evaluated value is a valid lower bound obtained from Proposition 7 by this further relaxation. It
starts from 20 degree classes (widths 1, 2, 4, ..., the last class takes the rest), runs 400 iterations of a
class-capacity bound to seed gamma, then 20 outer iterations (stage A), splits every class once by the in-loads
averaged over stage-A iterations 10 to 19, which gives 67 classes on W3-1M, and runs 15 more outer iterations
(stage B). Each outer iteration takes 60 steps on mu. All step rules are fixed in the code; the defaults are the
settings of the reported run.

```sh
python bounds/lagrangian_bound.py --sets <TEST_UNIONS> --out <OUT.json> --records 1000000 --save <BEST.npz>
```

- `L_best_per_set` is the bound per group: 543.2 on the W3-1M test groups (543.2558, rounded down).
  `split.classes` is the number of classes after the split (67).
- `--records` is the number of records of the index; it sets the array sizes and so the summation order, and the
  reported run used the 1M index (1,000,000).
- `--save` writes the best point (gamma, kappa, mu, classes) so that the value can be re-evaluated.
- The value depends on floating-point summation order, hence on the host and library versions: a rerun agrees to about
  1e-4 relative and may differ in the last printed digit.
- Resources of the reported run on W3-1M with 24 threads: at most 11.4 hours (253.6 CPU-hours) and 23.1 GiB peak
  memory.

Ratios in Section VII-D (arithmetic, no script): 543.2558 / 193.1439 = 2.81 times L_cap; the read-once group means of
Table VI divided by 543.2558 and rounded up give 2.05 (P05-G, 1,112.66) to 2.20 (ARC-G, 1,194.43) and 2.59 (MARGO,
1,405.05).

## In-sample reference layout (974.4 reads per group)

The upper end of 543.2 <= R* <= 974.4 is a layout refined on the test groups themselves. It is the layout driver
`layouts/refine_layout.py` run with the P05 objective on `<TEST_UNIONS>` instead of training unions, starting from
MARGO's layout with B = 12, visit-order seed 0, up to 20 passes and no frozen records, all other settings at their
defaults (command in `layouts/README.md`, Section VII-D). Its read-once reads per group on `<TEST_UNIONS>` (the group
scope of Table VI) are the reference value:

```sh
python replay/scope_reads.py --unions <TEST_UNIONS> --sets <TEST_SETS> --schedule <TEST_SCHEDULE> --pages 83334 \
  --draws 0 --layout MARGO=LAYOUT_DIR/w3/MARGO_B12.npy --layout reference=LAYOUT_DIR/w3/P05-test-sets.npy
```

The value is the `group` column of the row `reference`. Not included: the statement that longer runs change it by under
0.5%.

## Small instances solved to optimality

`small_instances.py` draws instances from the test groups (a seed group, n of its records, and the projections of
2,000 groups onto them, merged with multiplicities; projections with fewer than two records are dropped), solves the minimum reads exactly with CP-SAT (and by
maximum-weight perfect matching, Corollary 2, at B = 2), and solves the bound of Proposition 7 with one class per record
as one linear program. `summarize_small_instances.py` counts the instances solved to proven optimality (CP-SAT status
OPTIMAL, or the matching at B = 2) and reports the median ratio of the bound to the optimum.

```sh
for nb in 24,2 12,3 24,3 16,4 24,4 24,6 36,6 24,12 36,12 48,12; do
  n=${nb%,*}; B=${nb#*,}
  python bounds/small_instances.py --unions <TEST_UNIONS> --out <DIR>/small_${n}_${B}.json --records $n --B $B
done
python bounds/summarize_small_instances.py <DIR>/small_*.json
```

Defaults: seed 764575224 (the same for every configuration), 10 instances per configuration, 2,000 groups per
instance, a CP-SAT time limit of 600 s with 8 workers. Each configuration takes at most 10 x 600 s of solver time.
On W3-1M, 71 of the 100 instances are proven optimal, with 12 to 48 records and B from 2 to 12; the median ratio is 96%
at B = 3 (0.959 for 12 records, 0.961 for the one proven instance with 24 records) and 0.812 to 0.896 at B = 12.

## Tightness instance of Proposition 3

`tightness_check.py` enumerates every layout of the grid instance of Proposition 3 (rows of weight 1 + eps, pairs of
cells in one column of weight 1, n = B^2 records) and prints, per (B, eps): the number of layouts, the number of
maximizers of the clique objective and whether the only one is the row layout, g(pi_c), g(pi*) for a minimizer pi* of
F_conn, and the ratio g(pi_c) / g(pi*) against [2/B, (1 + eps) 2/B].

```sh
python bounds/tightness_check.py            # B = 3 with eps = 1/9 and 1/99, B = 4 with eps = 1/9
```

It runs in seconds; B = 4 enumerates 2,627,625 layouts. The default cases print, in this order: one maximizer, the row
layout, ratios 20/27 = 0.7407, 200/297 = 0.6734 and 5/9 = 0.5556, each equal to the upper end (1 + eps) 2/B. With
eps = 0 (`--cases 3:0 4:0`), outside the proposition, the clique objective has two maximizers.

## Window bound (Proposition 8)

`window_bound.py` replays the test sub-queries in turn lockstep under the hybrid cache of every budget and share rho
and computes the bound of Proposition 8 for each training draw, maximised over a grid of window sizes. It reports the
bound per test step for every (draw, budget, rho) and, per budget, the minimum over draws and rho.

```sh
# <TEST_SCHEDULE> is the output of replay/make_schedule.py w3 for the test split (with --drop <drop.json>)
# W3-1M
python bounds/window_bound.py --sets <TEST_SETS> --schedule <TEST_SCHEDULE> \
  --train-unions <TRAIN_UNIONS_0> <TRAIN_UNIONS_1> <TRAIN_UNIONS_2> --out <OUT_1M.json> \
  --records 1000000 --budgets 833,4167,20834 --procs 6
# W3-21M
python bounds/window_bound.py --sets <TEST_SETS> --schedule <TEST_SCHEDULE> \
  --train-unions <TRAIN_UNIONS_0> <TRAIN_UNIONS_1> <TRAIN_UNIONS_2> --out <OUT_21M.json> \
  --records 21015324 --budgets 17513,87564,262144,437820 --procs 3
```

Run time: about 2 minutes at 1M with 6 processes and about 28 minutes at 21M with 3 processes.

The ratio in Section VII-D ("11 to 48 times below") is arithmetic: HyperPage's test reads per step in a Table III cell
divided by `min_lb_per_step` of that budget. Passing the reads with `--reads-per-step 833=<R1>,4167=<R5>,20834=<R25>`
prints and stores these ratios.
