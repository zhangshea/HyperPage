# Layouts

Scripts that build every layout the paper reports, and the exact command for each layout label.

## Scripts

| Script | Builds | Third-party program needed |
|---|---|---|
| `refine_layout.py` | Algorithm 1: ARC-G, CLQ-G, SET-G, P05-G, ARC-Q, SET-Q, ARC-C, P05-C, ARC-CF, HEAT, STARLING | none |
| `freq_layout.py` | FREQ: records packed by training frequency | none |
| `rcm_layout.py` | RCM: reverse Cuthill-McKee order of the symmetrized graph | none (scipy) |
| `proximity_layout.py` | Proximity: recursive balanced 2-means on the vectors | none |
| `shp_layout.py` | SHP-G: SHP's paired moves on the P05 objective | none |
| `mtkahypar_two_level.py` | Mt-KaHyPar in two levels (leaves, then blocks of B records) | `mtkahypar` Python package (not shipped) |
| `mtkahypar_band_refine.py` | Mt-KaHyPar V-cycles from MARGO's layout within bands of 43 pages | `mtkahypar` Python package (not shipped) |
| `margo_uncached.py` | MARGO on uncached records | your own build of MARGO's partitioner `new_mincut` (not shipped) |

Run every command from the release root after `pip install -e .`. Every script writes the layout (`.npy`, int32 page
of every record) and a `.json` with run details next to it. Every script prints its options with `--help`.

MARGO's code carries no licence, so this release contains none of it: `margo_uncached.py` only writes the input files of
MARGO's partitioner, runs the binary you point it to with `--partitioner`, and reads its output file. The reported
layouts were built with 16 OpenMP threads for the partitioner (`OMP_NUM_THREADS=16`).

The Mt-KaHyPar scripts use the `mtkahypar` Python package, version 1.7.post1 from PyPI (`pip install
mtkahypar==1.7.post1`; source at https://github.com/kahypar/mt-kahypar, MIT licence). Mt-KaHyPar runs with several threads are not deterministic, so its layouts cannot be rebuilt bit for bit. The scripts
without a third-party program are deterministic. The layouts of Algorithm 1 and SHP-G were built on one thread
(`NUMBA_NUM_THREADS=1`).

## File names

The replay scripts of `experiments/` read the layouts from `LAYOUT_DIR/<w>`, with one directory per index and page size:
`w3` (W3-1M; W4 replays these), `w21` (W3-21M), `w1`, `w2`, `deep`, `w5m`, `w10m`, and `w3_b6`, `w3_b3`, `w3_b2` (W3-1M
with B = 6, 3, 2). Every command below writes its layout under the name these scripts expect:

| File name | Used for |
|---|---|
| `MARGO_B<B>.npy` | MARGO's layout at B records per page (written by `prepare/margo_layout.py`) |
| `<label>.npy` | a layout that serves all draws: STARLING, RCM, Proximity |
| `<label>_r<r>.npy` | a layout of training draw r = 0, 1, 2: SET-G, ARC-G, CLQ-G, P05-G, FREQ, SHP-G, Mt-KaHyPar, HEAT, SET-shuffled |
| `<label>_h<h>_r<r>.npy` | a cache-aware layout of draw r refined for h hot pages: ARC-C, P05-C, ARC-CF, HEAT-res, Mt-KaHyPar-res, MARGO-uncached, CLQ-C |

HEAT-res is HEAT on residual unions ("HEAT's objective" in Table VII), Mt-KaHyPar-res is Mt-KaHyPar on residual unions,
MARGO-uncached is MARGO on uncached records, CLQ-C is CLQ on residual unions and SET-shuffled is SET refined on the
shuffled control. Layouts that no script of `experiments/` reads carry names of the same pattern.

## Notation in the commands

| Symbol | Meaning |
|---|---|
| `$IDX` | prefix of the index files of one scale: `$IDX_disk.index`, `$IDX_mem.index` (and `$IDX_mem.index.weights` for `margo_uncached.py`); W3-1M, W4 and the page sizes use W2's 1M index |
| `$L` | `LAYOUT_DIR/<w>` of that index and page size |
| `$MARGO` | MARGO's layout of that index at the page size, `$L/MARGO_B<B>.npy`; the start of every layout unless stated |
| `$ID` | identifier-order layout, page(v) = floor(v / B), `$L/ID_B<B>.npy` (written by `prepare/margo_layout.py`) |
| `$U_r`, `$Q_r` | training group unions and training per-query sets of draw r, `HYPER_DIR/<w>/train<r>_U.npz` and `train<r>_Q.npz` |
| `$USHUF_r` | group unions of draw r built from sub-queries shuffled across the groups of a step, `HYPER_DIR/w3/train<r>_shuf_U.npz` |
| `$UTEST` | group unions of the W3 test steps, `HYPER_DIR/w3/test_U.npz` |
| `$VEC8`, `$VEC32` | record vectors: int32 n, int32 d, then n * d int8 values (`$VEC8`, `DATA_DIR/<w>/base.bin`) or float32 values (`$VEC32`, `DATA_DIR/<w>/base_f32.fbin`) |
| `$r` | training draw, 0, 1 and 2 (run each command once per draw); tables report the mean over the three layouts |

The inputs are built by `prepare/` (`prepare/README.md`).

## MARGO's layouts at B = 12, 6, 3, 2

`prepare/margo_layout.py` writes `MARGO_B<B>.npy` and `ID_B<B>.npy` with B = page bytes // 644. The B = 12 layouts are
part of `prepare/README.md`. The incumbents of the page-size runs come from the same script on the 1M index of W2 with
smaller pages:

```sh
for pb in "6 4096" "3 2048" "2 1536"; do set -- $pb
  python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/w2/margo_R64_L125 \
      --data-f32 DATA_DIR/w2/base_f32.fbin --page-bytes $2 --out LAYOUT_DIR/w3_b$1 --threads 12
done
```

## Cached pages per budget

Every cache-aware layout (`--cached-pages h`) is refined for one budget of D pages. Its record cache holds the
B * h most frequent records of the draw's training unions, ties by smaller identifier, where

    h = floor(rho * D),  rho = MARGO's validation share for that budget,

chosen on validation steps before any layout is trained. D is the budget share times the page count in the table,
rounded to the nearest integer; the 2 GiB budget is 2^31 / 8192 = 262,144 pages. The full-corpus index has 1,751,277
pages, and its budgets in pages are round(f * 1,751,278), the values below.

| Index (B) | Records | Pages used for D | Budget: D pages, MARGO's rho, h |
|---|---|---|---|
| W3-1M (12) | 1,000,000 | 83,334 | 1%: 833, 0.5, **416**; 5%: 4,167, 0.9, **3,750**; 25%: 20,834, 0.75, **15,625** |
| W3-21M (12) | 21,015,324 | 1,751,278 | 1%: 17,513, 0.5, **8,756**; 5%: 87,564, 0.25, **21,891**; 2 GiB ("15%"): 262,144, 0.75, **196,608**; 25%: 437,820, 0.9, **394,038** |
| W1 (12) | 1M index | 83,334 | 1%: 833, 0, **0**; 5%: 4,167, 0.25, **1,041**; 25%: 20,834, 0.9, **18,750** |
| W2 (12) | 1M index | 83,334 | 1%: 833, 0.5, **416**; 5%: 4,167, 0.75, **3,125**; 25%: 20,834, 0.75, **15,625** |
| W3-1M (6) | 1,000,000 | 166,667 | 1%: 1,667, 0.5, **833**; 5%: 8,333, 0.75, **6,249**; 25%: 41,667, 0.25, **10,416** |
| W3-1M (3) | 1,000,000 | 333,334 | 1%: 3,333, 0.75, **2,499**; 5%: 16,667, 0.5, **8,333**; 25%: 83,334, 0.5, **41,667** |
| W3-1M (2) | 1,000,000 | 500,000 | 1%: 5,000, 0.75, **3,750**; 5%: 25,000, 0.5, **12,500**; 25%: 125,000, 0.75, **93,750** |
| W3-5M (12) | 5,000,000 | 416,667 | 1%: 4,167, 0.75, **3,125**; 5%: 20,833, 0.75, **15,624** |
| W3-10M (12) | 10,000,000 | 833,334 | 1%: 8,333, 0.75, **6,249**; 5%: 41,667, 0.5, **20,833** |

W4 shares the W3-1M index and uses the W3-1M layouts. DEEP1M is reported cache-blind only.

## Layouts of Algorithm 1 (`refine_layout.py`)

Common part of every command, written `REFINE` below:

    python layouts/refine_layout.py --index-prefix $IDX --capacity B --seed $r

with the defaults `--max-passes 8 --tol 1e-4 --max-cand 16 --max-sets 64 --window 32`, which are the paper's settings
(at most 8 passes, epsilon = 1e-4). B = 12 unless stated.

| Label | Arguments after `REFINE` | Hypergraph |
|---|---|---|
| ARC-G, CLQ-G, SET-G, P05-G | `--objective ARC\|CLQ\|SET\|P05 --sets $U_r --start $MARGO --out $L/<label>_r$r.npy` | group unions |
| ARC-Q, SET-Q | `--objective ARC\|SET --sets $Q_r --start $MARGO --out $L/<label>_r$r.npy` | per-query sets |
| ARC-C, P05-C (h hot pages) | `--objective ARC\|P05 --sets $U_r --start $MARGO --cached-pages h --out $L/<label>_h<h>_r$r.npy` | residual unions of $U_r |
| ARC-CF (h hot pages) | `--objective ARC --sets $U_r --start $MARGO --cached-pages h --freeze-uncovered --out $L/ARC-CF_h<h>_r$r.npy` | residual unions of $U_r |
| HEAT | `--objective HEAT --sets $U_r --query-sets $Q_r --start $MARGO --out $L/HEAT_r$r.npy` | arcs traversed in $Q_r (candidates from $U_r) |
| HEAT-res (h hot pages) | the HEAT arguments plus `--cached-pages h`, `--out $L/HEAT-res_h<h>_r$r.npy` | same, without arcs at cached records |
| CLQ-C (h hot pages) | `--objective CLQ --sets $U_r --start $MARGO --cached-pages h --out $L/CLQ-C_h<h>_r$r.npy` | residual unions of $U_r |
| STARLING | `--objective STARLING --start $ID --out $L/STARLING.npy` with `--seed 0`, one layout | none (index graph only) |

HEAT uses the weighting with one more per arc (the default): the paper takes the weighting with fewer validation reads
(Section VII-A, Compared methods). The other weighting, traversal counts alone, is `--heat-without-arc-term`; comparing
the two on validation reads needs both built for the same draws and budgets on W3-1M and replayed as in `table3.sh`.

### Main workload W3 (Tables III, VI, VII; Figs. 3 and 4)

B = 12, draws r = 0, 1, 2, `$L` = `LAYOUT_DIR/w3` (1M) or `LAYOUT_DIR/w21` (full corpus).

| Paper label | W3-1M | W3-21M |
|---|---|---|
| ARC-G, CLQ-G, SET-G, P05-G | the G row above | same, full-corpus files |
| ARC-Q, SET-Q (Section VII-A, Compared methods) | the Q row above | not built |
| P05-C, ARC-C, ARC-CF | the C and CF rows above with h = 416, 3750, 15625 | h = 8756, 21891, 196608, 394038 |
| HEAT (Table III) | the HEAT row above | same, full-corpus files |
| HEAT-res (Table III; "HEAT's objective" in Table VII) | the HEAT-res row with h = 416, 3750, 15625 | h = 8756, 21891, 196608, 394038 |
| STARLING (Tables III, VI) | the STARLING row above | same, full-corpus files |

The W4 columns of Table III replay these W3-1M layouts; no layout is trained on W4.

### Section VII-D: in-sample reference

The layout that reads 974.4 per test group is P05 refined on the test group unions themselves, with up to 20 passes
(B = 12, seed 0, W3-1M):

    python layouts/refine_layout.py --index-prefix $IDX --capacity 12 --seed 0 --objective P05 --sets $UTEST \
        --start $MARGO --max-passes 20 --out LAYOUT_DIR/w3/P05-test-sets.npy

`--max-passes` is an upper limit; refinement stops earlier once a pass gains at most epsilon times the objective.

### Section VII-C: shuffled control

    REFINE --objective SET --sets $USHUF_r --start $MARGO --out LAYOUT_DIR/w3/SET-shuffled_r$r.npy      (W3-1M)

### Other workloads (Section VII-C)

B = 12, r = 0, 1, 2, `$IDX` and `$MARGO` of the workload's own 1M index (DEEP1M: its index and its MARGO layout).

| Workload (`$L`) | Layouts |
|---|---|
| W1 (`LAYOUT_DIR/w1`) | ARC-G, CLQ-G, SET-G, P05-G (G row); P05-C with h = 1041 (5%) and 18750 (25%), `P05-C_h1041_r$r.npy`, `P05-C_h18750_r$r.npy`; at 1% h = 0, so P05-G serves |
| W2 (`LAYOUT_DIR/w2`) | ARC-G, CLQ-G, SET-G, P05-G (G row); P05-C with h = 416, 3125, 15625 |
| DEEP1M (`LAYOUT_DIR/deep`) | ARC-G and P05-G (G row), cache-blind only |

### Page sizes B = 6, 3, 2 (Section VII-C)

W3-1M index (`$IDX` = W2's 1M index) and training draws, `--capacity B`, `$L` = `LAYOUT_DIR/w3_b<B>`, `$MARGO` =
`$L/MARGO_B<B>.npy` (section "MARGO's layouts" above):

| B | Cache-blind (G row) | Cache-aware (objective that led the cache-blind layouts on validation) |
|---|---|---|
| 6 (4 KB pages) | ARC-G, CLQ-G, SET-G, P05-G | P05-C (C row), h = 833, 6249, 10416 |
| 3 | ARC-G, CLQ-G, SET-G, P05-G | P05-C (C row), h = 2499, 8333, 41667 |
| 2 | ARC-G, CLQ-G, SET-G, P05-G | CLQ-C (CLQ row), h = 3750, 12500, 93750 |

For example, `REFINE` with B = 6 and `--objective P05 --sets $U_r --start LAYOUT_DIR/w3_b6/MARGO_B6.npy --cached-pages 833
--out LAYOUT_DIR/w3_b6/P05-C_h833_r$r.npy`.

### Scale: 5M and 10M samples (Fig. 4)

Index, training unions and MARGO layout of each sample, B = 12, r = 0, 1, 2:

| Sample (`$L`) | Layouts |
|---|---|
| W3-5M (`LAYOUT_DIR/w5m`) | ARC-G, P05-G (G row); ARC-C and P05-C (C row) with h = 3125 (1%) and 15624 (5%) |
| W3-10M (`LAYOUT_DIR/w10m`) | ARC-G, P05-G (G row); ARC-C and P05-C (C row) with h = 6249 (1%) and 20833 (5%) |

Fig. 4 at 1M and 21M uses the W3 layouts above.

### Engine order and Table IV (Section VII-B)

W3-1M, `$L` = `LAYOUT_DIR/w3`, r = 0, 1, 2. `experiments/engine_order.sh` reads ARC-C and P05-C with h = 416 and 3750
(the C rows above) and P05-C refined for MARGO's share under the engine's arrival order:

    REFINE --objective P05 --sets $U_r --start $MARGO --cached-pages 208 --out $L/P05-C_h208_r$r.npy
    REFINE --objective P05 --sets $U_r --start $MARGO --cached-pages 2083 --out $L/P05-C_h2083_r$r.npy

## Compared methods outside Algorithm 1

B = 12. "1M" is W3-1M (`$L` = `LAYOUT_DIR/w3`), "21M" the full-corpus index (`$L` = `LAYOUT_DIR/w21`).

### FREQ (Table III, Table VI)

    python layouts/freq_layout.py --sets $U_r --start $MARGO --capacity 12 --out $L/FREQ_r$r.npy

Same command at 1M and 21M with that scale's files. Ties keep MARGO's page order, then identifier order.

### RCM (Table III)

1M, from the records of a disk index of the 1M graph stored in sectors of 4,096 bytes with R = 64 neighbour slots
(the disk index that `prepare/build_index.py` writes has this format, so the prefix is `$IDX`):

    python layouts/rcm_layout.py --index-prefix $IDX --graph-source disk --sector-bytes 4096 --max-degree 64 \
        --capacity 12 --out LAYOUT_DIR/w3/RCM.npy

One layout serves all draws. 21M: not included. `rcm_layout.py --graph-source mem` (the default) computes RCM from the
graph file `$IDX_mem.index` of an index; the arguments of the full-corpus RCM layout of Table III are not part of this
release.

### Proximity (Table III)

1M:

    python layouts/proximity_layout.py --vectors $VEC8 --capacity 12 --out LAYOUT_DIR/w3/Proximity.npy

One layout serves all draws. 21M: not included; the vector file and arguments of the full-corpus Proximity layout of
Table III are not part of this release.

### SHP-G (Table III, Table VI, Section VII-C)

    python layouts/shp_layout.py --index-prefix $IDX --sets $U_r --start $MARGO --capacity 12 --max-rounds R \
        --out $L/SHP-G_r$r.npy

| Scale | Rounds R | Stop rule of the reported layouts |
|---|---|---|
| 1M | 43 (r = 0), 45 (r = 1), 46 (r = 2) | a process CPU budget equal to P05-G's refinement time of the draw (`--cpu-budget 3277.3`, `3089.0`, `3250.7` s), reached after these rounds |
| 21M | 60 (`--max-rounds 60`, the default) | the 60-round cap, on every draw |

A CPU budget depends on the host, so `--max-rounds` with the round counts above rebuilds the layouts on any host. The
JSON history records, per round, the gain predicted on the round's snapshot (`snapshot_gain`) and the decrease actually
realized (`decrease`); Section VII-C reports 1 - sum `decrease` / sum `snapshot_gain` over the rounds of each 1M draw.

### Mt-KaHyPar (Table III, Table VI)

Seed = draw r, 6 threads, preset QUALITY, imbalance 0.03 (the defaults). Every command reads `$U_r` and `$MARGO`.

| Paper cell | Variant | Command |
|---|---|---|
| Mt-KaHyPar, 1M (Table III; Table VI 1M) | two levels on group unions | `python layouts/mtkahypar_two_level.py --sets $U_r --start $MARGO --seed $r --out $L/Mt-KaHyPar_r$r.npy` |
| on residual unions, 1M 1% and 5% | two levels on residual unions | the same plus `--cached-pages h` with h = 416 or 3750, `--out $L/Mt-KaHyPar-res_h<h>_r$r.npy` |
| on residual unions, 1M 25% | band refinement on residual unions | `python layouts/mtkahypar_band_refine.py --sets $U_r --start $MARGO --seed $r --cached-pages 15625 --out $L/Mt-KaHyPar-res_h15625_r$r.npy` |
| Mt-KaHyPar, 21M (Table III; Table VI 21M) | band refinement on group unions | `python layouts/mtkahypar_band_refine.py --sets $U_r --start $MARGO --seed $r --out $L/Mt-KaHyPar_r$r.npy` |
| on residual unions, 21M | band refinement on residual unions | the same plus `--cached-pages h` with h = 8756, 21891, 196608, 394038, `--out $L/Mt-KaHyPar-res_h<h>_r$r.npy` |

Two-level defaults: leaves of 512 records (`--leaf-records 512`), then blocks of at most B records per leaf. Band
refinement defaults: bands of 43 consecutive MARGO pages (`--band-pages 43`), 3 V-cycles (`--vcycles 3`), every block
weight at most B; a block that Mt-KaHyPar leaves above B is repaired greedily within its band. Band refinement needs a
large stack: run it after `ulimit -s unlimited`. The W4 columns replay the W3-1M layouts.

### MARGO on uncached records (Table VII; Table IV, 21M at 25%)

    python layouts/margo_uncached.py --index-prefix $IDX --vectors <vectors> --vector-type <type> --sets $U_r \
        --margo-layout $MARGO --partitioner <your new_mincut> --cached-pages h --work-dir <dir> \
        --out $L/MARGO-uncached_h<h>_r$r.npy

| Scale | Vectors | h |
|---|---|---|
| 1M | `$VEC8 --vector-type int8` | 416, 3750, 15625 |
| 21M | `$VEC32 --vector-type float32` | 8756, 21891, 196608, 394038 |

`$IDX` must hold MARGO's arc weights (`$IDX_mem.index.weights`). The W4 columns replay the W3-1M layouts.

### HyperPage (Tables III and VII, Fig. 4)

HyperPage is no further layout: for each budget and draw it is the pair of layout and share with the fewest validation
reads among the candidates of Section VII-A, Compared methods (MARGO, STARLING, the layouts of Algorithm 1 for the
budget, HEAT on residual unions and MARGO on uncached records), all built by the commands above. At 5M and 10M (Fig. 4)
the candidates are MARGO, ARC-G, P05-G, ARC-C and P05-C.

### MARGO and Starling

MARGO's layouts (`$MARGO`) come from MARGO's own partitioner and are the incumbent and start layout. STARLING is the
layout of `refine_layout.py --objective STARLING` above.

## Layouts of `engine/` (Tables IV and V, Section VII-B)

All W3, B = 12, `--start $MARGO`, r = 0, 1, 2 unless stated. No script of `experiments/` reads these files; the names
follow the pattern above.

| Where | Layout | Arguments after `REFINE` |
|---|---|---|
| Table IV, 1M 1% | CLQ-G | the G row (`$L/CLQ-G_r$r.npy`) |
| Table IV, 1M 5% (shares chosen in the engine's arrival order) | P05-C | `--objective P05 --sets $U_r --cached-pages 2083 --out $L/P05-C_h2083_r$r.npy` |
| Table IV, 1M 25% | ARC-C | `--objective ARC --sets $U_r --cached-pages 15625 --out $L/ARC-C_h15625_r$r.npy` |
| Table IV, 21M 1% and 5% | ARC-C | `--objective ARC --sets $U_r --cached-pages 8756 \| 21891 --out $L/ARC-C_h<h>_r$r.npy` |
| Table IV, 21M 15% | HEAT on residual unions | `--objective HEAT --sets $U_r --query-sets $Q_r --cached-pages 196608 --out $L/HEAT-res_h196608_r$r.npy` |
| Table V, 1M (page search, cache of the whole budget) | P05 on the records that cache leaves uncached | `--objective P05 --sets $U_r --cached-pages 833 \| 4166 \| 20833 --out $L/P05-PS_h<h>_r$r.npy` |
| Table V, 21M (page search) | ARC on the records that cache leaves uncached | `--objective ARC --sets $U_0 --seed 0 --cached-pages 17513 \| 87564 --out $L/ARC-PS_h<h>_r0.npy` |
| Section VII-B, memory charged as measured | P05 and ARC with an exact record count | `--objective P05\|ARC --sets $U_r --cached-records 2902 \| 25862 \| 90582 --out $L/<P05\|ARC>-CR_c<records>_r$r.npy` |
