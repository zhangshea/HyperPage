# Experiments: tables, figures and numbers of Sections VI and VII

One shell script per table and figure that the replay produces, and the commands for the numbers in the text. Run
everything from the release root after `pip install -e .`. Every script has its input directories as variables at the
top (`DATA_DIR`, `HYPER_DIR`, `LAYOUT_DIR`, `OUT_DIR`, ...); every Python script prints its options with `--help`.
The plotting scripts need matplotlib. `replay/_results.py` and `replay/_scopes.py` are helper modules of the scripts in
`replay/`; the scripts import them by name, which works because Python puts a script's directory on the import path.

| Script | Produces | Run after |
|---|---|---|
| `table2.sh` | Table II | `prepare/` (all workloads) |
| `table3.sh` | Table III, HyperPage's deployments, the bootstrap of every margin, the budget-to-match row | `prepare/`, `layouts/` |
| `table6.sh` | Table VI | `prepare/`, `layouts/` |
| `table7.sh` | Table VII | `table3.sh` |
| `fig3.sh`, `plot_share_curves.py` | Fig. 3 and the same-share table of all seven budgets | `prepare/`, `layouts/` |
| `fig4.sh`, `plot_scale.py` | Fig. 4 | `prepare/` (with the 5M and 10M samples), `layouts/` |
| `other_workloads.sh` | W1, W2, DEEP1M and page sizes B = 6, 3, 2 (Section VII-C) | `prepare/`, `layouts/` |
| `engine_order.sh` | replay in the engine's logged arrival order and the selection of the timed 1M layouts (Section VII-B) | `prepare/`, `layouts/`, `engine/` (order logs) |

Table I has no measured entries. Tables IV and V and the timing numbers belong to `engine/`; the bounds of Section
VII-D belong to `bounds/`.

## Inputs

`prepare/README.md` describes `DATA_DIR/<w>` (`calls.jsonl.gz`, `splits.npz`, W4 also `qstr.json` and
`meta_<split>.npy`) and `HYPER_DIR/<w>` (`<split>_Q.npz` per-query sets, `<split>_U.npz` group unions,
`train{0,1,2}_U.npz`). Workload directories: `w3` (W3-1M), `w21` (W3-21M), `w4`, `w5m`, `w10m`, `w1`, `w2`.

The scripts expect the layouts of `layouts/README.md` under these names in `LAYOUT_DIR/<w>` (`r` is the training
draw 0, 1, 2; `h` the hot-page count of the budget):

| File | Labels |
|---|---|
| `MARGO_B12.npy` | MARGO (written by `prepare/`) |
| `<label>.npy` | STARLING, RCM, Proximity (one layout for all draws) |
| `<label>_r<r>.npy` | SET-G, ARC-G, CLQ-G, P05-G, FREQ, SHP-G, Mt-KaHyPar, HEAT |
| `<label>_h<h>_r<r>.npy` | ARC-C, P05-C, ARC-CF, HEAT-res, Mt-KaHyPar-res, MARGO-uncached |

`other_workloads.sh` reads `LAYOUT_DIR/w1`, `w2` and `deep` with the names above (W1 and W2: `P05-C_h<h>_r<r>.npy`)
and, for B records per page, `LAYOUT_DIR/w3_b<B>` with `MARGO_B<B>.npy`, the four cache-blind labels and
`P05-C_h<h>_r<r>.npy` (B = 6, 3) or `CLQ-C_h<h>_r<r>.npy` (B = 2, CLQ on residual unions). `engine_order.sh` reads
`LAYOUT_DIR/w3` with `ARC-C` and `P05-C` at h = 416 and 3750 and `P05-C` at h = 208 and 2083.

HEAT-res is HEAT on residual unions ("HEAT's objective" in Table VII), Mt-KaHyPar-res is Mt-KaHyPar on residual
unions, MARGO-uncached is MARGO on uncached records. The W4 columns replay the W3-1M layouts and hot-record rankings.

## Settings

| Setting | Value |
|---|---|
| Budgets, W3-1M and W4 (83,334 pages) | 1%, 5%, 25% = 833, 4167, 20834 pages |
| Budgets, full corpus (1,751,277 pages) | 1%, 5%, 15%, 25% = 17513, 87564, 262144, 437820 pages (round(f x 1,751,278); 15% is 2^31 / 8192) |
| Budgets, 5M and 10M | 4167 and 20833; 8333 and 41667 pages (1% and 5%) |
| Hot pages h of the cache-aware layouts | 1M 416, 3750, 15625; full corpus 8756, 21891, 196608, 394038; 5M 3125, 15624; 10M 6249, 20833 |
| Draws | 0, 1, 2 |
| Share grid rho | 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 |
| Validation | 10 warm-up steps, then 20 scored steps (W3: steps 41 to 70; W4: steps 11 to 30 of its 88) |
| HyperPage's candidates (Table III) | MARGO plus ARC-C, ARC-CF, ARC-G, CLQ-G, HEAT-res, MARGO-uncached, P05-C, P05-G, SET-G, STARLING, every column; ties go to MARGO, then to the smaller share, then to the candidate listed first |
| HyperPage's candidates (Fig. 4) | MARGO plus ARC-G, P05-G, ARC-C, P05-C |
| HyperPage's candidates (W1, W2; `other_workloads.sh`) | MARGO plus ARC-G, CLQ-G, P05-C, P05-G, SET-G (W1 at 1%: without P05-C) |
| MARGO's share (W1, W2; `other_workloads.sh`) | the share with the fewest validation reads on each draw (`saving_table.py --incumbent-per-draw`) |
| HyperPage's candidates (B = 6, 3, 2; `other_workloads.sh`) | MARGO plus ARC-G, the cache-aware layout (P05-C; CLQ-C at B = 2), CLQ-G, P05-G, SET-G |
| HyperPage's candidates (engine order; `engine_order.sh`) | MARGO plus ARC-C, ARC-G, CLQ-G, P05-C (h = 416, 3750), P05-C (h = 208, 2083), P05-G, SET-G |
| Budgets and hot pages h, W1 and W2 (83,334 pages) | 833, 4167, 20834 pages; h: W1 0, 1041, 18750; W2 416, 3125, 15625 |
| Budgets and hot pages h, W3-1M at B = 6, 3, 2 | B = 6: 166,667 pages, 1667, 8333, 41667, h 833, 6249, 10416; B = 3: 333,334 pages, 3333, 16667, 83334, h 2499, 8333, 41667; B = 2: 500,000 pages, 5000, 25000, 125000, h 3750, 12500, 93750 |
| DEEP1M (83,334 pages) | 833, 4167, 20834 pages; validation 30 steps, test 130 steps of equal size |
| Bootstrap | 2,000 resamples, the default seed of `replay/significance.py`, moving blocks of 1, 5, 10 and 26 steps |
| Budget grid of the last row of Table III | `replay/budget_match.py --help` (37 points at 1M and W4, 42 on the full corpus) |

Savings are pooled over the draws, 100 (1 - sum_k R_k / sum_k R_MARGO,k), each layout at its own validation share.
The test replays of `table3.sh` and `other_workloads.sh` run every layout at its validation share and at each share at
which a draw deploys it (`test_shares` of `select_deployment.py`, read by `replay_hybrid.py --at`). W3 schedules skip
the groups listed in `drop.json`, as the splits do (`make_schedule.py --drop`). `saving_table.py` also writes the
per-draw savings (`per_draw`) to its `--out` file; where a number below is a mean
of per-draw savings, the row says so.

## Every item

Paths `results/...` are the default `OUT_DIR`s of the scripts. `<col>` is a column of `table3.sh`, for example
`w3-1m_833` (W3-1M, 1%) or `w3-21m_262144` (full corpus, 15%).

### Sections I to VI

| Paper item | Command | Inputs |
|---|---|---|
| Introduction: "83.9% of the node expansions of a rollout group repeat a node that the group has already expanded" | Table II, overlap of W3-1M (`experiments/table2.sh`) | |
| Introduction: "a billion records fill about 683 GB of pages, so a workstation with 64 GB of DRAM caches under 5% of the index after 32 GB of PQ codes" | arithmetic, no script: 10^9 / 12 pages of 8,192 bytes; (64 - 32) / 683 | |
| Introduction: "1% of the records serve 24.6% of the test accesses at 1M and 36.4% at 21M" (mean over the draws) | `python replay/hot_coverage.py --sets HYPER_DIR/w3/test_Q.npz --train-unions 'HYPER_DIR/w3/train{draw}_U.npz' --records 1000000 --capacity 1 --cell 10000:1`, and on `w21` with `--records 21015324 --cell 210153:1`: with `--capacity 1` and share 1 the cell caches exactly PAGES records, round(0.01 n); the number is the printed share of the test accesses | W3-1M and W3-21M test sets and training unions |
| Definition 3: "On our logs nu = 8, a step has 128 prompts" | `process_run_rollouts_per_prompt` and `process_run_prompts_per_step` in `DATA_DIR/w3/checks.json` (`prepare/extract_calls.py w3`) | W3 rollouts |
| Definition 3: "a request holds 1.68 sub-queries on average" | not included: no script prints this ratio. On the W3 test split, `tool_calls` of `results/table6/scope_w3.json` (`replay/scope_reads.py`, 278,146 requests) and the test sub-queries of Table II (467,989) give 467,989 / 278,146 = 1.68 | `table6.sh`, `table2.sh` |
| Definition 7: the 2 GiB limit "is 15%" of the full-corpus index | arithmetic, no script: 2^31 / 8192 = 262,144 of 1,751,277 pages | |
| Section IV: "a 384-dimensional int8 record takes 644 bytes, which gives B = 6 at 4 KB and B = 12 at 8 KB" | `max_node_len` of the index header (`prepare/margo_layout.py --expect-record-bytes 644`; B = page bytes // 644) | index of `prepare/` |
| Section IV: "on 10K test queries each of SIFT1M, DEEP1M and T2I-1M issued reads equal expanded nodes" | not included: this check uses three further public datasets and is not part of this release | |
| Proposition 3: "We checked the instance by exhaustive enumeration for B = 3 and B = 4" | `python bounds/tightness_check.py` (`bounds/README.md`) | none |

### Section VII-A

| Paper item | Command | Inputs |
|---|---|---|
| Table II (units, sub-queries, \|e_q\|, \|E_b\|, overlap, L, recall@10); "L = 500 at 1M and L = 800 on the full corpus" | `experiments/table2.sh` | hypergraphs and logging summaries of `prepare/` |
| "Prompt sets align between the runs on all 400 steps, and no validation or test prompt occurs in a training step" | `steps_with_identical_prompt_sets` and `heldout_prompts_in_training_steps` in `DATA_DIR/w3/checks.json` (`prepare/extract_calls.py w3`) | W3 rollouts |
| "W3-21M indexes all of them in 1,751,277 pages, 13.36 GiB at 8 KB"; "The 683 MB index of W3-1M" | arithmetic, no script: ceil(n / 12) pages of 8,192 bytes for n = 21,015,324 and 1,000,000 (`npages` of `LAYOUT_DIR/<w>/MARGO_B12.json` from `prepare/margo_layout.py` holds MARGO's page count) | |
| Compared methods: "one hypergraph of all pages exceeds 90 GB" (Mt-KaHyPar at 21M) | not included: no script of this release measures it | |
| 3,836 validation groups, 16,640 test groups | printed by `replay/make_schedule.py` (`table3.sh`) | `DATA_DIR/w3`, `HYPER_DIR/w3` |
| Each layout's share: fewest validation reads after a warm-up of 10 steps | `replay/select_deployment.py` (`shares` in `results/table3/<col>.selection.json`) | validation replays |
| ARC-Q and SET-Q (Section VII-A, Compared methods) | built by `layouts/refine_layout.py --sets $Q_r` (`layouts/README.md`); the paper names them among the compared methods and reports no number for them | |
| "On one thread, eight passes of a cache-blind layout take 1,991 to 13,830 s, and cache-aware refinement 317 to 5,943 s, because the residual unions keep only 8% to 73% of the incidences" | `refine_seconds` in the `.json` that `layouts/refine_layout.py` writes next to each W3 layout of Algorithm 1 (built with `NUMBA_NUM_THREADS=1`; the times depend on the host, and `prepare_seconds` holds the time before the passes); the share of incidences is `incidences` of a cache-aware layout's `.json` over `incidences` of the cache-blind layout of the same objective and draw | W3 layouts of Algorithm 1 |
| "Without P05-G, P05-C and the 1M ARC-C, HyperPage saves 19.8%, 14.8% and 13.0% at 1M" (mean of per-draw savings) | in each W3-1M column of `table3.sh`, also replay CLQ on residual unions (`layouts/refine_layout.py --objective CLQ --cached-pages h`, label CLQ-C) on validation and test, then `replay/select_deployment.py <col>.val.json <CLQ-C val> --candidates ARC-G,CLQ-C,CLQ-G,SET-G --out <col>.sel4.json`, the test replay with `--at <col>.sel4.json` and `replay/saving_table.py --selection <col>.sel4.json` (`HyperPage.per_draw`) | as Table III |
| "Without the three added last, it saves 3.1% and 1.9% at 15% and 25% of the full corpus" | `replay/select_deployment.py results/table3/<col>.val.json --candidates ARC-C,ARC-CF,ARC-G,CLQ-G,P05-C,P05-G,SET-G --out <col>.sel7.json` for `w3-21m_262144` and `w3-21m_437820`, then the test replay of `table3.sh` with `--at <col>.sel7.json` and `replay/saving_table.py --selection <col>.sel7.json` on it | `table3.sh` |
| "every margin of HyperPage and 85 of the 90 margins of the compared methods differ from zero at level 0.05"; "only CLQ-G at 25% of the 1M index and MARGO on uncached records at 5% of the full corpus lose it" | `replay/significance.py` (`table3.sh`, `results/table3/<col>.significance.json`) | test replays |
| "the margins of SET-G over MARGO and of the first three ablation steps are positive at level 0.05 under a bootstrap over prompts" | `python replay/prompt_bootstrap.py --unions HYPER_DIR/w3/test_U.npz --pages 83334 --layout MARGO=... --layout ARC-G=... --layout CLQ-G=... --layout SET-G=... --pair SET-G:MARGO --pair ARC-G:MARGO --pair CLQ-G:ARC-G --pair SET-G:CLQ-G` | W3-1M test unions, layouts |

### Section VII-B

| Paper item | Command | Inputs |
|---|---|---|
| Table III (all rows but the last; MARGO's reads per step in the first row) | `experiments/table3.sh` (`replay/saving_table.py` output per column). Not included: the full-corpus layouts of RCM and Proximity, so `table3.sh` leaves these two rows out of the four W3-21M columns; and the layouts of the W4 cells of Mt-KaHyPar on residual unions at 1% and 5% (10.0 and 1.8), where `table3.sh` replays the W3-1M ones, so these two cells are not reproduced | `prepare/`, `layouts/` |
| Table III, last row (MARGO's budget to match, x) | `experiments/table3.sh` (`replay/budget_match.py`, `results/table3/<col>.budget_match.json`); the paper prints only the upper end at 15% and 25% of the full corpus | MARGO over the budget grid |
| "saves 16.4% to 21.2% at 1M and 12.5% and 16.9% at 1% and 5% of the full corpus"; "the saving falls to 5.2% and 6.8%" | Table III, HyperPage row | |
| "MARGO needs 2.5 to 3.0 times ... and 1.5 to 1.75 times" | Table III, last row | |
| "the extra 68 to 103 MiB are 5% to 8% of the engine's peak resident memory" | not included: arithmetic on the last row of Table III and the engine's memory, with no script | |
| "Charged the measured memory of DiskANN's node cache, HyperPage still saves 18.20% to 29.02% at 1M on every draw" (mean of per-draw savings) | `replay/replay_hybrid.py --record-scale s` with s = 1.72, 1.45, 1.38 at 1%, 5%, 25% (validation and test), candidates SET-G, ARC-G, CLQ-G, P05-G and P05 / ARC refined with `--cached-records 2902 \| 25862 \| 90582` (`layouts/README.md`; the counts are 12 floor(rho D) / s at rho = 0.5, 0.75 and 0.5, MARGO's validation shares under these record scales), then `select_deployment.py` and `saving_table.py` | W3-1M |
| "Under a pure page LRU, our four cache-blind objectives beat MARGO at all seven budgets, by 14.22% to 40.02%"; "At 25% of the full corpus, they read more than MARGO for every share from 0.5 to 0.9" | `experiments/fig3.sh` (second table: share 0, and shares 0.5 to 0.9 of panel "21M, 25%") | |
| "At 15% and 25%, they read 3.5% to 11.5% more at their validation shares" | Table III files: SET-G, ARC-G, CLQ-G, P05-G in `w3-21m_262144` and `w3-21m_437820` | |
| "These shares cache 11.2% and 22.5% of the records, which serve 74.8% and 84.7% of the test accesses and carry 79% and 92% of the training incidences" | `python replay/hot_coverage.py --sets HYPER_DIR/w21/test_Q.npz --train-unions 'HYPER_DIR/w21/train{draw}_U.npz' --records 21015324 --cell 262144:0.75 --cell 437820:0.9` (shares: MARGO's validation shares) | |
| "at the other large budget it reads 0.8% to 3.0% more than MARGO on validation steps" | replay ARC-C and ARC-CF of h = 196608 at 437820 pages and those of h = 394038 at 262144 pages on validation (`replay/replay_hybrid.py`, all shares), under names that differ from the column's own layouts (for example `ARC-C-15=...`, `ARC-CF-15=...`), then `python replay/validation_margin.py <those files> results/table3/<col>.val.json` | full corpus |
| "If the share holds the most accessed pages instead, P05-C's saving falls to 8.43%, -3.33% and 1.56% at 1M" | not included as a script. The replays are `replay/replay_hybrid.py --cache-pages` for MARGO and P05-C at their 1M deployment shares; the numbers are P05-C's savings over MARGO on these replays (means of per-draw savings) | |
| STARLING, Proximity, RCM, HEAT, Mt-KaHyPar comparisons ("saves at most 2.8%", "5.4 to 7.8 points less", "9 of 10 columns", "10.7 to 12.1 points less", "2.2% to 7.8% more") | Table III (without the full-corpus cells of RCM and Proximity, see the Table III row) | |
| "it lowers the connectivity of the training unions by 18% to 26%" (Mt-KaHyPar against MARGO, full corpus) | `python replay/residual_connectivity.py --train-unions 'HYPER_DIR/w21/train{draw}_U.npz' --records 21015324 --pages 1751277 --hot-pages h --layout MARGO=... --layout Mt-KaHyPar-res=... --out ...` for every h of the full corpus, and with `--hot-pages 0` and `--layout Mt-KaHyPar=...`; the saving is 1 - O(Mt-KaHyPar) / O(MARGO) per draw | full corpus |
| "P05-C's layouts also have 11.7% to 12.0% lower connectivity of the residual training unions than the two-level partitioner's" | `replay/residual_connectivity.py` at 1M with P05-C and the two-level Mt-KaHyPar layout on residual unions (`layouts/mtkahypar_two_level.py --cached-pages h`) for h = 416, 3750, 15625 | W3-1M |
| "residual training connectivity ranks the validation reads of each W3 column's 18 layouts with Spearman's r_s of 0.06 to 0.92. Validation replay ranks test reads at 0.92 to 1.00" | per W3 column: `replay/residual_connectivity.py --hot-pages h` with the 18 layouts of `table3.sh`, then `python replay/rank_correlation.py --connectivity <that output> --validation results/table3/<col>.val.json --test results/table3/<col>.test.json --budget D --share rho` (rho = MARGO's validation share of the column) | `table3.sh` |
| Fig. 4 | `experiments/fig4.sh` | 1M, 5M, 10M, 21M |
| "At 1%, its saving falls by 6.1 points from 1M to 5M and by 2.6 more to the full corpus" | differences of the HyperPage values that `experiments/plot_scale.py` prints | `fig4.sh` |
| "the training unions cover 33% of the records instead of 86%" | first line of `replay/hot_coverage.py` output (training coverage), for `w21` (21015324 records) and `w3` (1000000 records) | |
| "From step 1 to step 240, the savings of the frozen 1M deployments fall by only 2.0 to 2.6 points" | `replay/make_schedule.py w3 --split s1_40` and `--split s201_240` (with `--drop DATA_DIR/<w>/drop.json`), `replay/replay_hybrid.py --at results/table3/w3-1m_<b>.selection.json` with MARGO and the deployed layout, `replay/saving_table.py`; the number is the difference of the two windows | W3-1M windows |
| "On full-corpus steps 1 to 40 and 201 to 240 ... read 4.7% to 22.1% and 4.9% to 12.5% fewer pages than MARGO" (means of per-draw savings); "Each also beats every other candidate at its validation share on every draw" | as above at the full corpus with every candidate of the column (`saving_table.py --out`, `per_draw`) | full-corpus windows |
| Table III W4 columns; "HyperPage reads 19.4%, 14.0% and 14.5% fewer pages ... on every draw" | `experiments/table3.sh` (W4 part) | |
| "its saving moves by at most 1.21 points from the first checkpoint to the last" | `python replay/block_savings.py --selection results/table3/w4_<b>.selection.json --test results/table3/w4_<b>.test.json --block 41` (41 test steps per checkpoint) | `table3.sh` |
| "MARGO's share moves from 0.90 on W3 to 0.75 on W4, so selection deploys P05-G" | `shares` and `deployment` of `results/table3/w3-1m_4167.selection.json` and `w4_4167.selection.json` | |
| "Replayed in the arrival order that the engine logs with MARGO's layout, MARGO's reads match the engine's within 0.001%" | `experiments/engine_order.sh` (engine runs in its header): `replay/engine_order.py` turns the engine's order log into one replay set per search round, and `replay_hybrid.py` replays them. The number compares the reads of row `MARGO\|0\|D\|rho` of `results/engine_order/w3-1m_<D>.val.json` (draw 0, the engine's hot set) with the engine's `page_reads`; no script computes this difference | MARGO's 12-thread order logs on W3-1M validation, D = 833 and 4167 |
| "The timed 1M layouts at 1% and 5% are selected under this order": CLQ-G at share 0.1 and P05-C (2,083 hot pages) at 0.5 on draw 0 | `experiments/engine_order.sh` (`deployment` of `results/engine_order/w3-1m_<D>.selection.json`); P05-C with 208 hot pages is `layouts/refine_layout.py --objective P05 --sets $U_r --start $MARGO --cached-pages 208` | as above |
| Table VI | `experiments/table6.sh` | |
| "our objectives read 14.99% to 20.81% fewer pages than MARGO at 1M and 9.64% to 13.65% fewer on the full corpus. Mt-KaHyPar saves 12.44% at 1M but 0.63% on the full corpus" | Table VI | |
| "the 128 live groups of a step hold 144,703 pages at 1M" (SET-G, three draws) | `python replay/buffer_memory.py --unions HYPER_DIR/w3/test_U.npz --schedule results/table6/schedule_w3_test.npz --pages 83334 --layout 'SET-G=LAYOUT_DIR/w3/SET-G_r{draw}.npy'` | |
| "2.54 to 2.95 GiB on the full corpus" (draw-0 layouts) | `replay/buffer_memory.py` on `w21` with `--draws 0` and MARGO, ARC-G, CLQ-G, SET-G, P05-G | |

### Section VII-C

| Paper item | Command | Inputs |
|---|---|---|
| Table VII | `experiments/table7.sh` | `table3.sh` |
| "the better cache-blind layout saves 8.2% to 20.0%"; "Refining ARC cache-aware adds 1.2 to 11.9 points"; "ARC-C beats MARGO in every W3 column, by 1.2% to 18.1%"; MARGO on uncached records "4.6% and 6.8%", "adds 0.6 points ... loses 4.2" | Table VII | |
| "at 5% of the full corpus it reads 0.9% more than MARGO, a margin not significant with moving blocks" | `results/table3/w3-21m_87564.significance.json`, MARGO-uncached | `table3.sh` |
| "in every column its pick is also the best of its candidates on test" | `replay/saving_table.py` per column: HyperPage against every candidate | `table3.sh` |
| "an arc-restricted objective on logged group unions saves 14.99%"; "Dropping the arc restriction adds 3.90%"; "The set form and P05 add 1.51% and 1.58%" | Table VI group means: 1 - CLQ-G / ARC-G, 1 - SET-G / CLQ-G, 1 - P05-G / SET-G | `table6.sh` |
| "the off-arc share of co-located pairs rises from 0.404 to 0.787" (draw 0) | `python replay/offarc_share.py --unions HYPER_DIR/w3/test_U.npz --graph INDEX_DIR/w2/margo_R64_L125_mem.index --records 1000000 --pages 83334 --layout ARC-G=LAYOUT_DIR/w3/ARC-G_r0.npy --layout CLQ-G=LAYOUT_DIR/w3/CLQ-G_r0.npy` | graph of the 1M index |
| "SHP's move rule saves 0.08% where P05-G saves 20.81%" | Table VI (SHP-G, P05-G) | |
| "Refining on sub-queries shuffled across the groups of a step keeps 16.28% of the 19.54% at group scope" | `replay/scope_reads.py` as in `table6.sh` with `--layout 'SET-shuffled=LAYOUT_DIR/w3/SET-shuffled_r{draw}.npy'` (SET refined on the shuffled unions, `layouts/README.md`) | |
| "An exact-string result cache of at most 1/64 of the budget ... saves MARGO 16.82% to 25.17% of its reads and leaves HyperPage 12.14% to 20.59% ahead" (1M part: 25.17 / 25.17 / 21.19 and 20.59 / 15.91 / 15.76, means of per-draw savings) | per W3-1M budget D: (1) `replay/replay_result_cache.py --sets HYPER_DIR/w3/val_Q.npz --schedule <validation schedule> --budgets D --layout MARGO=... --layout <deployed label>=... --rc-share PHI` for PHI = 0, 0.00390625, 0.0078125, 0.015625, 0.0625, 0.25 (all shares, one file per PHI); (2) `replay/result_cache_saving.py select --validation <those files> --selection results/table3/<col>.selection.json --out rc_selection.json`, which prints the chosen (layout, share, phi) per draw; (3) per chosen (share S, phi P): `replay/replay_result_cache.py` on the test split with `--shares S --rc-share P` and the same layouts; (4) `replay/result_cache_saving.py read --rc-selection rc_selection.json --test <the files of (3)> --reference-test results/table3/<col>.test.json --selection results/table3/<col>.selection.json` | W3-1M; the full-corpus part (16.82 and 12.14) is not included |
| W1 "16.71% to 24.24%", W2 "20.50% to 25.38%" (means of per-draw savings) | `experiments/other_workloads.sh` (`HyperPage.per_draw` of `results/other_workloads/w{1,2}_<D>.saving.json`). Settings: candidates MARGO plus ARC-G, CLQ-G, P05-C, P05-G, SET-G (W1 at 1%: without P05-C); MARGO at the share with the fewest validation reads on each draw (`saving_table.py --incumbent-per-draw`) | W1, W2 |
| "18.62% for ARC-G at 1% on W1" (mean of per-draw savings) | `experiments/other_workloads.sh` (`ARC-G.per_draw` of `results/other_workloads/w1_833.saving_shared_share.json`: both layouts at their shares averaged over the draws) | W1 |
| DEEP1M "the cache-blind layouts save only 0.33% to 4.74%" | `experiments/other_workloads.sh` (`saving` of ARC-G and P05-G in `results/other_workloads/deep_<D>.saving.json`); `replay/make_schedule.py deep` cuts the sub-queries into 30 validation and 130 test steps of equal size | DEEP1M |
| Page sizes: "13.96% and 12.00% with 6 records per page and 4.49% and 5.56% with 2" (means of per-draw savings); "HyperPage still reads less than MARGO on W3-1M on every draw" | `experiments/other_workloads.sh` (`HyperPage.per_draw` of `results/other_workloads/w3_b<B>_<D>.saving.json`, B = 6, 3, 2) | W3-1M layouts at B = 6, 3, 2 |

### Elsewhere

| Paper item | Directory |
|---|---|
| Tables IV and V, search-time, payback and engine-check numbers | `engine/` |
| Section VII-C, "Of the gain that SHP predicts on a round's snapshot, 78% to 80% does not materialize" | `layouts/shp_layout.py`: per 1M draw, 1 - sum `decrease` / sum `snapshot_gain` over the rounds of `history` in the SHP-G layout's `.json` (`layouts/README.md`, SHP-G) |
| Section VII-D (543.2, 974.4, 71 instances, window bound "11 to 48 times") | `bounds/`, `layouts/` |
