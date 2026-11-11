# Replay engine

The engine of Section VI-E: a patched DiskANN that reads pages in the order of any layout, under read-once scoped
buffers or under the hybrid cache H_{D,rho} of Definition 7, plus the tools that check it against the replay and time
it (Table IV, Table V and the Search-time paragraph of Section VII-B).

```
engine/
  fetch_sources.sh          clone DiskANN (and, for Table V, MARGO) at the pinned commits
  build.sh                  apply the edit scripts to copies of the DiskANN checkout and build the binaries
  relayout.py               write an index file in a given layout, with its node -> (page, slot) map
  src/                      our sources, copied into the DiskANN tree by the edit scripts
    traversal_log.h           per-thread traversal log
    search_and_log.cpp        logging search application
    batch_page_buffer.h       read-once scoped page buffer
    batch_search.cpp          read-once scoped search application
    hybrid_page_cache.h       page LRU of the hybrid cache
    hybrid_search.cpp         hybrid-cache search application
  patches/                  edit scripts for a DiskANN (or MARGO) source tree
    edit_util.py              anchored text edits (exact strings, or regular expressions that must match once)
    add_logger.py             traversal logger + search_and_log
    add_layout_map.py         node -> (page, slot) map from <disk index>.layout
    add_scoped_buffer.py      page size at compile time, load-time guards, scoped buffer hook, timers, batch_search
    add_hybrid_cache.py       hybrid-cache hook (shared page LRU, frontier order), hybrid_search
    add_order_log.py          arrival-order log (separate copy, logging only)
    set_aio_events.py         asynchronous I/O queue depth for 64 to 256 threads (separate copy)
    port_margo_page_search.py MARGO's page-search engine at 8 KB with a fixed record cache list
    THIRD_PARTY_NOTICE.md     DiskANN's licence, for the excerpts of DiskANN that the edit scripts quote
  checks/                   engine against the stock reference and against the replay
    stage_scoped_sample.py    training-group sample, query file, scope pointers, replayed reads per scope
    check_scoped.py           results, expansions and per-scope reads of the scoped engine
    stage_hybrid.py           waves, queries, hot-record lists and replayed per-step reads of hybrid runs
    check_hybrid.py           per-step reads, expansions and results of one-thread hybrid runs
  timing/                   timed comparisons
    run_arm.sh                one timed arm (hybrid engine or page search), bracketed by affinity snapshots
    read_speed.py             read speed of the index files (O_DIRECT random page reads)
    read_check.py             read-speed rule before a round (and the cross-reference rule of `--cross`)
    affinity_check.py         foreign processes on the run's CPUs
    summarize.py              validity of rounds and the statistics of Table IV and Table V
    payback.py                runs until a layout repays its refinement and rewrite
  page_search/              inputs of MARGO's page-search engine (Table V)
    build_margo.sh            build the ported engine from a local MARGO checkout
    write_partition.py        a layout in the partition format of MARGO's index_relayout
    cache_list.py             record cache list ranked by training frequency
    check_partition.py        partition file against a layout, page-search index file against the stock index
    recall.py                 recall@10 of page-search results (staged sub-queries or sampled queries)
    readout.py                L chosen on validation sweeps; test reads per call against MARGO at equal recall
```

Python scripts need numpy and numba and import the `hyperpage` package of this release (run them with the release
root on `PYTHONPATH`, or install it with `pip install -e .`). Every script prints its arguments with `--help`.

## Upstream code

* **DiskANN**, https://github.com/microsoft/DiskANN, branch `cpp_main`, commit
  `78256bbab4685e1774e78d331e081a153be26823`, MIT licence (Copyright (c) Microsoft Corporation). The engine is this
  tree plus the edits of `patches/`. `add_logger.py`, `add_layout_map.py`, `add_scoped_buffer.py` and
  `add_hybrid_cache.py` quote short excerpts of DiskANN's sources: as exact anchors that locate each edit, and inside
  the replacement texts, where the upstream lines around an insertion are kept; `edit_util.py` quotes the link line of
  DiskANN's applications. `patches/THIRD_PARTY_NOTICE.md` reproduces DiskANN's licence for these excerpts.
  `add_order_log.py` anchors on our own inserted lines only, and `set_aio_events.py` locates the definition of the
  macro `MAX_EVENTS` by a regular expression on its name. The applications in `src/` follow the structure of
  DiskANN's search applications and say so in their first lines. The upstream tree itself is not part of this release.
* **MARGO**, https://github.com/CodenameYZY/MARGO, commit `7e24023b950c1a31d993476a40f444dd69617ba7`. Its repository
  states no licence, so none of its files is distributed here: `fetch_sources.sh --with-margo` clones it, and
  `patches/port_margo_page_search.py` edits the local copy. That script contains no line of MARGO's source: it
  locates its four edits by regular expressions made of identifier names and the punctuation that delimits them (the
  definition of the macro `SECTOR_LEN`, the call of the member function `generate_cache_list_from_sample_queries`, the
  declaration of the variable `recall`, the `add_subdirectory` command of `graph_partition`), stops unless each
  pattern matches exactly once, and inserts code of our own, which reads MARGO's variables by name.

```sh
bash engine/fetch_sources.sh third_party            # third_party/diskann at the pinned commit
bash engine/fetch_sources.sh third_party --with-margo
```

## Build

DiskANN's build dependencies are needed (our builds: GCC 11.4 and CMake 3.25 on Linux, with OpenMP, Boost
program_options, Intel MKL and libaio). `build.sh` leaves the checkout unchanged and builds each tree on a copy of the
tree before it, applying the edits in order: three trees, and one more for each of `--order-log` and `--aio-events`:

| tree | edits | page size | binaries |
|---|---|---|---|
| `logger` | `add_logger.py` | 4 KB (stock) | `search_and_log`: every traversal log of the replay |
| `engine` | + `add_layout_map.py`, `add_scoped_buffer.py` | 8 KB | `batch_search` (read-once scopes), `search_and_log` |
| `hybrid` | + `add_hybrid_cache.py` | 8 KB | `hybrid_search` (every timed run) |
| `hybrid_order` (`--order-log`) | + `add_order_log.py` | 8 KB | `hybrid_search` with the arrival-order log |
| `hybrid_aio` (`--aio-events 128`) | + `set_aio_events.py` | 8 KB | `hybrid_search` for 64 to 256 threads |

```sh
bash engine/build.sh --diskann third_party/diskann --work build/engine --order-log --aio-events 128 \
    --cxx-flags "-I<dependency include dir>" -- \
    -DMKL_PATH=<mkl lib dir> -DMKL_INCLUDE_PATH=<mkl include dir> -DOMP_PATH=<openmp lib dir> \
    -DBOOST_ROOT=<boost prefix> -DCMAKE_CXX_STANDARD_LIBRARIES=-latomic
```

Arguments after `--` go to CMake unchanged. The page size is fixed at compile time (`-DDISKANN_SECTOR_LEN=8192`,
`--page-size`); at load time the engine refuses an index file whose records per page do not fit the compiled page size
or whose size does not match its page count. Every edit leaves the stock code path in place: without a `.layout` file,
a scoped buffer or a hybrid cache installed, the binaries search exactly as stock DiskANN.

The page-search engine of Table V:

```sh
bash engine/page_search/build_margo.sh --margo third_party/margo --work build/margo --aio-events 128 -- <cmake args>
```

builds `search_disk_index` and `index_relayout` with 8 KB pages, and a second copy with 128 I/O events per thread.

## Writing an index file in a layout

A layout is an `.npy` array of int32 page numbers, one per record, with at most B = 12 records per page at 8 KB
(records are 644 bytes). `relayout.py` reads the stock 4 KB index written by the index builder and writes the same
records in the order of the layout:

```sh
python engine/relayout.py <stock prefix> <layout.npy> <out prefix>        # --unit 8192 by default
```

It writes `<out prefix>_disk.index` (page p at byte 8192 (1 + p), records of a page in increasing id, the stock header
with records per page and file size rewritten) and `<out prefix>_disk.index.layout` (u64 n, u32 page[n], u32 slot[n]),
and hard-links the PQ files, so identifiers, PQ codes and the entry point are identical across layouts. The engine uses
the map whenever `<disk index>.layout` exists: page of node v = 1 + page[v], offset = slot[v] x 644.

## Applications

All three take `<float|int8> <l2|mips> <index prefix> <query.bin>` first; query files are DiskANN `.bin` matrices
(int32 rows, int32 columns, then the rows). The timed W3 runs and the checks use int8, L2, W = 4, K = 10, L = 500 at 1M
and L = 800 on the full corpus; the logging runs of `prepare/` set their own data type and L.

**Logging** (`logger/build/apps/search_and_log`):
`search_and_log int8 l2 <prefix> <query.bin> <L> <W> <K> <threads> <out prefix> 0`.
Stock search with no node cache. Writes `<out>.log.bin`: u64 nq, total expanded, total rounds, total reads; u64
offsets noff[nq+1], roff[nq+1], soff[nq+1]; u32 expanded nodes in expansion order; u32 parent of each expanded node;
u32 end of each round; u32 page (sector) of each issued read; u32 n_ios[nq], n_hops[nq], cache hits[nq]; f32
io_us[nq]. And `<out>.res.bin`: u32 nq, u32 K, u32 ids[nq K], f32 distances[nq K].
`hyperpage.formats.read_log` reads the log.

**Read-once scopes** (`engine/build_8192/apps/batch_search`):
`batch_search int8 l2 <prefix> <query.bin> <scope_ptr.u64> <L> <W> <K> <threads> batch <out> <log 0|1> <warm scopes>`.
`scope_ptr.u64` holds u64 ptr[0..m] over the rows of the query file. Threads take whole scopes and run their
sub-queries in order, each scope with its own page buffer, emptied at its end. Writes `<out>.batches.bin` (u64 m, then
13 u64 per scope: reads, hits, shared reads, calls, expanded, latency ns, thread, cpu, io ns, pq ns, neighbour ns,
rounds, rounds without reads), `<out>.res.bin`, `<out>.log.bin` (log 1) and one `summary` line.

**Hybrid cache** (`hybrid/build_8192/apps/hybrid_search`):
`hybrid_search int8 l2 <prefix> <query.bin> <wave_ptr.u64> <wave_step.u32> <L> <W> <K> <threads> <hot.u32> <lru pages>
<out> <log 0|1> <warm.bin> frontier`.
The records of `hot.u32` enter DiskANN's node cache; every other page passes through one LRU of `<lru pages>` pages
shared by all threads, from an empty start. Waves (u64 pointers over the query rows, u32 step of each wave) arrive one
after the other, and the threads of a wave take its sub-queries dynamically. `warm.bin` runs first, untimed, with the
page LRU off. Writes `<out>.steps.bin` (u64 steps, then 14 u64 per step: reads, hits, waits, calls, expanded, cached,
wall ns, io ns, pq ns, neighbour ns, lock ns, wait ns, busy ns, rounds without reads), `<out>.calls.bin` (u64 nq, then
latency ns, reads, expanded per call), `<out>.res.bin`, `<out>.log.bin` (log 1), `<out>.host` (`/proc/stat` and
`/proc/diskstats` around the timed section) and one `summary` line with, among others, `secs`, `ms_per_step`,
`page_reads`, `lat_ms_median`, `lat_ms_p99`, `proc_read_bytes`, `hwm_kb`. The last argument `stock` instead of
`frontier` processes the cached records of a round before the read ones, as stock DiskANN does.

Preconditions of the hybrid cache: the page LRU needs at least one page (`<lru pages>` >= 1), so a share rho = 1,
which leaves no page to the LRU, cannot be timed; and a miss needs an unpinned page. A miss takes a free page or evicts
the least recent unpinned page; a round pins the pages of its frontier (at most W) until its records are processed, and
the engine stops with an error when every page of the LRU is pinned. An LRU of at least W x threads pages therefore
always has an unpinned page (the timed runs with 12 threads and W = 4 have LRUs of 625 pages or more).

With the `hybrid_order` build and the environment variable `HYB_ORDER_LOG` set, `hybrid_search` also writes
`<out>.order`: u64 quadruples (sequence, call, round of the call, nodes expanded in the round), one per search round,
the sequence taken under the cache mutex. Sorted by sequence, they give the arrival order of the LRU operations of all
threads, in which the replay can be run.

## Checking the engine against the replay

**Read-once scopes** (Section VII-A: the stock results and expansion sequences of 7,850 W3 sub-queries).

```sh
python engine/checks/stage_scoped_sample.py --splits <splits.npz> --key train1 --unions <train1 group unions.npz> \
    --queries <qx8.npy> --layout MARGO=<MARGO.npy> --layout ARC-G=<ARC-G.npy> --out scope
# reference: the stock logging binary on the stock index
search_and_log int8 l2 <stock prefix> scope/scope_q.bin 500 4 10 12 scope/ref 0
# engine: one run per layout, 8 KB file of that layout, logging on
batch_search int8 l2 <layout prefix> scope/scope_q.bin scope/scope_bptr.u64 500 4 10 12 batch scope/eng_ARC-G 1 0
python engine/checks/check_scoped.py scope/ref scope/eng_ARC-G scope/scope_bptr.u64 <ARC-G.npy> scope/lambda_ARC-G.npy \
    --proc-read-bytes <proc_read_bytes of the summary line>
```

The sample is 200 training groups of draw 1, the first 200 of a seeded permutation of the groups (the default seed of `checks/stage_scoped_sample.py`); its first 20 groups
form `scope/warm_q.bin` (799 sub-queries on W3), the untimed warm-up of every timed run at both scales. `check_scoped.py`
reports check (i) (ids, distances, expansion sequences and rounds identical to stock) and check (ii) (reads of every
scope equal to lambda_pi of the scope's union, the read pages exactly the union's pages).

**Hybrid cache, one thread** (Section VII-A: with one thread the engine reads exactly the replayed pages of every 1M
validation step).

```sh
python engine/checks/stage_hybrid.py --sets <val per-query sets.npz> --schedule <val schedule.npz> \
    --key val --queries <qx8.npy> --train-unions <train0 group unions.npz> --records 1000000 --pages 83334 \
    --budget 833 --arm MARGO <MARGO.npy> 0.5 --arm HP <layout.npy> 0.5 --out hyb
OMP_PLACES="{0}" OMP_PROC_BIND=true hybrid_search int8 l2 <MARGO prefix> hyb/val_q.bin hyb/val_wptr.u64 hyb/val_wstep.u32 \
    500 4 10 1 hyb/hot_MARGO.u32 <lru_pages of val_info.json> hyb/MARGO 1 scope/warm_q.bin frontier
python engine/checks/check_hybrid.py --sets <val per-query sets.npz> --lock hyb/val_lock.npy \
    --run MARGO hyb/MARGO hyb/ref_val_MARGO.npy --run HP hyb/HP hyb/ref_val_HP.npy
```

`stage_hybrid.py` orders the sub-queries in turn lockstep, cuts them into one wave per (step, turn), writes the query
file in that order, the hot-record list of every arm (the B floor(rho D) records most frequent in training draw 0) and
the replayed per-step reads computed with the replay's kernel (`hyperpage.cache.hybrid_cache_misses`). `check_hybrid.py`
compares the engine's per-step reads with them, the logged expansions with the replayed sets, and the results across
arms.

## Timed comparisons (Table IV)

Each budget compares MARGO with HyperPage's timed layout over the 130 test steps, in rounds. Stage the test split once
per budget with `stage_hybrid.py --key test --no-reference`, with the arms and shares below. A round:

1. `read_speed.py --cpus <timed CPUs> <file A> <file B>` (20,000 random 8 KB reads per file, 3 repetitions, one
   thread per timed CPU), then `read_check.py` on its output with a per-budget `--reference` file: the round runs only if
   the two mean latencies are within 5% of each other and each within 10% of its value at the first round; otherwise it
   is void and has no runs.
2. Both arms with `timing/run_arm.sh`, MARGO first in even rounds and HyperPage first in odd rounds:

   ```sh
   bash engine/timing/run_arm.sh --engine hybrid --bin hybrid/build_8192/apps/hybrid_search --cpus <12 CPUs> --threads 12 \
       --index <layout prefix> --queries test/test_q.bin --wave-ptr test/test_wptr.u64 --wave-step test/test_wstep.u32 \
       --L 500 --hot test/hot_HP.u32 --lru <lru pages> --warm scope/warm_q.bin --out runs/r0_HP
   ```

Rounds continue until a budget has 10 valid pairs or 5 void ones (at most 16 rounds). Then

```sh
python engine/timing/summarize.py --round r0 MARGO=runs/r0_MARGO HP=runs/r0_HP --round r1 ... --void-round r2 \
    --idle-cpus <CPUs kept idle during the runs> --disk <device of the index files> --out summary.json
python engine/timing/payback.py summary.json --investment <prep + refine seconds of the layout> --rewrite <relayout seconds>
```

`summarize.py` voids a round when any of its runs is flagged: more than 0.6 busy cores on the idle CPUs or foreign disk
traffic above 5% of the run's own reads (host load), a foreign process on the run's CPUs above 1% of its wall time
(`affinity_check.py`, written by `run_arm.sh`; processes can be exempted by name with `--exempt`, and the reported runs
exempted the host's metrics exporter; the check compares a snapshot before the run with one after it, so a process
that exits before the second snapshot is not seen), or I/O time per read changing by more than 15% from the first to the last fifth
of the run. A round in which a run has no summary line is reported as incomplete, apart from the valid and void rounds,
and enters no statistic. Over the first 10 valid rounds it prints the columns of Table IV: pairs, median seconds per
step of both arms, reads saved (1 - median reads of HyperPage / median reads of MARGO), time saved (the median of the
paired savings 1 - t_HP / t_MARGO), the two-sided sign-test p, and the p50 and p99 falls. All thresholds are arguments
with these defaults. `payback.py` computes the saved wall seconds per step (MARGO's seconds per step x the median
saving), the steps until (prep + refine + rewrite) is repaid, and the runs of 400 steps in wall time and in core-seconds
(12 threads).

Arms of Table IV (draw 0; `--budget` and per-arm share of `stage_hybrid.py`; HyperPage's layouts come from the layout
tools of this release):

| row | budget D (pages) | MARGO share | timed layout, share | L |
|---|---|---|---|---|
| 1M, 1% | 833 | 0.25 | CLQ-G, 0.1 | 500 |
| 1M, 5% | 4,167 | 0.5 | P05-C (2,083 hot pages), 0.5 | 500 |
| 1M, 25% | 20,834 | 0.75 | ARC-C (15,625 hot pages), 0.75 | 500 |
| 21M, 1% | 17,513 | 0.5 | ARC-C (8,756 hot pages), 0.5 | 800 |
| 21M, 5% | 87,564 | 0.25 | ARC-C (21,891 hot pages), 0.25 | 800 |
| 21M, 15% | 262,144 (2 GiB) | 0.75 | HEAT on residual unions, 0.75 | 800 |
| 21M, 25% | 437,820 | 0.9 | MARGO on uncached records (394,038 hot pages), 0.9 | 800 |

Hot pages are floor(rho D) and the page LRU holds the other D - floor(rho D) pages (1M, 1%: 208 hot pages and an LRU
of 625 for MARGO, 83 and 750 for CLQ-G; 21M, 25%: 394,038 hot pages and an LRU of 43,782). The budgets are those of the
replay (Table III); the timed run at 25% of the full corpus used D = 437,820 pages.
The 1M index has 1,000,000 records in 83,334 pages and the full corpus 21,015,324 records in 1,751,277 pages. The
21M rows at 1% and 5% come from rounds of five arms (MARGO, HyperPage and HEAT on residual unions in the hybrid engine,
MARGO and HyperPage in page search): pass all five runs of a round to `--round`, use `--compare HP:MARGO` (or the arm
names chosen), `--max-pairs 5`, the five files to `read_speed.py` with `read_check.py --drift-form reference_over_current`,
and rotate the arm order by one per round; these rounds run until 3 valid or 5 void rounds (at most 7).

## MARGO's page-search engine (Table V)

The port changes three things in MARGO's `search_disk_index`: 8 KB pages; a record cache list read from the file in the
environment variable `RECORD_CACHE_LIST` instead of MARGO's own selection from sample queries; and one `HPSTAT` line
per L with the total page reads, seconds, QPS, p50 and p99 latency and peak memory. Inputs:

```sh
python engine/page_search/write_partition.py <layout.npy> ps/m_partition.bin
build/margo/margo_page_search/release/tests/utils/index_relayout <stock prefix>_disk.index ps/m_partition.bin
mv ps/m_partition_tmp.index ps/m_disk.index          # the 8 KB page-search file written by MARGO's tool
for s in pq_compressed.bin pq_pivots.bin sample_data.bin sample_ids.bin mem.index mem.index.data mem.index.weights; do
    ln -s "$(realpath <stock prefix>_$s)" ps/m_$s    # the other index files are shared with the stock index
done
python engine/page_search/cache_list.py <train0 group unions.npz> <D> <records> <cache.u32>
bash engine/timing/run_arm.sh --engine page-search --bin build/margo/margo_page_search/release/tests/search_disk_index \
    --cpus <12 CPUs> --threads 12 --index-dir <dir of m_disk.index> --queries test/test_q.bin --L <L> \
    --cache-list cache.u32 --cache-records <12 D> --disk <device> --out runs/r0_PS-M
```

`--index-dir` is the directory of the page-search index (`ps` above, files with the prefix `m`). The record cache holds
the whole budget, 12 D records ranked by training frequency. The page-search runs used D = 833, 4,166 and 20,833 at 1M
(cache lists of 9,996, 49,992 and 249,996 records), one page fewer at 5% and 25% than the 4,167 and 20,834 pages of the
hybrid arms and the replay, and D = 17,513 and 87,564 on the full corpus, as in the hybrid arms. Page search takes the
smallest L with validation recall@10 of 0.95 (1M: 350, 375 and 500 with MARGO's layout, 375, 400 and 500 with
HyperPage's page-search layout; full corpus: 600 at 1% for both layouts, 600 and 700 at 5%), W = 1. Rounds hold the
page-search and hybrid arms of a budget; `summarize.py` reads both kinds of run (`--compare PS-H:PS-M --compare
PS-M:HYB-M ...`) and, for page-search runs, takes seconds per step as seconds / `--steps` (130), checks host load on
writes only, and the device drift on the 5 s samples of `run_arm.sh`. Runs with 64, 128 and 256 threads use the
`hybrid_aio` and `margo_page_search_aio` builds and `--run NAME=PREFIX`.

**Checks of the page-search files.** For MARGO's layout, the partition file that MARGO's partitioner wrote
(`<stock prefix>_margo_B12/m_partition.bin`, see `prepare/margo_layout.py`) can go to `index_relayout` directly; the
1M MARGO file of Table V is written from it. `same-pages` confirms that a partition file holds the pages of a layout
up to page numbering, and `file` that the 8 KB file has one header page plus one page per layout page, B = 12 records
per page in its header, and, on 100 sampled pages (seed 23), the stock record of every listed id byte for byte:

```sh
python engine/page_search/check_partition.py same-pages LAYOUT_DIR/w3/MARGO_B12.npy <stock prefix>_margo_B12/m_partition.bin
python engine/page_search/check_partition.py file <stock prefix>_disk.index ps/m_partition.bin ps/m_disk.index
```

**Choice of L.** Validation sweeps run MARGO's binary directly, with one HPSTAT line and one result file per L. At 1M
the sweep queries are the staged validation sub-queries (`stage_hybrid.py --key val`, directory `hyb` above) whose
query string has exact neighbours in `prepare/ground_truth.py`'s sample, 11,491 sub-queries on W3-1M; on the full
corpus they are the 5,000 sampled validation queries `DATA_DIR/w21/gtq_val.bin` themselves. The record cache list of a
HyperPage page-search layout is ranked on the training draw of that layout (`cache_list.py` on `train<r>_U.npz`), and
MARGO's on draw 0.

```sh
python engine/page_search/recall.py subset --queries hyb/val_q.bin --schedule <val schedule.npz> --lock hyb/val_lock.npy \
    --gt-strings DATA_DIR/w3/gtq_val_strid.npy --out hyb/val_gt          # hyb/val_gt_q.bin, hyb/val_gt_rows.npy
mkdir -p sweep/PS-M_833
RECORD_CACHE_LIST=cache_833.u32 search_disk_index --data_type int8 --dist_fn l2 --index_path_prefix ps/m \
    --query_file hyb/val_gt_q.bin --gt_file null -K 10 --result_path sweep/PS-M_833/r --num_nodes_to_cache 9996 -T 12 \
    -L 50 75 100 125 150 200 250 300 325 350 375 400 500 600 800 -W 1 --mem_L 0 --use_page_search 1 --use_ratio 1.0 \
    --disk_file_path ps/m_disk.index > sweep/PS-M_833/search.log
python engine/page_search/recall.py sweep sweep/PS-M_833/search.log sweep/PS-M_833/r --gt DATA_DIR/w3/gt_val.npy \
    --schedule <val schedule.npz> --lock hyb/val_lock.npy --gt-strings DATA_DIR/w3/gtq_val_strid.npy \
    --rows hyb/val_gt_rows.npy --out sweep/PS-M_833/recall.txt
python engine/page_search/readout.py select --sweep PS-M_833 sweep/PS-M_833/search.log sweep/PS-M_833/recall.txt \
    --sweep PS-H_833_r0 ... --out select.json
```

`readout.py select` takes, per arm, the smallest L whose validation recall@10 reaches 0.95 (`--rule smallest-L`);
sweeps given under one arm name are pooled. The 1M grid is L in {50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600,
800}, with 325, 350 and 375 added at 1% and 5%; the full-corpus grid is {200, 300, 400, 500, 600, 700, 800, 1000, 1200,
1500}, recall computed against `gt_val.npy` row by row (`recall.py sweep ... --gt DATA_DIR/w21/gt_val.npy`, no
mapping arguments). For the comparison with MARGO's own cache selection, sweep MARGO's layout at 1M once without
`RECORD_CACHE_LIST` (MARGO then selects its cache from sample queries) and once with the training-frequency list, each
with W in {1, 2, 4} (`-W 1`, `-W 2`, `-W 4`, the 12 L values above), and select with
`readout.py select --rule fewest-reads --fewer PS-M-own_833,PS-M_833 ...`: per arm the (W, L) with recall@10 of at
least 0.95 and the fewest validation reads (ties to the smaller W, then L), and of the two arms the one that reads less.

**Test page reads at equal recall (1M).** One run over all 467,989 test sub-queries at each arm's chosen L, MARGO's
layout once and HyperPage's page-search layout of every training draw, each with its own cache list:

```sh
bash engine/timing/run_arm.sh --engine page-search --bin <search_disk_index> --cpus <12 CPUs> --threads 12 \
    --index-dir ps --queries test/test_q.bin --L 350 --cache-list cache_833.u32 --cache-records 9996 --disk <device> \
    --out counts/PS-M_833
python engine/page_search/recall.py recall counts/PS-M_833_350_idx_uint32.bin --gt DATA_DIR/w3/gt_test.npy \
    --schedule <test schedule.npz> --lock test/test_lock.npy --gt-strings DATA_DIR/w3/gtq_test_strid.npy \
    --out counts/PS-M_833.recall.json
python engine/page_search/readout.py saving --margo counts/PS-M_833.log counts/PS-M_833.recall.json \
    --hp counts/PS-H_833_r0.log counts/PS-H_833_r0.recall.json --hp ... --hp ...
```

`saving` prints the reads per call (HPSTAT `sum_ios` / `nq`), the test recall@10 over the 11,159 test sub-queries with
exact neighbours, and per draw the saving 1 - reads(HyperPage) / reads(MARGO), with mean, standard deviation and
whether every draw saves. On the full corpus, test recall comes from a run on `DATA_DIR/w21/gtq_test.bin` at the chosen
L (`recall.py recall ... --gt DATA_DIR/w21/gt_test.npy`).

**Cells of Table V.** One `summarize.py` call per budget over the rounds of that budget, the four arms (five on the full
corpus, with HEAT on residual unions as `HYB-E`) named as below:

```sh
python engine/timing/summarize.py --round r0 PS-M=runs/r0_PS-M PS-H=runs/r0_PS-H HYB-M=runs/r0_HYB-M HYB-H=runs/r0_HYB-H \
    --round r1 ... --compare HYB-H:HYB-M --compare PS-H:PS-M --compare PS-M:HYB-M --compare PS-H:HYB-H \
    --max-pairs 4 --idle-cpus <CPUs kept idle> --disk <device> --out table5_1m_833.json
# full corpus: five arms per round, add --compare HYB-H:HYB-E, --max-pairs 5
```

A cell of Table V is `reads_per_call` / `s_per_step` of its arm in the pairs `HYB-H:HYB-M` and `PS-H:PS-M` (medians over
the valid rounds used, `used`). `median_saving` and `faster` of these two pairs give HyperPage's time saving within each
engine, `time_ratio` of `PS-M:HYB-M` and `PS-H:HYB-H` the page-search over hybrid time ratio, `peak_mb` the memory, and
`median_saving` of `HYB-H:HYB-E` the saving over HEAT. At 1M, 5%, the cell uses four valid rounds on the same four
files; one of them has a read-speed reference of its own and enters as one more `--round` when every read-speed mean
of its check lies within 10% of every mean of the reference of the other rounds:

```sh
python engine/timing/read_check.py <read_speed.py output of that round> --reference <reference of the other rounds> \
    --cross --drift-form reference_over_current
```

## Reported numbers and their commands

Commands are those of the sections above; `summarize.py` stands for `timing/summarize.py` on the rounds of the budget.
Section VI-E states no number; its last sentence is the one-thread check below.

| Paper item | Command | Inputs |
|---|---|---|
| Section VI-E: "With one thread, the engine performs the replay's cache operations in the replay's order" | `checks/check_hybrid.py` (`every_step_equal`, `a1.calls_differing`) | one-thread `hybrid_search` runs, `checks/stage_hybrid.py` |
| "A patched DiskANN reproduces the stock results and expansion sequences of 7,850 W3 sub-queries" | `checks/stage_scoped_sample.py`, `search_and_log`, `batch_search`, `checks/check_scoped.py` (`calls`, check (i)) | 200 training groups of draw 1 and the layouts checked |
| "With one thread, it reads exactly the replayed pages of every 1M validation step under the hybrid cache" | `checks/stage_hybrid.py`, one-thread `hybrid_search`, `checks/check_hybrid.py` (`all_pass`) | W3-1M validation steps 41 to 70 |
| "On the full corpus, near-ties of PQ distances change 44 of 66,375 validation calls" | not included (no script): the differing calls of one-thread runs split into order-only and set differences | |
| "the engine's read savings stay within 0.1 points of the replay" | not included (no script); computed from the engine's reads saved (`summarize.py`, `reads_saved`) and the replay's saving of the same layouts | |
| "DiskANN's node cache spends 1.25 to 1.72 times the declared share on the hot records" | not included (no script); computed from the node-cache memory of the timed runs (`rss_cache_kb` - `rss_loaded_kb` of the engine's summary line, median over MARGO's runs) and the declared share B floor(rho D) records x 8 KiB / 12 | |
| Table IV: pairs, s per step of both arms, reads saved, time saved | `summarize.py` (pair `HP:MARGO`: `n`, `s_per_step`, `reads_saved`, `median_saving`), arms of the Table IV arm table above | test split staged by `checks/stage_hybrid.py --key test --no-reference` |
| "The median paired saving is 9.60% to 13.87% per step at 1M and 2.59% to 10.38% on the full corpus" | Table IV, time saved | |
| "Two pairs were void, one at 25% of 1M and one at 1% of the full corpus" | `summarize.py` (`void`, `why`) and `timing/read_check.py` | |
| "HyperPage is faster in every valid pair, and each budget with 10 pairs gives a sign-test p of 0.002" | `summarize.py` (`faster`, `sign_p`) | |
| "Median sub-query latency falls at every budget, by 2.2% to 16.1%, and p99 falls by 2.1% to 6.3%" | `summarize.py` (`p50_fall`, `p99_fall`); the engine times every sub-query search, and the percentiles are taken over the sub-queries of the run | |
| "12 concurrent sub-queries share a 417-page LRU, and the engine reads 31% to 38% more than the replay" | not included (no script); computed from the engine's page reads and the replayed reads of the same arms (P05-C with 416 and 3,750 hot pages at shares 0.5 and 0.9) | |
| "Replayed in the arrival order that the engine logs with MARGO's layout, MARGO's reads match the engine's within 0.001%" | the replay in the engine's logged arrival order: see `experiments/README.md`; the order comes from the `hybrid_order` build with `HYB_ORDER_LOG` set | |
| "the engine reads 0.64% and 0.38% fewer pages than the replay at 1% and 5%" (full corpus) | not included (no script); computed from the engine's `page_reads` (`summarize.py`) and the replayed test reads of MARGO at its deployed share (`replay/replay_hybrid.py`) | |
| Payback: "within 0.8 to 3.1 later runs in core-seconds ... 9.7 to 37.5 runs"; "45.9 and 6.6 runs in core-seconds and 551 and 79 runs of wall time" | `timing/payback.py summary.json --investment I --rewrite W` with W = 1 s at 1M and 17 s on the full corpus; I (prep + refine seconds of the timed layout, draw 0): 1M 1% 5,714, 5% 2,004.9, 25% 2,584.22; full corpus 1% 6,023, 5% 5,442, 15% 12,036.28, 25% 691.09 + 16 x 77.30 (the 16-thread partitioner step charged per thread) | `summarize.py` output per budget |
| Table V, every cell (page reads per test call / seconds per test step) | `summarize.py` with the four (five) arms and `--compare` pairs of "Cells of Table V" | rounds of `timing/run_arm.sh` |
| Table V caption: "medians of 3, 4 and 2 valid rounds at 1M and 3 at 21M" | `summarize.py` (`used`); the fourth 1M 5% round by `timing/read_check.py --cross` | |
| Table V caption: "Page search takes the smallest L with validation recall@10 of 0.95" (1M: 350, 375, 500 with MARGO's layout, 375, 400, 500 with HyperPage's; full corpus: 600, 600 at 1%, 600, 700 at 5%) | `page_search/recall.py subset` (1M), `recall.py sweep`, `page_search/readout.py select` | validation sweeps |
| Table V caption: "Test recall@10 is 0.946 to 0.954" | `page_search/recall.py recall` on the test runs at the chosen L (1M: mapped through the test schedule; full corpus: `gtq_test.bin`) | |
| Table V caption: "Its 1M hybrid arms at 1% and 5% are those of Table IV"; "At 25% it times P05-C on older copies of the files, where MARGO takes 1.85 s per step instead of 1.79 s" | `summarize.py` of Table V (`HYB-M` at 25%) and of Table IV (`MARGO` at 25%) | |
| "The hybrid engine with HyperPage's layout is the fastest configuration we time" | Tables IV and V, s per step | |
| "a record cache of the whole budget ranked by training frequency, which beats its own cache selection" | `page_search/readout.py select --rule fewest-reads --fewer PS-M-own_<D>,PS-M_<D>` on 1M validation sweeps with W in {1, 2, 4} | `page_search/cache_list.py` |
| "At 1M, HyperPage's page-search layouts read 11.8%, 8.3% and 7.5% fewer pages than MARGO's on every draw" | `page_search/readout.py saving` (`saving.mean`, `saving.all_positive`) | test runs at the chosen L, one per training draw |
| "at test recall within 0.3 points" | difference of the recalls that `readout.py saving` prints (MARGO against the mean over draws) | |
| "With either layout, page search takes 1.06 to 2.19 times as long as the hybrid engine" | `summarize.py` (`time_ratio` of `PS-M:HYB-M` and `PS-H:HYB-H`); the range is taken over the five budgets | |
| "and holds 20% to 33% more memory" | not included (no script); ratios of the peak-memory medians (`peak_mb` of `PS-M:HYB-M` and `PS-H:HYB-H` in `summarize.py`) | |
| "At 1M, HyperPage's layouts take 9.8% to 14.0% less time than MARGO's in the hybrid engine and 1.9% to 10.6% less in page search, in every round" | `summarize.py` (`median_saving` and `faster` of `HYB-H:HYB-M` and `PS-H:PS-M`), range over the three 1M budgets | |
| "On the full corpus, its page-search layout takes 8.2% less time at 1%" | `summarize.py` (`median_saving` of `PS-H:PS-M`, full corpus 1%) | |
| "At 5% it needs L = 700 instead of 600 to reach the recall target, so it reads 11.4% more pages and takes 13.6% more time" | `page_search/readout.py select`; `summarize.py` (`reads_saved` and `median_saving` of `PS-H:PS-M`, full corpus 5%, both negative) | |
| "HyperPage takes 1.44% and 2.62% less time than HEAT on residual unions, the fifth arm on the full corpus" | `summarize.py` (`median_saving` of `HYB-H:HYB-E`) | |
| "At 5% with 64 to 256 concurrent sub-queries, it serves about twice the sub-queries per second of page search with MARGO's layout, at lower p99 latency" | `summarize.py --run NAME=PREFIX` (`qps`, `p99_ms`) on the 64-, 128- and 256-thread runs of `hybrid_aio` and `margo_page_search_aio`; "about twice" is the ratio of the printed `qps` | |
| "MARGO's hybrid runs slowed within each run at this load, so we leave them out" | `summarize.py --run` (`flags`: I/O time per read of the last fifth against the first) | |
