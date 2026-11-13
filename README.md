# HyperPage

Code for the paper "HyperPage: Learned Page Layouts for Disk-Resident Graph Vector Indexes"

A disk-resident graph index such as DiskANN stores each vector with its graph edges as one record on SSD pages, so
the page reads of a search depend on which records share a page. HyperPage uses the logged graph traversals of a
finished run of a search agent to lay out the index for a declared memory budget, so that later runs read fewer
pages. It builds layouts on the logged traversals of the uncached records (Algorithm 1) and selects one layout,
with its cache split, by replaying validation steps.

## Contents

| Directory | Content | Paper |
|---|---|---|
| `hyperpage/` | library: hypergraphs, objectives, refinement (Algorithm 1), hybrid-cache replay, selection | Sections III to VI |
| `prepare/` | from the public data to the index, the traversal logs and the hypergraph files | Section VII-A, Table II |
| `layouts/` | layout construction: Algorithm 1 and the compared layouts | Sections VI-B to VI-D, VII-A |
| `replay/` | replay under the hybrid cache and under read-once scopes, selection, statistics | Sections VI-D, VII-B, VII-C |
| `bounds/` | lower bounds and small instances solved to optimality | Section VII-D |
| `engine/` | edits to DiskANN, the relayout writer, engine checks and timing | Section VI-E, Tables IV and V |
| `experiments/` | one script per table and figure, and the map from reported numbers to commands | Section VII |
| `tests/` | an end-to-end check on a synthetic workload | |

The repository holds code only. It contains no third-party data, model or source tree, and none of our traversal
logs, layouts or result files. `prepare/README.md` lists the third-party datasets, the model and the code bases with
their addresses, pinned revisions, licences and download commands, and `engine/README.md` does the same for the code
the engine builds on. The engine is a set of edit scripts for a DiskANN checkout. The scripts in `engine/patches/`
quote short excerpts of DiskANN, which is MIT-licensed, to locate their edits and inside the texts that replace them
(`engine/patches/THIRD_PARTY_NOTICE.md`).

## Installation

Python 3.9 or later.

```sh
pip install -e .                 # numpy, numba, scipy
python tests/test_synthetic.py   # about one minute, needs no data
```

Some steps need further packages, named in the README of their directory: `torch` and `transformers` (text
embedding), `pyarrow` (trajectory files), `matplotlib` (figures), `ortools` and `networkx` (small instances),
`mtkahypar` (the Mt-KaHyPar layouts). The engine is C++ and builds against a DiskANN checkout (`engine/README.md`).

## Pipeline

Each step has a README with the exact commands and file formats.

1. **Prepare** (`prepare/README.md`): download the public data, extract the search calls, embed and quantize, build
   the index and MARGO's layout, log the traversals of every sub-query with the logging application of `engine/`,
   and build the hypergraph files (per-query sets and group unions of every split).
2. **Build layouts** (`layouts/README.md`): `layouts/refine_layout.py` refines the incumbent layout under one
   objective, cache-blind or for the record cache of one budget. The README maps every layout label of the paper to
   its command.
3. **Replay and select** (`experiments/README.md`): `replay/make_schedule.py` writes the arrival order of a split,
   `replay/replay_hybrid.py` replays layouts under the hybrid cache, `replay/select_deployment.py` chooses the
   hot-record share of every layout and HyperPage's deployed pair on validation steps, and `replay/saving_table.py`
   reports the savings on test steps.
4. **Bounds** (`bounds/README.md`) and **engine** (`engine/README.md`) are independent of step 3.

## Paper to code

| Paper | Code |
|---|---|
| Definition 2 (layout, page counts, connectivity) | `hyperpage/objectives.py: connectivity_and_clique` |
| Definition 7 (memory budget, hybrid cache) | `hyperpage/cache.py: split_budget, cached_mask, hybrid_cache_misses` |
| Definition 8, Eq. (1) (scope hypergraph, F_conn) | `hyperpage/hypergraph.py: scope_unions, per_query_sets` (identical unions stay separate sets of weight 1); `hyperpage/objectives.py` |
| Definition 9 (pairwise objectives) | `hyperpage/weights.py: arc_pairs`; `hyperpage/objectives.py: pairwise_value` |
| Corollary 1 (reads of a scoped buffer) | `replay/scope_reads.py` |
| Proposition 1 (set credit plus concentration bonus) | checked at the end of `hyperpage/layout.py: refine_layout` |
| Propositions 6 to 8 (capacity bound, lower bound, window bound) | `bounds/` |
| Algorithm 1, Eq. (4) and (5), Proposition 9 | `hyperpage/refine.py: refine` |
| Section VI-C (candidate pages, strict descent) | `hyperpage/refine.py: refine`; `hyperpage/layout.py: refine_layout` |
| Corollary 3, Section VI-D (cache-aware refinement) | `hyperpage/hypergraph.py: residual_sets`; `hyperpage/layout.py: refine_layout` |
| Section VI-D (selection by replay) | `hyperpage/selection.py: deployment`; `replay/select_deployment.py` |
| Section VI-E (replay engine) | `engine/` |
| Section VII-A (metrics, statistics) | `hyperpage/selection.py`; `replay/significance.py` |
| Tables II to VII, Figs. 3 and 4 | `experiments/README.md` |

Comments of the form `# [Paper] ...` mark the places that implement a definition, equation, algorithm line or
reported result.

## Reproducibility notes

- Refinement is deterministic: one thread, integer gains, and a visiting order fixed by `--seed` (the index of the
  training draw). The replay is deterministic for a given arrival order.
- The default seeds of the scripts are those of the reported runs. They fix the corpus sample, the record
  numbering, the splits and the resampling of the tests, so another seed gives other samples and splits.
- The budgets of the paper are stated in pages in `layouts/README.md` and `experiments/`: 833, 4,167 and 20,834
  pages for the 1M index, and 17,513, 87,564, 262,144 and 437,820 pages for the full corpus (the third is the
  2 GiB limit, about 15% of that index).
- Mt-KaHyPar runs with several threads, so its layouts are not reproducible bit for bit.
- Search times depend on the host and its SSD. `engine/README.md` describes the protocol of the timed comparisons.
- The tables of reported numbers in `experiments/README.md`, `engine/README.md` and `bounds/README.md` give the
  command for each number and mark the numbers that have no script in this repository.

## Licence

The code in this repository is released under the MIT licence (`LICENSE`). Third-party data, models and code keep
their own licences; `prepare/README.md` and `engine/README.md` list them.

## Reference
```
@misc{zhang2026hyperpagelearnedpagelayouts,
      title={HyperPage: Learned Page Layouts for Disk-Resident Graph Vector Indexes}, 
      author={Xinyu Zhang and Jingze Li and Baihua Zheng and Weiwei Sun},
      year={2026},
      eprint={2611.16149},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2611.16149}, 
}
```
