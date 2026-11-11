# prepare/: from public data to logged hypergraphs

This directory turns the public trajectories, corpora and vectors into the files that layout construction and
replay read. It covers every workload of Table II (W3-1M, W3-21M, W2, W1, W4 proxy, DEEP1M) and the 5M and 10M
corpus samples of Fig. 4. No third-party data, model or code is included. Section 1 lists where to get each item.

All scripts take their inputs and outputs as arguments (`--help` lists them). They import the shared library as
`from hyperpage import formats, hypergraph` (`pip install -e .` at the repository root). Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `RAW` | downloaded third-party files (Section 1) |
| `MODEL_DIR` | local copy of bge-small-en-v1.5 |
| `MARGO` | build directory of MARGO (`MARGO/tests/my_build_disk_index`, `MARGO/my_gp/new_mincut`) |
| `APP` | the logging search application `search_and_log` built by `engine/build.sh`; command line `search_and_log <float\|int8> l2 <index_prefix> <query.bin> <L> <W> <K> <threads> <out_prefix> 0` |
| `DATA_DIR/<w>` | per-workload data: texts, embeddings, int8 vectors, calls, splits, ground truth |
| `INDEX_DIR/<w>` | disk index files, prefix `INDEX_DIR/<w>/margo_R64_L125` |
| `LAYOUT_DIR/<w>`, `LOG_DIR/<w>`, `HYPER_DIR/<w>` | layout files, traversal logs, hypergraph files |
| `T` | number of threads |

Workload directory names: `w1`, `w2`, `w3` (W3-1M), `w4`, `w21` (W3-21M), `w5m`, `w10m`, `deep`.

## 1. Third-party data, model and code

| Item | Address | Pinned revision / checksum | Licence | Use |
|---|---|---|---|---|
| Search-R1 / Qwen3-8B rollouts: outcome-reward run `rollouts/search_r1_qwen3_8b_4gpu/{1..400}.jsonl`, process-reward run `rollouts/search_r1_qwen3_8b_perturnnorm/{1..400}.jsonl`; OLMo-3-7B samples `eval/search_r1_metrics/eval_step{330,340,...,400}_olmo3_metrics_samples.jsonl` | https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts | revision `352fd3111eb3647f5dca07126830f800af01d15f` | Apache-2.0 | W2, W3, W4 sub-queries |
| OpenResearcher-Dataset, `seed_{42..57}/train-0000{0,1,2}-of-00003.parquet` | https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset | revision `1edca18c5d7c9c105762da4f3c5c2b83855b3c6e` | MIT | W1 sub-queries |
| OpenResearcher-Corpus, `data/train-0000{0..6}-of-00098.parquet` (docid 0..999,999 lie in these shards) | https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Corpus | revision `8367703babe6e080899a48782f365c284c32b668`; sha256 of shards 0-6 below | ODC-BY | W1 corpus |
| DPR Wikipedia passages `psgs_w100.tsv.gz` (21,015,324 passages) | https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz | sha256 `c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d` | CC-BY-NC 4.0 (DPR repository) | corpus of W2, W3, W4, 5M, 10M, 21M |
| bge-small-en-v1.5 | https://huggingface.co/BAAI/bge-small-en-v1.5 | revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; `model.safetensors` sha256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad` | MIT | all embeddings |
| DEEP `base.10M.fbin` | https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin | md5 of bytes 0-396,288,007 (rows 0-1,031,999): `a21078e6a8f69f2b5262e2a85039192b` | none stated at the download address | DEEP1M base and queries |
| MARGO (index builder, layout optimiser) | https://github.com/CodenameYZY/MARGO | commit `7e24023b950c1a31d993476a40f444dd69617ba7` | **the repository states no licence**: obtain it from its authors' repository yourself; nothing of it is redistributed here | Vamana index, MARGO edge weights and layout |
| DiskANN (branch `cpp_main`) | https://github.com/microsoft/DiskANN | commit `78256bbab4685e1774e78d331e081a153be26823` | MIT | logging search application (built in `engine/`) |

sha256 of the OpenResearcher-Corpus shards:

```
1e46b50b1bd8de7318ce237f2a16d73172af8c6afb6a97da12abbf245af5c511  train-00000-of-00098.parquet
4535c3738a2da2aa62b7cf9223f770f0e8ab16f04bbb15718078e7093a30ec91  train-00001-of-00098.parquet
4539b7007cf8b9adf9b40969bef9c571dc3cb2c438122882a8a1c1789f309288  train-00002-of-00098.parquet
cf42673c80c2e61483299241fd0afba0bea1c86180d0dd18e84ee9c5206adf6c  train-00003-of-00098.parquet
959ed2a9bf5708e5be66366d1997446a2e45e73f36f10eeaff57f4c0d3698bc6  train-00004-of-00098.parquet
3bb0916843f8101b92cf896295e1de6e7f4484a88949b6c7f2ebf4be1176a055  train-00005-of-00098.parquet
6782be4d6a140b3f4f72525486437ae4dd02952bfe15ca4ed1b581d68de7a7a3  train-00006-of-00098.parquet
```

Download commands (all public, no login):

```sh
HF=https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts/resolve/352fd3111eb3647f5dca07126830f800af01d15f
mkdir -p RAW/outcome RAW/process RAW/olmo
for s in $(seq 1 400); do
  curl -sL -o RAW/outcome/$s.jsonl $HF/rollouts/search_r1_qwen3_8b_4gpu/$s.jsonl
  curl -sL -o RAW/process/$s.jsonl $HF/rollouts/search_r1_qwen3_8b_perturnnorm/$s.jsonl
done
for c in 330 340 350 360 370 380 390 400; do f=eval_step${c}_olmo3_metrics_samples.jsonl
  curl -sL -o RAW/olmo/$f $HF/eval/search_r1_metrics/$f; done

OR=https://huggingface.co/datasets/OpenResearcher
for s in $(seq 42 57); do for p in 0 1 2; do f=seed_$s/train-0000$p-of-00003.parquet; mkdir -p RAW/traj/seed_$s
  curl -sL -o RAW/traj/$f $OR/OpenResearcher-Dataset/resolve/1edca18c5d7c9c105762da4f3c5c2b83855b3c6e/$f; done; done
mkdir -p RAW/or_corpus; for p in 0 1 2 3 4 5 6; do f=train-0000$p-of-00098.parquet
  curl -sL -o RAW/or_corpus/$f $OR/OpenResearcher-Corpus/resolve/8367703babe6e080899a48782f365c284c32b668/data/$f; done

curl -L -o RAW/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
sha256sum RAW/psgs_w100.tsv.gz   # c39b020c...53ea8d

B=https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt \
         modules.json 1_Pooling/config.json config_sentence_transformers.json; do
  mkdir -p MODEL_DIR/$(dirname $f); curl -sL -o MODEL_DIR/$f $B/$f; done

# DEEP: rows 0..1,031,999 of base.10M.fbin (header + rows, by an HTTP range request)
curl -r 0-396288007 -o RAW/deep_base_head.fbin \
  https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin
```

`deep_base_head.fbin` keeps the 8-byte header of the full file. `deep_prepare.py` reads only the rows it needs.

**MARGO build.** Clone the repository at the commit above. In its top-level `CMakeLists.txt`, disable the
subdirectory `graph_partition` (comment out the statement that adds it); MARGO's own partitioner `my_gp` does not
need that subdirectory. Configure a
Release build with MKL, OpenMP and Boost, and build the targets `my_build_disk_index` and `new_mincut`:

```sh
git clone https://github.com/CodenameYZY/MARGO.git && cd MARGO && git checkout 7e24023b950c1a31d993476a40f444dd69617ba7
# edit CMakeLists.txt as described above
mkdir -p release && cd release && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j my_build_disk_index new_mincut
```

## 2. Steps per workload

Each step is one command. Run the steps in the order given. Steps that a workload shares with another are named
once.

**The int8 vectors depend on the encoder path**, so use the setting given for each workload:
- 1M DPR corpus, W1 corpus, W1 and W2 sub-queries: GPU, fp16 (`--device cuda`).
- W3 and W4 sub-queries: CPU, fp32 (`--device cpu`). This covers the outcome-run training steps of W3 and all of
  W4. The same string can get different int8 vectors on the two paths, so W3 and W4 use their own query files, never
  W2's.
- Full 21M corpus: GPU, fp32 (`--device cuda --dtype fp32`, 4 shards), stored as fp16. The 5M and 10M samples
  reuse these records.

### W2 (also builds the 1M corpus, index and MARGO layout of W3-1M and W4)

```sh
python prepare/extract_calls.py w2 --outcome-run RAW/outcome --out DATA_DIR/w2
python prepare/prepare_corpus.py dpr --src RAW/psgs_w100.tsv.gz --out DATA_DIR/w2 --n 1000000
python prepare/embed.py docs    --data-dir DATA_DIR/w2 --model MODEL_DIR --device cuda
python prepare/embed.py queries --data-dir DATA_DIR/w2 --model MODEL_DIR --device cuda
python prepare/quantize.py corpus --data-dir DATA_DIR/w2
python prepare/build_index.py --builder MARGO/tests/my_build_disk_index --data DATA_DIR/w2/base.bin \
    --index-prefix INDEX_DIR/w2/margo_R64_L125 --pq-budget-gb 0.03 --build-mem-gb 32 --threads 12
python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/w2/margo_R64_L125 \
    --data-f32 DATA_DIR/w2/base_f32.fbin --out LAYOUT_DIR/w2 --threads 12
python prepare/make_splits.py w2 --data-dir DATA_DIR/w2
python prepare/ground_truth.py --data-dir DATA_DIR/w2 --backend torch --device cuda
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w2/margo_R64_L125 --data-dir DATA_DIR/w2 \
    --out-dir LOG_DIR/w2 --threads 12
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w2 --log-dir LOG_DIR/w2 --out HYPER_DIR/w2
```

`tune_and_log.py` takes the smallest L of its grid with validation recall@10 of at least 0.95 and stops with an error
if no L of the grid reaches it; a fixed L (`--fixed-L`) skips the sweep.

`quantize.py corpus` sets gamma = 127 / Q_0.9999(|x|) over the corpus. W2's gamma (`DATA_DIR/w2/quant.json`) is
reused for W3, W4, W3-21M and the 5M and 10M samples.

### W1

```sh
python prepare/extract_calls.py w1 --trajectories RAW/traj --out DATA_DIR/w1
python prepare/prepare_corpus.py openresearcher --src RAW/or_corpus --out DATA_DIR/w1
python prepare/embed.py docs    --data-dir DATA_DIR/w1 --model MODEL_DIR --device cuda
python prepare/embed.py queries --data-dir DATA_DIR/w1 --model MODEL_DIR --device cuda
python prepare/quantize.py corpus --data-dir DATA_DIR/w1
python prepare/build_index.py --builder MARGO/tests/my_build_disk_index --data DATA_DIR/w1/base.bin \
    --index-prefix INDEX_DIR/w1/margo_R64_L125 --pq-budget-gb 0.03 --build-mem-gb 32 --threads 12
python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/w1/margo_R64_L125 \
    --data-f32 DATA_DIR/w1/base_f32.fbin --out LAYOUT_DIR/w1 --threads 12
python prepare/make_splits.py w1 --data-dir DATA_DIR/w1
python prepare/ground_truth.py --data-dir DATA_DIR/w1 --backend torch --device cuda
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w1/margo_R64_L125 --data-dir DATA_DIR/w1 \
    --out-dir LOG_DIR/w1 --threads 12
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w1 --log-dir LOG_DIR/w1 --out HYPER_DIR/w1
```

W1 uses its own corpus gamma.

### W3-1M (after W2)

```sh
python prepare/extract_calls.py w3 --outcome-run RAW/outcome --process-run RAW/process --out DATA_DIR/w3
python prepare/embed.py queries --data-dir DATA_DIR/w3 --model MODEL_DIR --device cpu --threads 12
python prepare/quantize.py queries --data-dir DATA_DIR/w3 --gamma-from DATA_DIR/w2/quant.json
ln -s "$(realpath DATA_DIR/w2/base.bin)" DATA_DIR/w3/base.bin
mkdir -p LAYOUT_DIR/w3 && cp LAYOUT_DIR/w2/MARGO_B12.npy LAYOUT_DIR/w2/MARGO_B12.json LAYOUT_DIR/w2/ID_B12.npy LAYOUT_DIR/w3/
python prepare/make_splits.py w3 --data-dir DATA_DIR/w3
python prepare/ground_truth.py --data-dir DATA_DIR/w3 --backend torch --device cpu --threads 12
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w2/margo_R64_L125 --data-dir DATA_DIR/w3 \
    --out-dir LOG_DIR/w3 --threads 12
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w3 --log-dir LOG_DIR/w3 --out HYPER_DIR/w3
```

W3-1M uses W2's index, int8 records and layouts. `extract_calls.py w3` also writes `checks.json` (prompt alignment
of the two runs per step) and `drop.json` (held-out groups whose prompt occurs in a training step).

### W4 proxy (after W2)

```sh
python prepare/extract_calls.py w4 --outcome-run RAW/outcome --olmo RAW/olmo --out DATA_DIR/w4
python prepare/embed.py queries --data-dir DATA_DIR/w4 --model MODEL_DIR --device cpu --threads 12
python prepare/quantize.py queries --data-dir DATA_DIR/w4 --gamma-from DATA_DIR/w2/quant.json
ln -s "$(realpath DATA_DIR/w2/base.bin)" DATA_DIR/w4/base.bin
mkdir -p LAYOUT_DIR/w4 && cp LAYOUT_DIR/w2/MARGO_B12.npy LAYOUT_DIR/w2/MARGO_B12.json LAYOUT_DIR/w2/ID_B12.npy LAYOUT_DIR/w4/
python prepare/make_splits.py w4 --data-dir DATA_DIR/w4
python prepare/ground_truth.py --data-dir DATA_DIR/w4 --backend torch --device cpu --threads 12
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w2/margo_R64_L125 --data-dir DATA_DIR/w4 \
    --out-dir LOG_DIR/w4 --fixed-L 500 --threads 5
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w4 --log-dir LOG_DIR/w4 --out HYPER_DIR/w4
```

W4 is logged at W2's L = 500.

### W3-21M (after W3-1M)

```sh
python prepare/prepare_corpus.py dpr --src RAW/psgs_w100.tsv.gz --out DATA_DIR/w21
for r in 0 1 2 3; do   # four independent shards; they can run in parallel, one per GPU
  python prepare/embed.py docs --data-dir DATA_DIR/w21 --model MODEL_DIR --device cuda --dtype fp32 \
      --shard $r --nshards 4
done
python prepare/quantize.py corpus --data-dir DATA_DIR/w21 --nshards 4 --gamma-from DATA_DIR/w2/quant.json
python prepare/build_index.py --builder MARGO/tests/my_build_disk_index --data DATA_DIR/w21/base.bin \
    --index-prefix INDEX_DIR/w21/margo_R64_L125 --pq-budget-gb 0.64 --build-mem-gb 64 --threads T
python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/w21/margo_R64_L125 \
    --data-f32 DATA_DIR/w21/base_f32.fbin --out LAYOUT_DIR/w21 --threads T \
    --expect-n 21015324 --expect-record-bytes 644 --expect-pq-bytes 32
# query-side files: see "Query files of the larger indexes" below
python prepare/ground_truth.py --data-dir DATA_DIR/w21 --backend scan --threads T --no-sample
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w21/margo_R64_L125 --data-dir DATA_DIR/w21 \
    --out-dir LOG_DIR/w21 --threads T
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w21 --log-dir LOG_DIR/w21 --out HYPER_DIR/w21 \
    --keys train0,train1,train2,val,test,s1_40,s201_240
```

The `torch` ground-truth backend needs n < 2^20, so the larger indexes use the `scan` backend. Both give exact
squared L2 with ties broken by the smaller identifier.

### 5M and 10M samples (Fig. 4; after W3-21M)

```sh
python prepare/quantize.py sample --full-dir DATA_DIR/w21 --data-dir DATA_DIR/w5m --n 5000000
# query-side files: see below
python prepare/build_index.py --builder MARGO/tests/my_build_disk_index --data DATA_DIR/w5m/base.bin \
    --index-prefix INDEX_DIR/w5m/margo_R64_L125 --pq-budget-gb 0.15 --build-mem-gb 64 --threads T
python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/w5m/margo_R64_L125 \
    --data-f32 DATA_DIR/w5m/base_f32.fbin --out LAYOUT_DIR/w5m --threads T \
    --expect-n 5000000 --expect-record-bytes 644 --expect-pq-bytes 32
python prepare/ground_truth.py --data-dir DATA_DIR/w5m --backend scan --threads T --no-sample
python prepare/tune_and_log.py --app APP --index-prefix INDEX_DIR/w5m/margo_R64_L125 --data-dir DATA_DIR/w5m \
    --out-dir LOG_DIR/w5m --threads T
python prepare/build_hypergraphs.py --data-dir DATA_DIR/w5m --log-dir LOG_DIR/w5m --out HYPER_DIR/w5m \
    --keys train0,train1,train2,val,test
```

For 10M, use `w10m`, `--n 10000000` and `--pq-budget-gb 0.30`. The sample takes the passage rows
`sort(default_rng(seed).choice(21015324, n))` and the node order `default_rng(seed).permutation(n)`, as the 1M
sample does. Its int8 records are gathered from the 21M records, so nothing is re-embedded.

**Query files of the larger indexes (no script).** W3-21M, 5M and 10M replay the W3 sub-queries. Their
`DATA_DIR/<w>` must hold the query-side files of `DATA_DIR/w3`, unchanged:
- `qx8.npy`, `calls.jsonl.gz`, `drop.json`;
- `splits.npz`, `splits_meta.json`, `meta_<key>.npy`, `queries_needed.npy`;
- `gtq_val.bin`, `gtq_test.bin`, `gtq_val_strid.npy`, `gtq_test_strid.npy`.

Copy them, or link them with absolute targets, for example

```sh
for w in w21 w5m w10m; do
  for f in qx8.npy calls.jsonl.gz drop.json splits.npz splits_meta.json queries_needed.npy gtq_val.bin gtq_test.bin \
           gtq_val_strid.npy gtq_test_strid.npy DATA_DIR/w3/meta_*.npy; do
    f=$(basename $f); ln -s "$(realpath DATA_DIR/w3/$f)" DATA_DIR/$w/$f
  done
done
```

The full-corpus logging run traverses every needed W3 string, including the step windows
`s1_40` and `s201_240`.

### DEEP1M (after W3-1M, whose split sizes it matches)

The held-out query rows are rows 1,032,000-2,599,999 of DEEP `base.10M.fbin`. Store them as one headerless
float32 file `RAW/deep_heldout.raw` of shape 1,568,000 x 96, where held-out row k is base row 1,032,000 + k. They
can be read with an HTTP range request of the same file. Our scripts do not download them. They use held-out rows
0-1,199,999 (training pool, three blocks of 400,000), 1,268,000-1,287,999 (validation) and
1,368,000-1,567,999 (test).

```sh
python prepare/deep_prepare.py --head RAW/deep_base_head.fbin --out DATA_DIR/deep
python prepare/build_index.py --builder MARGO/tests/my_build_disk_index --data-type float \
    --data DATA_DIR/deep/base.fbin --index-prefix INDEX_DIR/deep/margo_R64_L125 \
    --pq-budget-gb 0.03 --build-mem-gb 32 --threads 14
python prepare/margo_layout.py --partitioner MARGO/my_gp/new_mincut --index-prefix INDEX_DIR/deep/margo_R64_L125 \
    --data-f32 DATA_DIR/deep/base.fbin --out LAYOUT_DIR/deep --threads 4
python prepare/tune_and_log.py --app APP --data-type float --index-prefix INDEX_DIR/deep/margo_R64_L125 \
    --data-dir DATA_DIR/deep --out-dir LOG_DIR/deep --skip-test \
    --queries RAW/deep_heldout.raw --raw-dim 96 --chunk-size 0 --threads 12
python prepare/make_splits.py deep --data-dir DATA_DIR/deep --match DATA_DIR/w3/splits.npz
python prepare/build_hypergraphs.py --data-dir DATA_DIR/deep --log-dir LOG_DIR/deep --out HYPER_DIR/deep
python prepare/deep_recall.py --data-dir DATA_DIR/deep --heldout RAW/deep_heldout.raw --log-dir LOG_DIR/deep \
    --out LOG_DIR/deep/recall.json
```

DEEP records are float32 with 96 dimensions: 384 + 4 x 65 = 644 bytes, so B = 12 at 8 KB as for the text
workloads. L is chosen on base rows 1,030,000-1,031,999 against their exact top-100. Recall@10 in Table II comes
from `deep_recall.py`: 5,000 sampled test calls against the exact float64 top-10.

### Table II

```sh
python prepare/table2.py \
  --row W3-1M  HYPER_DIR/w3  LOG_DIR/w3/tune_log_summary.json \
  --row W3-21M HYPER_DIR/w21 LOG_DIR/w21/tune_log_summary.json \
  --row W2     HYPER_DIR/w2  LOG_DIR/w2/tune_log_summary.json \
  --row W1     HYPER_DIR/w1  LOG_DIR/w1/tune_log_summary.json \
  --row W4     HYPER_DIR/w4  LOG_DIR/w4/tune_log_summary.json \
  --row DEEP1M HYPER_DIR/deep LOG_DIR/deep/tune_log_summary.json LOG_DIR/deep/recall.json
```

Units, sub-queries, |e_q| and |E_b| are the number of test scopes, the number of test sub-queries and the mean
set sizes of `test_Q.npz` and `test_U.npz`. Overlap is `1 - nnz(test_U) / nnz(test_Q)`. L and recall@10 come from
the logging summary (DEEP1M: recall from `deep_recall.py`).

## 3. Produced files

Binary vector files (`base.bin`, `base_f32.fbin`, `base.fbin`, `gtq_*.bin`) use the DiskANN layout: int32 rows,
int32 dims, then row-major values (int8 or float32).

`DATA_DIR/<w>/`

| File | Content |
|---|---|
| `calls.jsonl.gz` | one JSON row per sub-query, in extraction order; last field = query string. W1: `[seed, qid, status, call_idx, query]`. W2: `[step, prompt_sha1_16, rollout_line, tool_call, list_pos, query]`. W3, W4: `[run, step_or_checkpoint, prompt_key, rollout_line_or_question_index, tool_call, list_pos, query]` with run `"4gpu"` (outcome run), `"perturnnorm"` (process run) or `"olmo3"`, and prompt_key = sha1[:16] of the normalised question (lower case, collapsed whitespace, trailing `?` removed) |
| `checks.json`, `drop.json` | W3, W4: prompt checks; `drop_groups` (W3, `[step, prompt_key]` of the process run) or `drop_index` (W4, question indices) removed from the splits |
| `texts.bin`, `offsets.npy` | corpus texts, UTF-8 concatenated; int64 offsets (n + 1). DPR text = title + newline + text |
| `sel_rows.npy` | 1M sample: sorted DPR passage rows of the text rows |
| `doc_emb.f16.npy` or `doc_s<r>.f16` + `doc_s<r>.done` | corpus embeddings (n x 384 float16), one array or raw shards |
| `qstr.json` | distinct query strings in first-occurrence order of `calls.jsonl.gz`; position = **string id** |
| `q_emb.f16.npy` | query embeddings, rows = string ids |
| `quant.json` | gamma and clipping shares |
| `perm.npy` | node order: node j is text row `perm[j]` (int64) |
| `sel.npy` | 5M/10M: sorted DPR passage rows of the sample; node j is passage `sel[perm[j]]` |
| `base.bin`, `base_f32.fbin` | int8 records in node order; the same values as float32 for MARGO's partitioner |
| `qx8.npy` | int8 query vectors, rows = string ids |
| `splits.npz` | per split key `<key>_ptr` (int64, scope boundaries: scope i holds positions `ptr[i]:ptr[i+1]`), `<key>_q` (int64, string id of every sub-query, in replay order), `<key>_unit` (scope unit id: question id for W1, scope index otherwise) |
| `meta_<key>.npy` | per scope: W1 `(seed, qid)`; W2 `(step, prompt hash)`; W3 `(run, step, prompt hash)` with run 0 = outcome, 1 = process; W4 held-out `meta_val/test.npy` = question indices |
| `queries_needed.npy` | sorted string ids used by any split (the strings that are logged) |
| `splits_meta.json` | scope and sub-query counts per key |
| `gtq_<split>.bin`, `gtq_<split>_strid.npy`, `gt_<split>.npy` | 5,000 sampled validation and test calls (int8 vectors, string ids) and their exact top-10 identifiers (int64) |

Split keys: `train0`, `train1`, `train2` (training draws), `val`, `test`. W3 also has `s1_40` and `s201_240`
(process-run steps 1-40 and 201-240) and the shuffled control `<key>_shuf` for the draws, `val` and `test`. In the
shuffled control, the sub-queries of each step are permuted across that step's groups and cut back into blocks of
the true group sizes.

DEEP1M (`DATA_DIR/deep`): `base.fbin` (1M x 96 float32), `gtq_val.bin` and `gt_val.npy` (validation queries
for L and their exact top-100), `queries_needed.npy` (held-out row ids), and `splits.npz` with the same keys as
above, where `<key>_q` holds held-out row ids.

`LAYOUT_DIR/<w>/`: `MARGO_B12.npy` and `ID_B12.npy`, a **layout file** each: int32 array of length n, page of
every record. `ID_B12` packs identifiers in order, page = id // 12. `MARGO_B12.json` holds MARGO's page count and
fill.

`LOG_DIR/<w>/`: `all_c<k>.log.bin` and `all_c<k>.res.bin`, one traversal log per query file of `chunk_size`
needed strings, in the order of `queries_needed.npy`. The formats are those of the logging search application:
`hyperpage.formats.read_log`, `read_expanded_sets` and `read_results` read them. `tune_log_summary.json` holds
`L`, `sweep`, `test_sample.recall10`, `chunk_size`, `n_logged` and the index header `meta`.

`HYPER_DIR/<w>/`:

| File | Arrays |
|---|---|
| `<key>_Q.npz` | per-query hypergraph: `e_ptr` (int64), `e_nodes` (int32), one set e_q per sub-query in split order; `batch_ptr` = the split's `<key>_ptr` |
| `<key>_U.npz` | scope hypergraph: `e_ptr`, `e_nodes`, one union E per scope, records in first-occurrence order (Definition 8) |
| `hyper_info.json` | per key: scopes, sub-queries, nnz of both files, mean sizes |

Shuffled keys have only `_U` files, because their sub-queries are those of the source split.

## 4. Software versions of the reported runs

- Embedding and torch ground truth: torch 2.8.0 (CUDA 12.8 build), transformers 4.57.6.
- 21M corpus embedding: torch 1.9 with transformers 4.30.2. This pair does not read `model.safetensors`; it loads the
  same weights from `MODEL_DIR/pytorch_model.bin`, which a recent torch with the `safetensors` package writes:
  `python -c "import torch; from safetensors.torch import load_file; torch.save(load_file('MODEL_DIR/model.safetensors'), 'MODEL_DIR/pytorch_model.bin')"`.
- CPU scripts of this directory: Python 3.11, numpy 2.4.6, numba 0.68.0, scipy 1.17.1. pyarrow is needed for W1 extraction and the
  OpenResearcher corpus.
- Native builds: gcc 11.4.0, cmake 3.25.0, Boost 1.92.0, MKL 2026.1.0, libaio 0.3.113.
