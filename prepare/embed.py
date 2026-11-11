"""bge-small-en-v1.5 embeddings of corpus texts and of the distinct sub-query strings: CLS pooling, L2 normalisation,
stored as float16. Corpus texts can be split into shards that are embedded independently and resumed per chunk."""
import argparse
import gzip
import json
import os
import time

import numpy as np

INSTRUCTION = 'Represent this sentence for searching relevant passages: '
DIM = 384
CHUNK = 65536


class Encoder:
    def __init__(self, model_dir, device, dtype, threads):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        if device == 'cpu':
            torch.set_num_threads(threads)
        dt = {'fp16': torch.float16, 'fp32': torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=dt).to(device).eval()
        self.device = device

    def encode(self, texts, max_len, bs):
        """Embeddings of one chunk of texts; batches are formed in order of token length to limit padding."""
        torch = self.torch
        out = np.zeros((len(texts), DIM), np.float16)
        ids = self.tok(texts, max_length=max_len, truncation=True, padding=False)['input_ids']
        order = np.argsort([len(x) for x in ids], kind='stable')
        for b0 in range(0, len(order), bs):
            idx = order[b0:b0 + bs]
            L = max(len(ids[i]) for i in idx)
            inp = np.zeros((len(idx), L), np.int64)
            att = np.zeros((len(idx), L), np.int64)
            for k, i in enumerate(idx):
                inp[k, :len(ids[i])] = ids[i]
                att[k, :len(ids[i])] = 1
            with torch.no_grad():
                h = self.model(input_ids=torch.from_numpy(inp).to(self.device),
                               attention_mask=torch.from_numpy(att).to(self.device)).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h.float(), dim=-1)
            out[idx] = h.cpu().numpy().astype(np.float16)
        return out


def docs(a, enc):
    off = np.load(os.path.join(a.data_dir, 'offsets.npy'), mmap_mode='r')
    tx = np.memmap(os.path.join(a.data_dir, 'texts.bin'), dtype=np.uint8, mode='r')
    n = len(off) - 1

    def texts(lo, hi):
        return [tx[off[i]:off[i + 1]].tobytes().decode('utf-8', errors='replace') for i in range(lo, hi)]

    bs = a.batch or 128
    t0 = time.time()
    if a.nshards is None:
        # whole corpus in one array; chunks of 65,536 texts from row 0
        emb = np.zeros((n, DIM), np.float16)
        for c in range(0, n, CHUNK):
            e = min(c + CHUNK, n)
            emb[c:e] = enc.encode(texts(c, e), 512, bs)
            print(f'docs {e}/{n} {time.time() - t0:.0f}s', flush=True)
        np.save(os.path.join(a.data_dir, 'doc_emb.f16.npy'), emb)
        return
    # shard r holds rows n*r//k .. n*(r+1)//k as raw float16; chunks of 65,536 texts from the shard start
    r, k = a.shard, a.nshards
    lo, hi = n * r // k, n * (r + 1) // k
    path = os.path.join(a.data_dir, f'doc_s{r}.f16')
    done = os.path.join(a.data_dir, f'doc_s{r}.done')
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.truncate((hi - lo) * DIM * 2)
    out = np.memmap(path, dtype=np.float16, mode='r+', shape=(hi - lo, DIM))
    start = int(open(done).read()) if os.path.exists(done) else lo
    for c in range(start, hi, CHUNK):
        e = min(c + CHUNK, hi)
        out[c - lo:e - lo] = enc.encode(texts(c, e), 512, bs)
        out.flush()
        # the marker holds the next unembedded row, so an interrupted shard resumes at a chunk boundary
        with open(done + '.tmp', 'w') as f:
            f.write(str(e))
        os.replace(done + '.tmp', done)
        print(r, e - lo, hi - lo, f'{time.time() - t0:.0f}s', flush=True)


def queries(a, enc):
    # distinct strings in first-occurrence order of the calls file; the position is the string id used by the splits
    seen = {}
    with gzip.open(os.path.join(a.data_dir, 'calls.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)[-1]
            if q not in seen:
                seen[q] = len(seen)
    qs = list(seen.keys())
    json.dump(qs, open(os.path.join(a.data_dir, 'qstr.json'), 'w'), ensure_ascii=False)
    bs = a.batch or (512 if a.device != 'cpu' else 256)
    texts = [INSTRUCTION + q for q in qs]
    emb = np.zeros((len(texts), DIM), np.float16)
    t0 = time.time()
    for c in range(0, len(texts), CHUNK):
        emb[c:c + CHUNK] = enc.encode(texts[c:c + CHUNK], 128, bs)
        print(f'queries {min(c + CHUNK, len(texts))}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    np.save(os.path.join(a.data_dir, 'q_emb.f16.npy'), emb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['docs', 'queries'])
    ap.add_argument('--data-dir', required=True, help='docs: texts.bin, offsets.npy; queries: calls.jsonl.gz')
    ap.add_argument('--model', required=True, help='local directory of bge-small-en-v1.5')
    ap.add_argument('--device', default='cuda', help="'cuda' or 'cpu'")
    ap.add_argument('--dtype', choices=['fp16', 'fp32'], default=None,
                    help='inference precision (default: fp16 on cuda, fp32 on cpu)')
    ap.add_argument('--batch', type=int, default=None, help='batch size (default: docs 128; queries 512 cuda, 256 cpu)')
    ap.add_argument('--threads', type=int, default=12, help='CPU threads on cpu')
    ap.add_argument('--shard', type=int, default=0, help='docs: shard index')
    ap.add_argument('--nshards', type=int, default=None, help='docs: number of shards (default: no sharding)')
    a = ap.parse_args()
    dtype = a.dtype or ('fp32' if a.device == 'cpu' else 'fp16')
    enc = Encoder(a.model, a.device, dtype, a.threads)
    (docs if a.mode == 'docs' else queries)(a, enc)
    print('DONE', a.mode, flush=True)


if __name__ == '__main__':
    main()
