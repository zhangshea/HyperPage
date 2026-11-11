"""Replay of logged sub-queries under the hybrid cache: a static record cache plus one shared page LRU."""
import numpy as np
import numba as nb

SHARE_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@nb.njit(cache=True)
def hybrid_cache_misses(arrival, e_ptr, e_nodes, page, npages, lru_pages, cached, step_of, n_steps):
    """Page reads of the hybrid cache per step.

    Sub-queries arrive in the order of arrival (indices into the sets e_ptr, e_nodes) and are replayed record by record
    in expansion order. An access to a cached record reads nothing. Every other access goes through one LRU of
    lru_pages pages that starts empty and is kept across steps; a miss is one page read, charged to the step of its
    sub-query. With lru_pages = 0 every uncached access is a read."""
    misses = np.zeros(n_steps, np.int64)
    if lru_pages == 0:
        for q in arrival:
            for j in range(e_ptr[q], e_ptr[q + 1]):
                if not cached[e_nodes[j]]:
                    misses[step_of[q]] += 1
        return misses
    # [Paper] Definition 7: the page part of the hybrid cache is one LRU shared by all sub-queries;
    # prev and next_ link the cached pages from the most to the least recently used
    prev = np.full(npages, -1, np.int32); next_ = np.full(npages, -1, np.int32); in_lru = np.zeros(npages, np.bool_)
    head = -1; tail = -1; size = 0
    for q in arrival:
        s = step_of[q]
        for j in range(e_ptr[q], e_ptr[q + 1]):
            v = e_nodes[j]
            if cached[v]:
                continue
            pg = page[v]
            if in_lru[pg]:
                if pg != head:
                    a = prev[pg]; b = next_[pg]; next_[a] = b
                    if b >= 0:
                        prev[b] = a
                    else:
                        tail = a
                    prev[pg] = -1; next_[pg] = head; prev[head] = pg; head = pg
            else:
                misses[s] += 1
                if size == lru_pages:
                    t = tail; tail = prev[t]; in_lru[t] = False; prev[t] = -1
                    if tail >= 0:
                        next_[tail] = -1
                    else:
                        head = -1
                else:
                    size += 1
                in_lru[pg] = True; prev[pg] = -1; next_[pg] = head
                if head >= 0:
                    prev[head] = pg
                else:
                    tail = pg
                head = pg
    return misses


@nb.njit(cache=True)
def record_frequency(e_ptr, e_nodes, n):
    """Number of sets that contain each record."""
    f = np.zeros(n, np.int64)
    for j in range(len(e_nodes)):
        f[e_nodes[j]] += 1
    return f


def rank_records(frequency):
    """Records in the order the record cache admits them: most frequent first, ties to the smaller identifier."""
    return np.lexsort((np.arange(len(frequency)), -frequency))


def split_budget(budget_pages, share, capacity, record_scale=1.0):
    """Split a budget of D page slots at hot-record share rho.

    Returns (hot_pages, cached_records, lru_pages) with hot_pages = floor(rho * D) and lru_pages = D - hot_pages. The
    record cache holds capacity * hot_pages records; record_scale > 1 charges every cached record that multiple of
    its on-disk size, so the same slots hold fewer records.
    """
    # [Paper] Definition 7: floor(rho * D) page slots hold the B * floor(rho * D) most frequent training records
    hot_pages = int(np.floor(share * budget_pages))
    cached_records = int(capacity * hot_pages / record_scale)
    return hot_pages, cached_records, budget_pages - hot_pages


def cached_mask(layout, n_pages, ranking, frequency, hot_pages, cached_records, cache_pages=False):
    """Boolean mask of the records served from memory.

    By default these are the first cached_records records of ranking. With cache_pages the static part of the cache
    holds whole pages instead: the hot_pages pages of the layout with the most training accesses.
    """
    mask = np.zeros(len(layout), np.bool_)
    if cached_records <= 0:
        return mask
    if cache_pages:
        per_page = np.bincount(layout, weights=frequency.astype(np.float64), minlength=n_pages)
        top = np.zeros(n_pages, np.bool_)
        top[np.lexsort((np.arange(n_pages), -per_page))[:hot_pages]] = True
        return top[layout]
    mask[ranking[:cached_records]] = True
    return mask


def lockstep_order(step, turn, rollout, position):
    """Turn lockstep: sub-queries ordered by step, then turn, then rollout, then position in the request."""
    # [Paper] Section VII-A, Metrics: the rollouts of a step are interleaved turn by turn
    return np.lexsort((position, rollout, turn, step)).astype(np.int64)
