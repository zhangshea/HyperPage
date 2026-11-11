// Page LRU of the hybrid cache H_{D,rho}: one LRU of `cap` pages shared by all threads and kept across calls, steps and
// waves, with pinning and waits on pages in flight. The hot records live in DiskANN's node cache and read nothing.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "utils.h"

namespace diskann
{
// [Paper] Definition 7 and Section VI-E. The hook in cached_beam_search (installed by the thread-local g_hyb) calls
// acquire(sector) once per search round, under the cache mutex, for every uncached frontier node in frontier order:
//   page in the cache     -> hit: moved to the LRU head and pinned; if another thread is still reading it, the caller
//                            waits for it after finishing its own reads (no cycle can form)
//   page not in the cache -> miss: a slot is taken (a free one, else the least recent unpinned page is evicted),
//                            pinned, marked loading, and the caller reads the page into it
// Pins are released after the round's records are processed. With one thread and a cache that always has an unpinned
// page (capacity at least the beam width) this is, operation for operation, the replay's LRU over the same accesses
// in the same order, so per-step reads equal the replay. With several threads the
// accesses of concurrent calls interleave, so reads may differ from the replay.
struct HybridPageCache
{
    uint64_t page_bytes = 0, cap = 0, used = 0;
    char *arena = nullptr;
    std::vector<int32_t> slot_of;  // sector -> slot, -1 when not cached
    std::vector<uint64_t> sec_of;  // slot -> sector
    std::vector<int32_t> prv, nxt; // LRU list over slots; head = most recent
    std::unique_ptr<std::atomic<int32_t>[]> pin;
    std::unique_ptr<std::atomic<uint8_t>[]> ready;
    int32_t head = -1, tail = -1;
    std::mutex mu;

    HybridPageCache(uint64_t cap_, uint64_t nsectors, uint64_t page_bytes_)
        : page_bytes(page_bytes_), cap(cap_), slot_of(nsectors, -1), sec_of(cap_, 0), prv(cap_, -1), nxt(cap_, -1),
          pin(new std::atomic<int32_t>[cap_]), ready(new std::atomic<uint8_t>[cap_])
    {
        if (cap == 0 || cap > (uint64_t)INT32_MAX)
            throw std::runtime_error("hybrid page cache: capacity must be in [1, INT32_MAX] pages");
        alloc_aligned((void **)&arena, cap * page_bytes, page_bytes);
        for (uint64_t s = 0; s < cap; s++)
        {
            pin[s].store(0, std::memory_order_relaxed);
            ready[s].store(0, std::memory_order_relaxed);
        }
    }
    ~HybridPageCache()
    {
        aligned_free(arena);
    }
    HybridPageCache(const HybridPageCache &) = delete;
    HybridPageCache &operator=(const HybridPageCache &) = delete;

    char *slot_ptr(int32_t s) const
    {
        return arena + (uint64_t)s * page_bytes;
    }
    void unlink(int32_t s)
    {
        const int32_t a = prv[s], b = nxt[s];
        if (a >= 0)
            nxt[a] = b;
        else
            head = b;
        if (b >= 0)
            prv[b] = a;
        else
            tail = a;
        prv[s] = nxt[s] = -1;
    }
    void push_front(int32_t s)
    {
        prv[s] = -1;
        nxt[s] = head;
        if (head >= 0)
            prv[head] = s;
        else
            tail = s;
        head = s;
    }
    // Called with mu held. Returns the slot of `sec`, pinned; miss = true when the caller must read the page into it.
    int32_t acquire(uint64_t sec, bool &miss)
    {
        int32_t s = slot_of[sec];
        if (s >= 0)
        {
            if (s != head)
            {
                unlink(s);
                push_front(s);
            }
            pin[s].fetch_add(1, std::memory_order_relaxed);
            miss = false;
            return s;
        }
        if (used < cap)
            s = (int32_t)used++;
        else
        {
            s = tail;
            while (s >= 0 && pin[s].load(std::memory_order_acquire) > 0)
                s = prv[s];
            if (s < 0)
                throw std::runtime_error("hybrid page cache: no unpinned page to evict");
            unlink(s);
            slot_of[sec_of[s]] = -1;
        }
        slot_of[sec] = s;
        sec_of[s] = sec;
        ready[s].store(0, std::memory_order_relaxed);
        pin[s].store(1, std::memory_order_relaxed);
        push_front(s);
        miss = true;
        return s;
    }
};

// Per-thread hook state and counters (counters are cumulative; the harness takes per-call differences).
struct HybridCtx
{
    HybridPageCache *pc = nullptr;
    bool stock_order = false;  // true: stock DiskANN order (cached records of a round first); false: frontier order
    uint64_t reads = 0;        // issued page reads (LRU misses)
    uint64_t hits = 0;         // LRU hits (including pages being read by this or another thread)
    uint64_t waits = 0;        // hits on a page still being read by another thread
    uint64_t expanded = 0;     // expanded nodes (cached + uncached)
    uint64_t cached = 0;       // expanded nodes served by the node cache
    uint64_t rounds = 0, rounds_noread = 0;
    uint64_t lock_ns = 0;      // time to take and hold the cache mutex (contention included)
    uint64_t wait_ns = 0;      // time spent waiting for other threads' in-flight pages
};
extern thread_local HybridCtx *g_hyb;
} // namespace diskann
