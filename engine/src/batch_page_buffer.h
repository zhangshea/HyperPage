// Read-once scoped page buffer: one page map shared by the sub-queries of a scope, which run in order on one thread,
// and emptied when the scope ends. Installed through the thread-local g_batch_buf; null (the default) keeps the stock
// search path of PQFlashIndex::cached_beam_search.
#pragma once
#include <cstdint>
#include <vector>
#include "tsl/robin_map.h"
#include "utils.h"

namespace diskann
{
// [Paper] Section VI-E: the scoped buffer is a hash map from page to an aligned page slot. In the frontier loop of
// cached_beam_search every uncached frontier node, in frontier order, looks up its page s = get_node_sector(id):
//   s in pages, stamped in an earlier round -> hit: the record is served from the buffer, no read
//   s in pages, stamped in this round       -> shared read: it uses the read already requested in this round
//   otherwise                               -> one page read into a fresh slot (page_reads++)
// Hits and misses stay in frontier order, so the visited and candidate insertions, and with them every expansion
// sequence, are the stock ones (Assumption 1). Co-resident records are never scored.
struct BatchPageBuffer
{
    struct Entry
    {
        char *buf;
        uint64_t round;
    };
    uint64_t page_bytes = 0;     // bytes per page read = num_sectors_per_node * SECTOR_LEN (one sector here)
    uint64_t align = 0;          // O_DIRECT buffer alignment (SECTOR_LEN)
    uint64_t chunk_pages = 512;  // arena grows in chunks of this many pages; allocations are kept across scopes
    std::vector<char *> chunks;
    uint64_t used = 0;           // pages handed out in the current scope
    uint64_t round_id = 0;       // global round stamp, incremented once per search round
    tsl::robin_map<uint64_t, Entry> pages; // sector number -> buffer slot

    // per-scope counters (reset by begin_batch)
    uint64_t page_reads = 0;     // issued page reads: equal to lambda_pi(E_sigma) of the scope (Corollary 1)
    uint64_t page_hits = 0;      // frontier nodes whose page was read earlier in the scope (earlier round or call)
    uint64_t inflight_dedup = 0; // frontier nodes whose page was requested earlier in the same round
    uint64_t calls = 0;          // cached_beam_search calls in the scope
    uint64_t expanded = 0;       // uncached frontier nodes (expanded nodes) in the scope
    uint64_t rounds = 0;         // search rounds in the scope
    uint64_t rounds_noread = 0;  // rounds whose frontier pages were all in the buffer (no I/O)
    uint64_t max_pages = 0;      // largest buffer occupancy seen (pages), over the buffer's lifetime

    BatchPageBuffer(uint64_t page_bytes_, uint64_t align_) : page_bytes(page_bytes_), align(align_)
    {
    }
    ~BatchPageBuffer()
    {
        for (char *c : chunks)
            aligned_free(c);
    }
    BatchPageBuffer(const BatchPageBuffer &) = delete;
    BatchPageBuffer &operator=(const BatchPageBuffer &) = delete;

    char *alloc_page()
    {
        uint64_t c = used / chunk_pages, o = used % chunk_pages;
        if (c == chunks.size())
        {
            char *p = nullptr;
            alloc_aligned((void **)&p, chunk_pages * page_bytes, align);
            chunks.push_back(p);
        }
        used++;
        if (used > max_pages)
            max_pages = used;
        return chunks[c] + o * page_bytes;
    }
    void begin_batch()
    {
        pages.clear();
        used = 0;
        page_reads = page_hits = inflight_dedup = calls = expanded = rounds = rounds_noread = 0;
    }
    void end_batch()
    {
        pages.clear(); // discard contents; chunk allocations are reused by the next scope
        used = 0;
    }
};
extern thread_local BatchPageBuffer *g_batch_buf;
} // namespace diskann
