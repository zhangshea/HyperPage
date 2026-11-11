// Per-thread traversal log of PQFlashIndex::cached_beam_search: expanded nodes, parents, rounds and issued page reads.
// While diskann::g_trav_log is null (the default) the stock search path runs unchanged.
#pragma once
#include <cstdint>
#include <vector>
namespace diskann
{
struct TraversalLog
{
    std::vector<uint32_t> nodes;     // expanded nodes in expansion order (frontier order within a round)
    std::vector<uint32_t> parents;   // node whose adjacency list first put the node into `visited` (UINT32_MAX for entry)
    std::vector<uint32_t> round_end; // exclusive end index into `nodes` of each round
    std::vector<uint32_t> read_sectors; // sector numbers of the AlignedRead requests actually issued, in order
    void clear()
    {
        nodes.clear();
        parents.clear();
        round_end.clear();
        read_sectors.clear();
    }
};
extern thread_local TraversalLog *g_trav_log;
} // namespace diskann
