"""Add the hybrid cache H_{D,rho} to cached_beam_search: the shared page LRU of hybrid_page_cache.h, read once per round
in frontier order, frontier-order processing of cached and read records, and the hybrid_search application."""
# With g_hyb == nullptr the code path of the scoped engine is unchanged (its two record-processing loops become lambdas
# called in the same order). Applied after add_scoped_buffer.py. DiskANN is MIT-licensed
# (https://github.com/microsoft/DiskANN); anchors that are not our own inserted lines are excerpts of its
# src/pq_flash_index.cpp at the pinned commit, used to locate the edits.
import argparse

import edit_util as E

CPP = 'src/pq_flash_index.cpp'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='DiskANN source tree with the logger, the layout map and the scoped buffer')
    T = ap.parse_args().tree
    if 'hybrid_page_cache.h' in E.read(T, CPP):
        raise SystemExit(f'{CPP}: the hybrid cache is present already')

    E.copy_source(T, 'hybrid_page_cache.h', 'include')
    E.replace(T, CPP, '#include "batch_page_buffer.h"\n',
              '#include "batch_page_buffer.h"\n#include "hybrid_page_cache.h"\n#include <thread>\n')
    E.replace(T, CPP, "thread_local BatchPageBuffer *g_batch_buf = nullptr; // read-once scoped page buffer (null = "
                      "stock)\n",
              "thread_local BatchPageBuffer *g_batch_buf = nullptr; // read-once scoped page buffer (null = "
              "stock)\nthread_local HybridCtx *g_hyb = nullptr; // hybrid cache page LRU (null = off)\n")
    E.replace(T, CPP, "    BatchPageBuffer *bb = g_batch_buf; // read-once scoped page buffer (null = stock path)\n",
              "    BatchPageBuffer *bb = g_batch_buf; // read-once scoped page buffer (null = stock path)\n"
              "    HybridCtx *hy = g_hyb;             // hybrid cache page LRU (null = off)\n"
              "    const bool hy_order = hy != nullptr && !hy->stock_order;\n"
              "    if (hy && (bb || num_sectors_per_node != 1 || hy->pc->page_bytes != defaults::SECTOR_LEN))\n"
              "        throw ANNException(\"hybrid cache: needs one-sector nodes and no batch buffer\", -1, "
              "__FUNCSIG__, __FILE__,\n"
              "                           __LINE__);\n")
    # the order of the round's nodes (node-cache records and frontier nodes) as the beam selects them
    E.replace(T, CPP, "    cached_nhoods.reserve(2 * beam_width);\n",
              "    cached_nhoods.reserve(2 * beam_width);\n"
              "    std::vector<std::pair<uint8_t, uint32_t>> hy_seq; // (cached?, index) of the round's nodes, in "
              "order\n"
              "    std::vector<int32_t> hy_pinned, hy_own, hy_wait;\n"
              "    hy_seq.reserve(2 * beam_width);\n")
    E.replace(T, CPP, "        cached_nhoods.clear();\n", "        cached_nhoods.clear();\n        hy_seq.clear();\n")
    E.replace(T, CPP, """            if (iter != _nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
""", """            if (iter != _nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (hy)
                    hy_seq.emplace_back((uint8_t)1, (uint32_t)(cached_nhoods.size() - 1));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
                if (hy)
                    hy_seq.emplace_back((uint8_t)0, (uint32_t)(frontier.size() - 1));
            }
""")
    E.replace(T, CPP, """        if (bb)
        {
            bb->round_id++;
            bb->rounds++;
            bb->expanded += frontier.size();
        }
""", """        if (bb)
        {
            bb->round_id++;
            bb->rounds++;
            bb->expanded += frontier.size();
        }
        if (hy)
        {
            hy->rounds++;
            hy->expanded += frontier.size() + cached_nhoods.size();
            hy->cached += cached_nhoods.size();
        }
""")
    # [Paper] Section VI-E: one LRU page cache shared by all threads; a page stays pinned while a round uses it, and a
    # request for a page in flight waits for that read
    E.replace(T, CPP, """            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                if (bb)
""", """            if (hy)
            {   // one critical section per round; LRU operations in frontier order (the replay's order)
                hy_pinned.clear();
                hy_own.clear();
                hy_wait.clear();
                Timer lk_timer;
                {
                    std::lock_guard<std::mutex> lk(hy->pc->mu);
                    for (uint64_t i = 0; i < frontier.size(); i++)
                    {
                        const uint32_t id = frontier[i];
                        const uint64_t sec = get_node_sector((size_t)id);
                        bool miss = false;
                        const int32_t s = hy->pc->acquire(sec, miss);
                        char *pbuf = hy->pc->slot_ptr(s);
                        frontier_nhoods.push_back(std::make_pair(id, pbuf));
                        hy_pinned.push_back(s);
                        if (miss)
                        {
                            hy_own.push_back(s);
                            frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, defaults::SECTOR_LEN, pbuf);
                            if (tl)
                                tl->read_sectors.push_back((uint32_t)sec);
                            hy->reads++;
                            if (stats != nullptr)
                            {
                                stats->n_4k++;
                                stats->n_ios++;
                            }
                            num_ios++;
                        }
                        else
                        {
                            hy->hits++;
                            if (!hy->pc->ready[s].load(std::memory_order_acquire))
                                hy_wait.push_back(s);
                        }
                    }
                }
                hy->lock_ns += (uint64_t)lk_timer.elapsed_ns();
            }
            for (uint64_t i = 0; i < (hy ? 0 : frontier.size()); i++)
            {
                auto id = frontier[i];
                if (bb)
""")
    E.replace(T, CPP, """            if (!frontier_read_reqs.empty())           // pages may all be buffered
                reader->read(frontier_read_reqs, ctx); // synchronous IO linux
            else if (bb)
                bb->rounds_noread++;
#endif
            if (stats != nullptr)
            {
                const long long e_ns = io_timer.elapsed_ns();
                stats->io_us += (float)(e_ns / 1000);
                stats->io_ns += (unsigned long long)e_ns;
            }
""", """            if (!frontier_read_reqs.empty())           // pages may all be buffered
                reader->read(frontier_read_reqs, ctx); // synchronous IO linux
            else if (bb)
                bb->rounds_noread++;
            else if (hy)
                hy->rounds_noread++;
#endif
            if (stats != nullptr)
            {
                const long long e_ns = io_timer.elapsed_ns();
                stats->io_us += (float)(e_ns / 1000);
                stats->io_ns += (unsigned long long)e_ns;
            }
            if (hy)
            {   // own reads are complete: publish them, then wait for pages other threads are still reading
                for (int32_t s : hy_own)
                    hy->pc->ready[s].store(1, std::memory_order_release);
                if (!hy_wait.empty())
                {
                    Timer w_timer;
                    for (int32_t s : hy_wait)
                    {
                        uint32_t spins = 0;
                        while (!hy->pc->ready[s].load(std::memory_order_acquire))
                            if (++spins % 64 == 0)
                                std::this_thread::yield();
                    }
                    hy->wait_ns += (uint64_t)w_timer.elapsed_ns();
                    hy->waits += hy_wait.size();
                }
            }
""")
    # the two record-processing loops become lambdas, so that hybrid mode can interleave them in frontier order
    E.replace(T, CPP, "        // process cached nhoods\n        for (auto &cached_nhood : cached_nhoods)\n        {\n",
              "        // process cached nhoods (as a lambda, so that hybrid mode can interleave them in frontier "
              "order)\n"
              "        auto hy_proc_cached = [&](std::pair<uint32_t, std::pair<uint32_t, uint32_t *>> &cached_nhood) "
              "{\n")
    E.replace(T, CPP, "                    retset.insert(nn);\n                }\n            }\n        }\n#ifdef "
                      "USE_BING_INFRA\n",
              "                    retset.insert(nn);\n                }\n            }\n        };\n"
              "        if (!hy_order)\n            for (auto &cached_nhood : cached_nhoods)\n                "
              "hy_proc_cached(cached_nhood);\n"
              "#ifdef USE_BING_INFRA\n")
    E.replace(T, CPP, "#else\n        for (auto &frontier_nhood : frontier_nhoods)\n        {\n#endif\n",
              "#else\n        auto hy_proc_frontier = [&](std::pair<uint32_t, char *> &frontier_nhood) {\n#endif\n")
    # [Paper] Section VI-E: hits and misses stay in frontier order, so Assumption 1 holds with a cache
    E.replace(T, CPP, "                stats->nbr_ns += (unsigned long long)e_ns;\n            }\n        }\n\n      "
                      "  hops++;\n",
              """                stats->nbr_ns += (unsigned long long)e_ns;
            }
        };
        if (hy_order)
        {   // the round's records in frontier order, cached or read
            for (auto &o : hy_seq)
            {
                if (o.first)
                    hy_proc_cached(cached_nhoods[o.second]);
                else
                    hy_proc_frontier(frontier_nhoods[o.second]);
            }
        }
        else
        {
            for (auto &frontier_nhood : frontier_nhoods)
                hy_proc_frontier(frontier_nhood);
        }
        if (hy)
            for (int32_t s : hy_pinned)
                hy->pc->pin[s].fetch_sub(1, std::memory_order_release);

        hops++;
""")
    E.copy_source(T, 'hybrid_search.cpp', 'apps')
    E.add_app(T, 'hybrid_search', 'hybrid_search.cpp')
    print('hybrid cache added to', T)


if __name__ == '__main__':
    main()
