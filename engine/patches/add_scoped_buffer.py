"""Add the compile-time page size with its load-time guards, the read-once scoped page buffer hook, nanosecond section
timers and the scoped search application (batch_search) to a DiskANN tree that has the logger and the layout map."""
# With g_batch_buf == nullptr the stock code path runs. Applied after add_logger.py and add_layout_map.py.
# DiskANN is MIT-licensed (https://github.com/microsoft/DiskANN); the anchors below are excerpts of its sources at
# the pinned commit (include/defaults.h, include/timer.h, include/percentile_stats.h, src/pq_flash_index.cpp).
import argparse

import edit_util as E

CPP = 'src/pq_flash_index.cpp'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='DiskANN source tree with the logger and the layout map')
    T = ap.parse_args().tree
    if 'batch_page_buffer.h' in E.read(T, CPP):
        raise SystemExit(f'{CPP}: the scoped buffer is present already')

    # 1. page size (read unit) at compile time: -DDISKANN_SECTOR_LEN=8192 gives 8 KB pages, B = 12 records of 644 bytes
    E.replace(T, 'include/defaults.h', "const uint64_t SECTOR_LEN = 4096;",
              "#ifndef DISKANN_SECTOR_LEN\n#define DISKANN_SECTOR_LEN "
              "4096\n#endif\nconst uint64_t SECTOR_LEN = DISKANN_SECTOR_LEN; // read unit (page size), set at "
              "compile time (4096 or"
              " 8192)")

    # 2. buffer header and hooks
    E.copy_source(T, 'batch_page_buffer.h', 'include')
    E.replace(T, CPP, '#include "pq_flash_index.h"\n', '#include "pq_flash_index.h"\n#include "batch_page_buffer.h"\n')
    E.replace(T, CPP, "thread_local TraversalLog *g_trav_log = nullptr;\n",
              "thread_local TraversalLog *g_trav_log = "
              "nullptr;\nthread_local BatchPageBuffer *g_batch_buf = nullptr; // read-once scoped page buffer (null "
              "= stock)\n")

    # load-time guards: the header's records per page must fit the compiled page size, and the file must hold every page
    E.replace(T, CPP, """    diskann::cout << "Disk-Index File Meta-data: ";""",
              """    if (_nnodes_per_sector > 0)
    {   // the file must be written for this read unit (SECTOR_LEN)
        if (_nnodes_per_sector * _max_node_len > defaults::SECTOR_LEN)
            throw ANNException("engine: nodes per sector x max node len exceeds SECTOR_LEN (file written for another "
                               "read unit?)",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        uint64_t npages_needed = DIV_ROUND_UP(this->_num_points, _nnodes_per_sector);
        if (_has_layout)
            npages_needed = 1 + (uint64_t)*std::max_element(_layout_page.begin(), _layout_page.end());
        std::ifstream szf(_disk_index_file, std::ios::binary | std::ios::ate);
        uint64_t fbytes = (uint64_t)szf.tellg();
        if (fbytes % defaults::SECTOR_LEN != 0 || fbytes < (1 + npages_needed) * defaults::SECTOR_LEN)
            throw ANNException("engine: disk index size does not match SECTOR_LEN and the page count", -1, __FUNCSIG__,
                               __FILE__, __LINE__);
        diskann::cout << "Engine read unit " << defaults::SECTOR_LEN << " B, " << npages_needed << " pages, file "
                      << fbytes << " B" << std::endl;
    }
    diskann::cout << "Disk-Index File Meta-data: ";""")

    E.replace(T, CPP, """    TraversalLog *tl = g_trav_log;
""", """    TraversalLog *tl = g_trav_log;
    BatchPageBuffer *bb = g_batch_buf; // read-once scoped page buffer (null = stock path)
    if (bb)
    {
        if (num_sectors_per_node != 1 || bb->page_bytes != defaults::SECTOR_LEN)
            throw ANNException("batch page buffer: multi-sector nodes are not supported", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        bb->calls++;
    }
""")
    E.replace(T, CPP, """        if (tl)
            tl->round_end.push_back((uint32_t)tl->nodes.size());
""", """        if (tl)
            tl->round_end.push_back((uint32_t)tl->nodes.size());
        if (bb)
        {
            bb->round_id++;
            bb->rounds++;
            bb->expanded += frontier.size();
        }
""")
    # [Paper] Section VI-E: an expanded node is a hit if the scope has read its page, a shared read if the round has
    # already requested it, and otherwise one page read; hits and misses stay in frontier order
    E.replace(T, CPP, """                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
""", """                auto id = frontier[i];
                if (bb)
                {   // hits and misses stay in frontier order in frontier_nhoods (A1)
                    const uint64_t sec = get_node_sector((size_t)id);
                    auto pit = bb->pages.find(sec);
                    if (pit != bb->pages.end())
                    {
                        frontier_nhoods.push_back(std::make_pair(id, pit->second.buf));
                        if (pit->second.round == bb->round_id)
                            bb->inflight_dedup++;
                        else
                            bb->page_hits++;
                        continue;
                    }
                    char *pbuf = bb->alloc_page();
                    bb->pages.insert({sec, BatchPageBuffer::Entry{pbuf, bb->round_id}});
                    frontier_nhoods.push_back(std::make_pair(id, pbuf));
                    if (tl)
                        tl->read_sectors.push_back((uint32_t)sec);
                    frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, defaults::SECTOR_LEN, pbuf);
                    bb->page_reads++;
                    if (stats != nullptr)
                    {
                        stats->n_4k++;
                        stats->n_ios++;
                    }
                    num_ios++;
                    continue;
                }
                std::pair<uint32_t, char *> fnhood;
""")
    E.replace(T, CPP, """            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
#endif
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
""", """            if (!frontier_read_reqs.empty())           // pages may all be buffered
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
""")

    # 3. nanosecond section timers: the stock Timer::elapsed() truncates to whole microseconds, which biases the
    # per-node CPU sections (a few microseconds each) low; these counters reuse the stock timers' clock reads
    E.replace(T, 'include/timer.h', "    float elapsed_seconds() const\n",
              "    long long elapsed_ns() const // elapsed() truncates to whole microseconds\n    {\n"
              "        return std::chrono::duration_cast<std::chrono::nanoseconds>(_clock::now() - "
              "check_point).count();\n"
              "    }\n\n    float elapsed_seconds() const\n")
    E.replace(T, 'include/percentile_stats.h', "    float cpu_us = 0;   // total time spent in CPU\n",
              "    float cpu_us = 0;   // total time spent in CPU\n"
              "    // nanosecond resolution: time in reader->read; PQ distance work (table build + code lookups);\n"
              "    // neighbour loop (visited set + candidate-list inserts). The full-precision distance is not "
              "separately timed.\n"
              "    unsigned long long io_ns = 0, pq_ns = 0, nbr_ns = 0;\n")
    E.replace(T, CPP, """    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);
""", """    Timer pq_table_timer; // PQ table build
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);
    if (stats != nullptr)
        stats->pq_ns += (unsigned long long)pq_table_timer.elapsed_ns();
""")
    E.replace(T, CPP, """            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
""", """            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                const long long e_ns = cpu_timer.elapsed_ns();
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)(e_ns / 1000);
                stats->pq_ns += (unsigned long long)e_ns;
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
""")
    E.replace(T, CPP, """            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        hops++;
""", """            if (stats != nullptr)
            {
                const long long e_ns = cpu_timer.elapsed_ns();
                stats->cpu_us += (float)(e_ns / 1000);
                stats->nbr_ns += (unsigned long long)e_ns;
            }
        }

        hops++;
""")
    E.prepend_if_missing(T, CPP, '#include <algorithm>')

    # 4. the scoped search application
    E.copy_source(T, 'batch_search.cpp', 'apps')
    E.add_app(T, 'batch_search', 'batch_search.cpp')
    print('scoped buffer added to', T)


if __name__ == '__main__':
    main()
