// Logging search application: stock DiskANN cached_beam_search over a query file, writing per sub-query the expanded
// nodes, rounds, search-tree parents, issued page reads and results (<out>.log.bin, <out>.res.bin).
// Follows the structure of DiskANN's search applications (https://github.com/microsoft/DiskANN, MIT licence).
#include "common_includes.h"
#include "pq_flash_index.h"
#include "percentile_stats.h"
#include "linux_aligned_file_reader.h"
#include "traversal_log.h"
#include "utils.h"
#include <omp.h>
#include <fstream>
#include <sstream>

static long long proc_read_bytes()
{
    std::ifstream f("/proc/self/io");
    std::string k;
    long long v;
    while (f >> k >> v)
        if (k == "read_bytes:")
            return v;
    return -1;
}

template <typename T> int run(char **argv)
{
    std::string dist = argv[0], prefix = argv[1], qfile = argv[2], out = argv[7];
    uint32_t L = std::stoi(argv[3]), W = std::stoi(argv[4]), K = std::stoi(argv[5]), NT = std::stoi(argv[6]);
    uint32_t ncache = std::stoi(argv[8]);
    diskann::Metric metric = dist == "mips" ? diskann::Metric::INNER_PRODUCT : diskann::Metric::L2;
    T *query = nullptr;
    size_t nq, qdim, qadim;
    diskann::load_aligned_bin<T>(qfile, query, nq, qdim, qadim);
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<T>> idx(new diskann::PQFlashIndex<T>(reader, metric));
    if (idx->load(NT, prefix.c_str()) != 0)
    {
        std::cerr << "load failed\n";
        return 2;
    }
    std::vector<uint32_t> node_list;
    if (ncache > 0)
    {
        idx->cache_bfs_levels(ncache, node_list);
        idx->load_cache_list(node_list);
    }
    omp_set_num_threads(NT);
    std::vector<uint64_t> res(nq * K);
    std::vector<float> resd(nq * K);
    std::vector<diskann::QueryStats> stats(nq);
    std::vector<diskann::TraversalLog> logs(nq);
    long long rb0 = proc_read_bytes();
    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)nq; i++)
    {
        diskann::g_trav_log = &logs[i];
        idx->cached_beam_search(query + i * qadim, K, L, res.data() + i * K, resd.data() + i * K, W, false, &stats[i]);
        diskann::g_trav_log = nullptr;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    long long rb1 = proc_read_bytes();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    uint64_t tot_n = 0, tot_r = 0, tot_s = 0, tot_ios = 0;
    for (auto &l : logs)
    {
        tot_n += l.nodes.size();
        tot_r += l.round_end.size();
        tot_s += l.read_sectors.size();
    }
    for (auto &s : stats)
        tot_ios += s.n_ios;
    std::ofstream f(out + ".log.bin", std::ios::binary);
    uint64_t hdr[4] = {nq, tot_n, tot_r, tot_s};
    f.write((char *)hdr, sizeof(hdr));
    std::vector<uint64_t> noff(nq + 1, 0), roff(nq + 1, 0), soff(nq + 1, 0);
    for (size_t i = 0; i < nq; i++)
    {
        noff[i + 1] = noff[i] + logs[i].nodes.size();
        roff[i + 1] = roff[i] + logs[i].round_end.size();
        soff[i + 1] = soff[i] + logs[i].read_sectors.size();
    }
    f.write((char *)noff.data(), 8 * (nq + 1));
    f.write((char *)roff.data(), 8 * (nq + 1));
    f.write((char *)soff.data(), 8 * (nq + 1));
    for (auto &l : logs)
        f.write((char *)l.nodes.data(), 4 * l.nodes.size());
    for (auto &l : logs)
        f.write((char *)l.parents.data(), 4 * l.parents.size());
    for (auto &l : logs)
        f.write((char *)l.round_end.data(), 4 * l.round_end.size());
    for (auto &l : logs)
        f.write((char *)l.read_sectors.data(), 4 * l.read_sectors.size());
    std::vector<uint32_t> nios(nq), nhops(nq), nch(nq);
    std::vector<float> iou(nq);
    for (size_t i = 0; i < nq; i++)
    {
        nios[i] = stats[i].n_ios;
        nhops[i] = stats[i].n_hops;
        nch[i] = stats[i].n_cache_hits;
        iou[i] = stats[i].io_us;
    }
    f.write((char *)nios.data(), 4 * nq);
    f.write((char *)nhops.data(), 4 * nq);
    f.write((char *)nch.data(), 4 * nq);
    f.write((char *)iou.data(), 4 * nq);
    f.close();
    std::ofstream g(out + ".res.bin", std::ios::binary);
    uint32_t h2[2] = {(uint32_t)nq, K};
    g.write((char *)h2, 8);
    std::vector<uint32_t> r32(nq * K);
    for (size_t i = 0; i < nq * K; i++)
        r32[i] = (uint32_t)res[i];
    g.write((char *)r32.data(), 4 * nq * K);
    g.write((char *)resd.data(), 4 * nq * K);
    g.close();
    std::cout << "nq " << nq << " L " << L << " W " << W << " threads " << NT << " secs " << secs << " qps " << nq / secs
              << " mean_n_ios " << (double)tot_ios / nq << " mean_expanded " << (double)tot_n / nq
              << " logged_reads " << tot_s << " proc_read_bytes_delta " << (rb1 - rb0)
              << " bytes_per_logged_read " << (tot_s ? (double)(rb1 - rb0) / tot_s : 0) << std::endl;
    diskann::aligned_free(query);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 11)
    {
        std::cerr << "usage: search_and_log <float|int8> <l2|mips> <index_prefix> <query.bin> <L> <W> <K> <threads> "
                     "<out_prefix> <num_nodes_to_cache>\n";
        return 1;
    }
    std::string dt = argv[1];
    if (dt == "int8")
        return run<int8_t>(argv + 2);
    if (dt == "float")
        return run<float>(argv + 2);
    std::cerr << "bad data type\n";
    return 1;
}
