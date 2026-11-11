// Hybrid-cache search application: hot records in DiskANN's node cache, every other page through one shared page LRU
// (hybrid_page_cache.h); sub-queries arrive in waves, and the threads of a wave take them dynamically.
// Follows the structure of DiskANN's search applications (https://github.com/microsoft/DiskANN, MIT licence).
#include "common_includes.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "traversal_log.h"
#include "hybrid_page_cache.h"
#include "utils.h"
#include <omp.h>
#include <sched.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>

// [Paper] Section VI-E, hybrid cache H_{D,rho} (Definition 7).
//   hot records : the node cache (load_cache_list) holds the records listed in <hot.u32>; their accesses read nothing.
//                 Loaded before the timed section.
//   page LRU    : one HybridPageCache of <lru_pages> pages shared by all threads, kept across the whole timed section
//                 from an empty start.
//   arrival     : sub-queries come in waves (one wave = one (step, turn) of turn lockstep: all rollouts of a step send
//                 the retrieval calls of a turn together). Inside a wave, NT threads take calls dynamically; waves are
//                 separated by a barrier. With one thread the LRU sees the calls in the replay's order.
//   order       : frontier processes the records of a round in frontier order, cached or read (every reported run);
//                 stock keeps DiskANN's order, cached records of a round first.
//   warm-up     : the calls of <warm.bin> run first, untimed, with the page LRU off (node cache on), so the LRU starts
//                 empty and no measured call runs untimed.
// Outputs:
//   <out>.steps.bin : u64 ns, then ns rows of 14 u64: reads, hits, waits, calls, expanded, cached, wall_ns, io_ns,
//                     pq_ns, nbr_ns, lock_ns, wait_ns, busy_ns (sum of call latencies), rounds_noread
//   <out>.calls.bin : u64 nq, then nq rows (query-file order) of 3 u64: latency_ns, reads, expanded
//   <out>.res.bin   : u32 nq, u32 K, ids (u32 nq*K), dists (f32 nq*K)
//   <out>.log.bin   : traversal log in search_and_log's format (log = 1 only; logging is off for timing)
//   <out>.host      : /proc/stat and /proc/diskstats taken right before and right after the timed section
//   stdout          : one "summary" line
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

static long long status_kb(const char *key)
{
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(key, 0) == 0)
            return std::stoll(line.substr(strlen(key)));
    return -1;
}

static std::string slurp(const char *p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename V> static std::vector<V> read_raw(const std::string &p)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot open " + p);
    size_t bytes = f.tellg();
    if (bytes % sizeof(V) != 0)
        throw std::runtime_error(p + ": file length " + std::to_string(bytes) + " is not a multiple of " +
                                 std::to_string(sizeof(V)) + " bytes");
    f.seekg(0);
    std::vector<V> v(bytes / sizeof(V));
    f.read((char *)v.data(), bytes);
    return v;
}

template <typename T> int run(char **argv)
{
    std::string dist = argv[0], prefix = argv[1], qfile = argv[2], wfile = argv[3], sfile = argv[4];
    uint32_t L = std::stoi(argv[5]), W = std::stoi(argv[6]), K = std::stoi(argv[7]), NT = std::stoi(argv[8]);
    std::string hotfile = argv[9];
    uint64_t lru_pages = std::stoull(argv[10]);
    std::string out = argv[11];
    int do_log = std::stoi(argv[12]);
    std::string warmfile = argv[13], order = argv[14];
    if (order != "frontier" && order != "stock")
    {
        std::cerr << "order must be frontier or stock\n";
        return 1;
    }
    diskann::Metric metric = dist == "mips" ? diskann::Metric::INNER_PRODUCT : diskann::Metric::L2;

    T *query = nullptr;
    size_t nq, qdim, qadim;
    diskann::load_aligned_bin<T>(qfile, query, nq, qdim, qadim);
    std::vector<uint64_t> wptr = read_raw<uint64_t>(wfile);
    std::vector<uint32_t> wstep = read_raw<uint32_t>(sfile);
    if (wptr.size() < 2 || wptr.front() != 0 || wptr.back() != nq || wstep.size() != wptr.size() - 1)
    {
        std::cerr << "bad wave files: need u64 ptr[0..nw] with ptr[0]=0, ptr[nw]=nq, and u32 step[nw]\n";
        return 1;
    }
    const uint64_t nw = wptr.size() - 1;
    for (uint64_t w = 0; w < nw; w++)
        if (wptr[w + 1] <= wptr[w] || (w > 0 && wstep[w] < wstep[w - 1]))
        {
            std::cerr << "empty wave or decreasing step at wave " << w << "\n";
            return 1;
        }
    const uint64_t ns = (uint64_t)wstep.back() + 1;

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<T>> idx(new diskann::PQFlashIndex<T>(reader, metric));
    if (idx->load(NT, prefix.c_str()) != 0)
    {
        std::cerr << "load failed\n";
        return 2;
    }
    const long long rss_loaded_kb = status_kb("VmRSS:");
    std::vector<uint32_t> hot;
    if (hotfile != "-")
    {
        hot = read_raw<uint32_t>(hotfile);
        idx->load_cache_list(hot);
    }
    const long long rss_cache_kb = status_kb("VmRSS:");
    uint64_t nsectors = 0;
    {
        std::ifstream f(prefix + "_disk.index", std::ios::binary | std::ios::ate);
        nsectors = (uint64_t)f.tellg() / diskann::defaults::SECTOR_LEN;
    }
    diskann::HybridPageCache pc(lru_pages, nsectors, diskann::defaults::SECTOR_LEN);
    const long long rss_lru_kb = status_kb("VmRSS:");
    omp_set_num_threads(NT);
    std::vector<diskann::HybridCtx> ctx(NT);
    for (auto &c : ctx)
    {
        c.pc = &pc;
        c.stock_order = order == "stock";
    }

    std::vector<uint64_t> res(nq * K);
    std::vector<float> resd(nq * K);
    std::vector<diskann::QueryStats> stats(nq);
    std::vector<diskann::TraversalLog> logs(do_log ? nq : 0);
    std::vector<uint64_t> call_lat(nq, 0), call_reads(nq, 0), call_exp(nq, 0), call_hits(nq, 0), call_waits(nq, 0),
        call_cached(nq, 0), call_lock(nq, 0), call_wait_ns(nq, 0), call_noread(nq, 0);
    std::vector<uint64_t> wave_ns(nw, 0);

    if (warmfile != "-")
    {   // untimed warm-up: page LRU off, node cache on
        T *wq = nullptr;
        size_t wn, wd, wad;
        diskann::load_aligned_bin<T>(warmfile, wq, wn, wd, wad);
        if (wad != qadim)
        {
            std::cerr << "warm-up dimension differs\n";
            return 1;
        }
        std::vector<uint64_t> wr(wn * K);
        std::vector<float> wdd(wn * K);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)wn; i++)
            idx->cached_beam_search(wq + i * wad, K, L, wr.data() + i * K, wdd.data() + i * K, W, false);
        diskann::aligned_free(wq);
    }

    const std::string stat0 = slurp("/proc/stat"), disk0 = slurp("/proc/diskstats");
    long long rb0 = proc_read_bytes();
    auto w0 = std::chrono::steady_clock::now();
    for (uint64_t w = 0; w < nw; w++)
    {
        auto a = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = (int64_t)wptr[w]; i < (int64_t)wptr[w + 1]; i++)
        {
            diskann::HybridCtx &c = ctx[omp_get_thread_num()];
            const uint64_t r0 = c.reads, e0 = c.expanded, h0 = c.hits, x0 = c.waits, k0 = c.cached, l0 = c.lock_ns,
                           t0 = c.wait_ns, n0 = c.rounds_noread;
            diskann::g_hyb = &c;
            if (do_log)
                diskann::g_trav_log = &logs[i];
            auto s = std::chrono::steady_clock::now();
            idx->cached_beam_search(query + i * qadim, K, L, res.data() + i * K, resd.data() + i * K, W, false,
                                    &stats[i]);
            auto e = std::chrono::steady_clock::now();
            diskann::g_trav_log = nullptr;
            diskann::g_hyb = nullptr;
            call_lat[i] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
            call_reads[i] = c.reads - r0;
            call_exp[i] = c.expanded - e0;
            call_hits[i] = c.hits - h0;
            call_waits[i] = c.waits - x0;
            call_cached[i] = c.cached - k0;
            call_lock[i] = c.lock_ns - l0;
            call_wait_ns[i] = c.wait_ns - t0;
            call_noread[i] = c.rounds_noread - n0;
        }
        auto b = std::chrono::steady_clock::now();
        wave_ns[w] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    auto w1 = std::chrono::steady_clock::now();
    long long rb1 = proc_read_bytes();
    const std::string stat1 = slurp("/proc/stat"), disk1 = slurp("/proc/diskstats");
    const double secs = std::chrono::duration<double>(w1 - w0).count();
    {
        std::ofstream h(out + ".host");
        h << "@@stat0\n" << stat0 << "@@disk0\n" << disk0 << "@@stat1\n" << stat1 << "@@disk1\n" << disk1;
    }

    const uint64_t NC = 14;
    std::vector<uint64_t> srow(ns * NC, 0);
    for (uint64_t w = 0; w < nw; w++)
    {
        uint64_t *r = srow.data() + (uint64_t)wstep[w] * NC;
        r[6] += wave_ns[w];
        for (uint64_t i = wptr[w]; i < wptr[w + 1]; i++)
        {
            r[0] += call_reads[i];
            r[1] += call_hits[i];
            r[2] += call_waits[i];
            r[3] += 1;
            r[4] += call_exp[i];
            r[5] += call_cached[i];
            r[7] += stats[i].io_ns;
            r[8] += stats[i].pq_ns;
            r[9] += stats[i].nbr_ns;
            r[10] += call_lock[i];
            r[11] += call_wait_ns[i];
            r[12] += call_lat[i];
            r[13] += call_noread[i];
        }
    }
    uint64_t tot[NC] = {0};
    for (uint64_t s = 0; s < ns; s++)
        for (uint64_t j = 0; j < NC; j++)
            tot[j] += srow[s * NC + j];
    std::set<int> cpus;
#pragma omp parallel num_threads(NT)
    {
#pragma omp critical
        cpus.insert(sched_getcpu());
    }
    std::ostringstream cpul;
    for (int c : cpus)
        cpul << (cpul.tellp() > 0 ? "," : "") << c;
    std::vector<double> ls(nq);
    for (size_t i = 0; i < nq; i++)
        ls[i] = call_lat[i] * 1e-6;
    std::sort(ls.begin(), ls.end());
    auto pct = [&](double p) {
        double pos = p * (nq - 1);
        size_t lo = (size_t)pos, hi = std::min(lo + 1, nq - 1);
        return ls[lo] + (pos - lo) * (ls[hi] - ls[lo]);
    };
    {
        std::ofstream f(out + ".steps.bin", std::ios::binary);
        f.write((char *)&ns, 8);
        f.write((char *)srow.data(), 8 * srow.size());
    }
    {
        std::ofstream f(out + ".calls.bin", std::ios::binary);
        uint64_t n64 = nq;
        f.write((char *)&n64, 8);
        for (size_t i = 0; i < nq; i++)
        {
            uint64_t row[3] = {call_lat[i], call_reads[i], call_exp[i]};
            f.write((char *)row, 24);
        }
    }
    {
        std::ofstream g(out + ".res.bin", std::ios::binary);
        uint32_t h2[2] = {(uint32_t)nq, K};
        g.write((char *)h2, 8);
        std::vector<uint32_t> r32(nq * K);
        for (size_t i = 0; i < nq * K; i++)
            r32[i] = (uint32_t)res[i];
        g.write((char *)r32.data(), 4 * nq * K);
        g.write((char *)resd.data(), 4 * nq * K);
    }
    if (do_log)
    {
        uint64_t tot_n = 0, tot_r = 0, tot_s = 0;
        for (auto &l : logs)
        {
            tot_n += l.nodes.size();
            tot_r += l.round_end.size();
            tot_s += l.read_sectors.size();
        }
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
    }
    const double busy = (double)tot[12];
    std::cout << "summary sector_len " << diskann::defaults::SECTOR_LEN << " threads " << NT << " L " << L << " W " << W
              << " K " << K << " order " << order << " hot_records " << hot.size() << " lru_pages " << lru_pages
              << " log " << do_log << " waves " << nw << " steps " << ns << " calls " << nq << " secs " << secs
              << " calls_per_s " << nq / secs << " ms_per_step " << 1e3 * secs / ns << " page_reads " << tot[0]
              << " lru_hits " << tot[1] << " inflight_waits " << tot[2] << " expanded " << tot[4] << " cached "
              << tot[5] << " proc_read_bytes " << (rb1 - rb0) << " bytes_per_read "
              << (tot[0] ? (double)(rb1 - rb0) / tot[0] : 0.0) << " lat_ms_median " << pct(0.5) << " lat_ms_p99 "
              << pct(0.99) << " lat_ms_mean " << std::accumulate(ls.begin(), ls.end(), 0.0) / nq << " busy_thread_s "
              << busy * 1e-9 << " io_share " << tot[7] / busy << " pq_share " << tot[8] / busy << " nbr_share "
              << tot[9] / busy << " lock_share " << tot[10] / busy << " wait_share " << tot[11] / busy
              << " us_per_read " << (tot[0] ? tot[7] * 1e-3 / tot[0] : 0.0) << " rounds_noread " << tot[13]
              << " rss_loaded_kb " << rss_loaded_kb << " rss_cache_kb " << rss_cache_kb << " rss_lru_kb " << rss_lru_kb
              << " rss_end_kb " << status_kb("VmRSS:") << " hwm_kb " << status_kb("VmHWM:") << " cpus " << cpul.str()
              << std::endl;
    diskann::aligned_free(query);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 17)
    {
        std::cerr << "usage: hybrid_search <float|int8> <l2|mips> <index_prefix> <query.bin> <wave_ptr.u64> "
                     "<wave_step.u32> <L> <W> <K> <threads> <hot.u32|-> <lru_pages> <out_prefix> <log 0|1> "
                     "<warm.bin|-> <frontier|stock>\n";
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
