// Read-once scoped search application: stock cached_beam_search (no node cache) over scopes of sub-queries, each scope
// on one thread with its own page buffer (batch_page_buffer.h), or one read per expanded node in mode call.
// Follows the structure of DiskANN's search applications (https://github.com/microsoft/DiskANN, MIT licence).
#include "common_includes.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "traversal_log.h"
#include "batch_page_buffer.h"
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

// Arguments: the batch file <batch_ptr.u64> holds u64 ptr[0..nb] with ptr[0] = 0 and ptr[nb] = number of queries; scope
// b is the query rows ptr[b]..ptr[b+1]-1. NT threads take whole scopes (dynamic schedule) and run the sub-queries of a
// scope in order.
//   mode batch : a thread-local BatchPageBuffer is installed for the scope (read-once page buffer, discarded at the end
//                of the scope); the per-scope count of issued page reads is recorded.
//   mode call  : stock engine, one read per expanded node; per-scope reads = sum of n_ios.
// The first `warm` scopes of the batch file are a warm-up: they run before the timed section and are not output.
// Outputs (timed scopes only):
//   <out>.batches.bin : u64 nb, then nb rows of 13 u64: reads, hits, inflight_dedup, calls, expanded, latency_ns,
//                       thread, cpu (sched_getcpu at scope end), io_ns, pq_ns, nbr_ns, rounds, rounds_noread
//   <out>.res.bin     : u32 nq, u32 K, ids (u32 nq*K), dists (f32 nq*K), as search_and_log
//   <out>.log.bin     : traversal log in search_and_log's format (only when log = 1; logging is off for timing)
//   <out>.host        : /proc/stat and /proc/diskstats taken right before and right after the timed section
//   stdout            : one "summary" line (wall time, scopes/s, calls/s, reads, /proc/self/io read bytes, latency)
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

static std::string slurp(const char *p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename T> int run(char **argv)
{
    std::string dist = argv[0], prefix = argv[1], qfile = argv[2], bfile = argv[3];
    uint32_t L = std::stoi(argv[4]), W = std::stoi(argv[5]), K = std::stoi(argv[6]), NT = std::stoi(argv[7]);
    std::string mode = argv[8], out = argv[9];
    int do_log = std::stoi(argv[10]);
    uint64_t warm = std::stoull(argv[11]);
    if (mode != "batch" && mode != "call")
    {
        std::cerr << "mode must be batch or call\n";
        return 1;
    }
    const bool batch_mode = mode == "batch";
    diskann::Metric metric = dist == "mips" ? diskann::Metric::INNER_PRODUCT : diskann::Metric::L2;

    T *query = nullptr;
    size_t nq, qdim, qadim;
    diskann::load_aligned_bin<T>(qfile, query, nq, qdim, qadim);
    std::vector<uint64_t> bptr;
    {
        std::ifstream bf(bfile, std::ios::binary | std::ios::ate);
        size_t bytes = bf.tellg();
        if (!bf || bytes % sizeof(uint64_t) != 0)
        {
            std::cerr << "bad batch file " << bfile << ": cannot open, or length not a multiple of 8 bytes\n";
            return 1;
        }
        bf.seekg(0);
        bptr.resize(bytes / 8);
        bf.read((char *)bptr.data(), bytes);
    }
    if (bptr.size() < 2 || bptr.front() != 0 || bptr.back() != nq || warm >= bptr.size() - 1)
    {
        std::cerr << "bad batch file: need u64 ptr[0..nb], ptr[0]=0, ptr[nb]=nq, warm < nb\n";
        return 1;
    }
    for (size_t b = 0; b + 1 < bptr.size(); b++)
        if (bptr[b + 1] <= bptr[b])
        {
            std::cerr << "empty or decreasing batch " << b << "\n";
            return 1;
        }
    const uint64_t nb_all = bptr.size() - 1, nb = nb_all - warm, q0 = bptr[warm], nqt = nq - q0;

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<T>> idx(new diskann::PQFlashIndex<T>(reader, metric));
    if (idx->load(NT, prefix.c_str()) != 0)
    {
        std::cerr << "load failed\n";
        return 2;
    }
    // no node cache is loaded: every expanded node is an uncached frontier node, so reads count whole scopes
    omp_set_num_threads(NT);
    std::vector<std::unique_ptr<diskann::BatchPageBuffer>> bufs(NT);
    for (uint32_t t = 0; t < NT; t++)
        bufs[t].reset(new diskann::BatchPageBuffer(diskann::defaults::SECTOR_LEN, diskann::defaults::SECTOR_LEN));

    std::vector<uint64_t> res(nq * K);
    std::vector<float> resd(nq * K);
    std::vector<diskann::QueryStats> stats(nq);
    std::vector<diskann::TraversalLog> logs(do_log ? nqt : 0);
    const uint64_t NC = 13;
    std::vector<uint64_t> rows(nb * NC, 0);

    auto run_batch = [&](uint64_t b, bool record) {
        int t = omp_get_thread_num();
        diskann::BatchPageBuffer *bb = batch_mode ? bufs[t].get() : nullptr;
        if (bb)
            bb->begin_batch();
        diskann::g_batch_buf = bb;
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = bptr[b]; i < bptr[b + 1]; i++)
        {
            if (record && do_log)
                diskann::g_trav_log = &logs[i - q0];
            idx->cached_beam_search(query + i * qadim, K, L, res.data() + i * K, resd.data() + i * K, W, false,
                                    &stats[i]);
            diskann::g_trav_log = nullptr;
        }
        auto t1 = std::chrono::steady_clock::now();
        diskann::g_batch_buf = nullptr;
        if (record)
        {
            uint64_t *r = rows.data() + (b - warm) * NC;
            if (bb)
            {
                r[0] = bb->page_reads;
                r[1] = bb->page_hits;
                r[2] = bb->inflight_dedup;
                r[3] = bb->calls;
                r[4] = bb->expanded;
                r[11] = bb->rounds;
                r[12] = bb->rounds_noread;
            }
            else
            {
                uint64_t s = 0;
                for (uint64_t i = bptr[b]; i < bptr[b + 1]; i++)
                    s += stats[i].n_ios;
                r[0] = s;
                r[3] = bptr[b + 1] - bptr[b];
                r[4] = s;
            }
            r[5] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            r[6] = (uint64_t)t;
            r[7] = (uint64_t)sched_getcpu();
            for (uint64_t i = bptr[b]; i < bptr[b + 1]; i++)
            {
                r[8] += stats[i].io_ns;
                r[9] += stats[i].pq_ns;
                r[10] += stats[i].nbr_ns;
            }
        }
        if (bb)
            bb->end_batch();
    };

    if (warm > 0)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t b = 0; b < (int64_t)warm; b++)
            run_batch((uint64_t)b, false);
    }
    const std::string stat0 = slurp("/proc/stat"), disk0 = slurp("/proc/diskstats");
    long long rb0 = proc_read_bytes();
    auto w0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b = (int64_t)warm; b < (int64_t)nb_all; b++)
        run_batch((uint64_t)b, true);
    auto w1 = std::chrono::steady_clock::now();
    long long rb1 = proc_read_bytes();
    const std::string stat1 = slurp("/proc/stat"), disk1 = slurp("/proc/diskstats");
    double secs = std::chrono::duration<double>(w1 - w0).count();
    {
        std::ofstream h(out + ".host");
        h << "@@stat0\n" << stat0 << "@@disk0\n" << disk0 << "@@stat1\n" << stat1 << "@@disk1\n" << disk1;
    }

    uint64_t tot_reads = 0, tot_hits = 0, tot_dedup = 0, tot_exp = 0, max_pages = 0, tot_rounds = 0, tot_noread = 0;
    double busy_ns = 0, io_ns = 0, pq_ns = 0, nbr_ns = 0;
    std::vector<double> lat(nb);
    std::set<uint64_t> cpus;
    for (uint64_t b = 0; b < nb; b++)
    {
        const uint64_t *r = rows.data() + b * NC;
        tot_reads += r[0];
        tot_hits += r[1];
        tot_dedup += r[2];
        tot_exp += r[4];
        lat[b] = r[5] * 1e-6;
        busy_ns += r[5];
        cpus.insert(r[7]);
        io_ns += r[8];
        pq_ns += r[9];
        nbr_ns += r[10];
        tot_rounds += r[11];
        tot_noread += r[12];
    }
    std::ostringstream cpul;
    for (uint64_t c : cpus)
        cpul << (cpul.tellp() > 0 ? "," : "") << c;
    for (auto &p : bufs)
        max_pages = std::max(max_pages, p->max_pages);
    std::vector<double> ls = lat;
    std::sort(ls.begin(), ls.end());
    auto pct = [&](double p) {
        double pos = p * (nb - 1);
        size_t lo = (size_t)pos;
        size_t hi = std::min(lo + 1, (size_t)nb - 1);
        return ls[lo] + (pos - lo) * (ls[hi] - ls[lo]);
    };

    {
        std::ofstream f(out + ".batches.bin", std::ios::binary);
        f.write((char *)&nb, 8);
        f.write((char *)rows.data(), 8 * rows.size());
    }
    {
        std::ofstream g(out + ".res.bin", std::ios::binary);
        uint32_t h2[2] = {(uint32_t)nqt, K};
        g.write((char *)h2, 8);
        std::vector<uint32_t> r32(nqt * K);
        for (size_t i = 0; i < nqt * K; i++)
            r32[i] = (uint32_t)res[q0 * K + i];
        g.write((char *)r32.data(), 4 * nqt * K);
        g.write((char *)(resd.data() + q0 * K), 4 * nqt * K);
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
        uint64_t hdr[4] = {nqt, tot_n, tot_r, tot_s};
        f.write((char *)hdr, sizeof(hdr));
        std::vector<uint64_t> noff(nqt + 1, 0), roff(nqt + 1, 0), soff(nqt + 1, 0);
        for (size_t i = 0; i < nqt; i++)
        {
            noff[i + 1] = noff[i] + logs[i].nodes.size();
            roff[i + 1] = roff[i] + logs[i].round_end.size();
            soff[i + 1] = soff[i] + logs[i].read_sectors.size();
        }
        f.write((char *)noff.data(), 8 * (nqt + 1));
        f.write((char *)roff.data(), 8 * (nqt + 1));
        f.write((char *)soff.data(), 8 * (nqt + 1));
        for (auto &l : logs)
            f.write((char *)l.nodes.data(), 4 * l.nodes.size());
        for (auto &l : logs)
            f.write((char *)l.parents.data(), 4 * l.parents.size());
        for (auto &l : logs)
            f.write((char *)l.round_end.data(), 4 * l.round_end.size());
        for (auto &l : logs)
            f.write((char *)l.read_sectors.data(), 4 * l.read_sectors.size());
        std::vector<uint32_t> nios(nqt), nhops(nqt), nch(nqt);
        std::vector<float> iou(nqt);
        for (size_t i = 0; i < nqt; i++)
        {
            nios[i] = stats[q0 + i].n_ios;
            nhops[i] = stats[q0 + i].n_hops;
            nch[i] = stats[q0 + i].n_cache_hits;
            iou[i] = stats[q0 + i].io_us;
        }
        f.write((char *)nios.data(), 4 * nqt);
        f.write((char *)nhops.data(), 4 * nqt);
        f.write((char *)nch.data(), 4 * nqt);
        f.write((char *)iou.data(), 4 * nqt);
    }
    std::cout << "summary mode " << mode << " sector_len " << diskann::defaults::SECTOR_LEN << " threads " << NT
              << " L " << L << " W " << W << " K " << K << " cache_nodes 0 log " << do_log << " warm_batches " << warm
              << " batches " << nb << " calls " << nqt << " secs " << secs << " batches_per_s " << nb / secs
              << " calls_per_s " << nqt / secs << " page_reads " << tot_reads << " page_hits " << tot_hits
              << " inflight_dedup " << tot_dedup << " expanded " << tot_exp << " proc_read_bytes " << (rb1 - rb0)
              << " bytes_per_read " << (tot_reads ? (double)(rb1 - rb0) / tot_reads : 0.0) << " lat_ms_median "
              << pct(0.5) << " lat_ms_p99 " << pct(0.99) << " lat_ms_mean "
              << std::accumulate(lat.begin(), lat.end(), 0.0) / nb << " max_buffer_pages " << max_pages
              << " busy_thread_s " << busy_ns * 1e-9 << " io_share " << io_ns / busy_ns << " pq_share "
              << pq_ns / busy_ns << " nbr_share " << nbr_ns / busy_ns << " other_share "
              << 1.0 - (io_ns + pq_ns + nbr_ns) / busy_ns << " rounds " << tot_rounds << " rounds_noread " << tot_noread
              << " cpus " << cpul.str() << std::endl;
    diskann::aligned_free(query);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 14)
    {
        std::cerr << "usage: batch_search <float|int8> <l2|mips> <index_prefix> <query.bin> <batch_ptr.u64> <L> <W> "
                     "<K> <threads> <batch|call> <out_prefix> <log 0|1> <warm_batches>\n";
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
