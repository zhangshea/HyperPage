"""Add the arrival-order log to a copy of the hybrid-cache tree: per search round one record (sequence, call, round of
the call, nodes expanded in the round), the sequence taken inside the cache mutex when the round touches the LRU."""
# Logging only; the search path is unchanged. Enabled by setting the environment variable HYB_ORDER_LOG when running
# hybrid_search; records go to <out>.order as u64 quadruples. All anchors are lines of our own sources and edits.
import argparse

import edit_util as E

H = 'include/hybrid_page_cache.h'
CPP = 'src/pq_flash_index.cpp'
APP = 'apps/hybrid_search.cpp'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='copy of the hybrid-cache tree (add_hybrid_cache.py applied)')
    T = ap.parse_args().tree

    E.replace(T, H, "    int32_t head = -1, tail = -1;\n    std::mutex mu;\n",
              "    int32_t head = -1, tail = -1;\n    std::mutex mu;\n"
              "    std::atomic<uint64_t> oseq{0}; // global round sequence (taken under mu when a round touches the "
              "LRU)\n")
    E.replace(T, H, "    uint64_t wait_ns = 0;      // time spent waiting for other threads' in-flight pages\n};",
              "    uint64_t wait_ns = 0;      // time spent waiting for other threads' in-flight pages\n"
              "    bool log_order = false;    // arrival-order log (logging only)\n"
              "    uint64_t call = 0, call_round0 = 0;\n"
              "    std::vector<uint64_t> olog; // (sequence, call, round of the call, nodes expanded in the round) "
              "per round\n};")
    E.replace(T, CPP, "            hy->cached += cached_nhoods.size();\n        }\n",
              "            hy->cached += cached_nhoods.size();\n"
              "            if (hy->log_order && frontier.empty())\n"
              "            {   // a round served by the node cache only; it makes no LRU request\n"
              "                const uint64_t q = hy->pc->oseq.fetch_add(1, std::memory_order_relaxed);\n"
              "                hy->olog.insert(hy->olog.end(), {q, hy->call, hy->rounds - hy->call_round0 - 1, "
              "(uint64_t)cached_nhoods.size()});\n"
              "            }\n"
              "        }\n")
    E.replace(T, CPP, "                    std::lock_guard<std::mutex> lk(hy->pc->mu);\n"
                      "                    for (uint64_t i = 0; i < frontier.size(); i++)\n",
              "                    std::lock_guard<std::mutex> lk(hy->pc->mu);\n"
              "                    if (hy->log_order)\n"
              "                    {   // the sequence is taken under the cache mutex, so it orders the LRU "
              "operations\n"
              "                        const uint64_t q = hy->pc->oseq.fetch_add(1, std::memory_order_relaxed);\n"
              "                        hy->olog.insert(hy->olog.end(), {q, hy->call, hy->rounds - hy->call_round0 - "
              "1,\n"
              "                                                         (uint64_t)(frontier.size() + "
              "cached_nhoods.size())});\n"
              "                    }\n"
              "                    for (uint64_t i = 0; i < frontier.size(); i++)\n")
    E.replace(T, APP, "        c.stock_order = order == \"stock\";\n",
              "        c.stock_order = order == \"stock\";\n"
              "        c.log_order = std::getenv(\"HYB_ORDER_LOG\") != nullptr; // arrival-order log on request\n")
    E.replace(T, APP, "            diskann::g_hyb = &c;\n",
              "            c.call = (uint64_t)i;\n            c.call_round0 = c.rounds;\n            diskann::g_hyb ="
              " &c;\n")
    E.replace(T, APP, "    {\n        std::ofstream h(out + \".host\");\n",
              "    if (std::getenv(\"HYB_ORDER_LOG\") != nullptr)\n"
              "    {   // arrival-order records of all threads\n"
              "        std::ofstream f(out + \".order\", std::ios::binary);\n"
              "        for (auto &c : ctx)\n"
              "            f.write((const char *)c.olog.data(), (std::streamsize)(c.olog.size() * sizeof(uint64_t)));\n"
              "    }\n"
              "    {\n        std::ofstream h(out + \".host\");\n")
    E.prepend_if_missing(T, APP, '#include <cstdlib>')
    print('arrival-order log added to', T)


if __name__ == '__main__':
    main()
