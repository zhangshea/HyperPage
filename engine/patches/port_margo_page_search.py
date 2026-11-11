"""Port MARGO's page-search engine to 8 KB pages, with an optional fixed record cache list and one summary line per L.

Edits a local checkout of MARGO's repository; nothing of MARGO is distributed here."""
# MARGO's repository states no licence, so this script holds no line of its source. Each edit is located by a regular
# expression made of identifier names and the punctuation that delimits them (the definition of the macro SECTOR_LEN,
# the call of the member function generate_cache_list_from_sample_queries, the declaration of the variable recall,
# the add_subdirectory command of graph_partition), and the script stops unless each pattern matches exactly once.
# The inserted code is ours; it reads identifiers of MARGO's tests/search_disk_index.cpp (node_list,
# num_nodes_to_cache, query_num, stats, L, optimized_beamwidth, diff, qps, cache_mem, load_mem) and DiskANN helpers
# (get_percentile_stats, getProcessPeakRSS, QueryStats).
# [Paper] Section VII-B, MARGO's page-search engine: 8 KB pages, a record cache of the whole budget ranked by training
# frequency (the list given in RECORD_CACHE_LIST, written by page_search/cache_list.py).
import argparse

import edit_util as E

HEADER = 'include/pq_flash_index.h'
SEARCH = 'tests/search_disk_index.cpp'
PAGE_BYTES = 8192   # B = 12 records of 644 bytes per page

# the definition of the macro SECTOR_LEN; group 2 is its trailing integer (the page size)
SECTOR_DEF = r'^([ \t]*#[ \t]*define[ \t]+SECTOR_LEN\b[^\n]*?)(\d+)[ \t]*$'
# the start of the statement that calls the member function generate_cache_list_from_sample_queries; group 1 = indent
CACHE_CALL = r'^([ \t]*)\w+[ \t]*->[ \t]*generate_cache_list_from_sample_queries[ \t]*\('
# the declaration of the variable recall with its initialiser; group 1 = indent
RECALL_DECL = r'^([ \t]*)\w+[ \t]+recall[ \t]*='
# the add_subdirectory command of the graph_partition directory; group 1 = indent
PARTITIONER = r'^([ \t]*)add_subdirectory[ \t]*\([ \t]*graph_partition[ \t]*\)'

# fixed cache list (u32 record ids) read when RECORD_CACHE_LIST is set; the trailing `else` makes MARGO's own cache
# selection, which follows it, the other branch
CACHE_BLOCK = ('const char* hp_cache = std::getenv("RECORD_CACHE_LIST");   // fixed cache list (u32 ids)\n'
               '    if (hp_cache != nullptr) {\n'
               '      std::ifstream cf(hp_cache, std::ios::binary); uint32_t id;\n'
               '      while (node_list.size() < num_nodes_to_cache && cf.read((char*) &id, sizeof(id))) '
               'node_list.push_back(id);\n'
               '      diskann::cout << "RECORD_CACHE_LIST " << hp_cache << ": " << node_list.size() << " nodes"'
               ' << std::endl;\n'
               '    } else\n')

# one summary line per L: total page reads, seconds, QPS, p50 and p99 latency, peak resident memory
HPSTAT_BLOCK = ('{   // exact totals and percentiles for the comparison with the hybrid engine\n'
                '      uint64_t sum_ios = 0;\n'
                '      for (_s64 i = 0; i < (int64_t) query_num; i++) sum_ios += stats[i].n_ios;\n'
                '      auto p50 = diskann::get_percentile_stats<float>(stats, query_num, 0.5, '
                '[](const diskann::QueryStats& st) { return st.total_us; });\n'
                '      auto p99 = diskann::get_percentile_stats<float>(stats, query_num, 0.99, '
                '[](const diskann::QueryStats& st) { return st.total_us; });\n'
                '      diskann::cout << "HPSTAT L " << L << " W " << optimized_beamwidth << " nq " << query_num '
                '<< " sum_ios " << sum_ios\n'
                '                    << " secs " << diff.count() << " qps " << qps << " p50_us " << p50 '
                '<< " p99_us " << p99\n'
                '                    << " peak_rss_mb " << getProcessPeakRSS() << " cache_rss " << cache_mem '
                '<< " load_rss " << load_mem << std::endl;\n'
                '    }\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='copy of a MARGO checkout at the pinned commit')
    T = ap.parse_args().tree

    if 'RECORD_CACHE_LIST' in E.read(T, SEARCH):
        raise SystemExit(f'{SEARCH}: the page-search port is present already')
    # locate every edit first, so that a tree that does not match leaves no file half edited
    for rel, pat, what in ((HEADER, SECTOR_DEF, 'definition of SECTOR_LEN'),
                           (SEARCH, CACHE_CALL, 'call of generate_cache_list_from_sample_queries'),
                           (SEARCH, RECALL_DECL, 'declaration of recall'),
                           ('CMakeLists.txt', PARTITIONER, 'add_subdirectory command of graph_partition')):
        E.find_once(T, rel, pat, what)

    # 1. page size: the integer of the SECTOR_LEN definition becomes 8192
    s, m = E.find_once(T, HEADER, SECTOR_DEF, 'definition of SECTOR_LEN')
    if m.group(2) != '4096':
        raise SystemExit(f'{HEADER}: SECTOR_LEN is {m.group(2)}, expected the stock 4096')
    E.splice(T, HEADER, s, m.start(2), m.end(2), str(PAGE_BYTES))

    # 2. fixed record cache list, inserted in front of the statement that calls MARGO's cache selection
    s, m = E.find_once(T, SEARCH, CACHE_CALL, 'call of generate_cache_list_from_sample_queries')
    E.splice(T, SEARCH, s, m.end(1), m.end(1), CACHE_BLOCK + m.group(1))

    # 3. HPSTAT summary line, inserted in front of the declaration of recall (after the timed search of one L)
    s, m = E.find_once(T, SEARCH, RECALL_DECL, 'declaration of recall')
    E.splice(T, SEARCH, s, m.end(1), m.end(1), HPSTAT_BLOCK + m.group(1))
    E.prepend_if_missing(T, SEARCH, '#include <cstdlib>')
    E.prepend_if_missing(T, SEARCH, '#include <fstream>')

    # 4. build without the graph partitioner subdirectory, which the page-search binaries do not use
    s, m = E.find_once(T, 'CMakeLists.txt', PARTITIONER, 'add_subdirectory command of graph_partition')
    E.splice(T, 'CMakeLists.txt', s, m.end(1), m.end(1), '# ')
    print('page-search port applied to', T)


if __name__ == '__main__':
    main()
