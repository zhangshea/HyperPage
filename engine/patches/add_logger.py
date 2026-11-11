"""Add the traversal logger to a DiskANN checkout and the logging search application (search_and_log) to its build.

Every logger statement sits behind `if (tl)`; with no log installed the stock search runs unchanged."""
# DiskANN is MIT-licensed (https://github.com/microsoft/DiskANN); the anchors below are excerpts of its
# src/pq_flash_index.cpp at the pinned commit, used only to locate the edits.
import argparse

import edit_util as E

CPP = 'src/pq_flash_index.cpp'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='DiskANN source tree (cpp_main at the pinned commit)')
    T = ap.parse_args().tree
    if 'traversal_log.h' in E.read(T, CPP):
        raise SystemExit(f'{CPP}: the logger is present already')

    E.copy_source(T, 'traversal_log.h', 'include')
    E.insert_after_first(T, CPP, '#include "pq_flash_index.h"\n', '#include "traversal_log.h"\n')
    E.insert_after_first(T, CPP, 'namespace diskann\n{\n', 'thread_local TraversalLog *g_trav_log = nullptr;\n')
    # the log of one call starts after the entry point enters the candidate list
    E.insert_after(T, CPP, "    compute_dists(&best_medoid, 1, dist_scratch);\n"
                           "    retset.insert(Neighbor(best_medoid, dist_scratch[0]));\n    "
                           "visited.insert(best_medoid);\n",
                   "    TraversalLog *tl = g_trav_log;\n    tsl::robin_map<uint32_t, uint32_t> parent_of;\n    if "
                   "(tl)\n    {\n"
                   "        tl->clear();\n        parent_of[best_medoid] = std::numeric_limits<uint32_t>::max();\n   "
                   " }\n")
    # every node selected for the beam is an expanded node, with the node that first discovered it
    E.insert_after(T, CPP, "            auto nbr = retset.closest_unexpanded();\n            num_seen++;\n",
                   "            if (tl)\n            {\n                tl->nodes.push_back(nbr.id);\n"
                   "                auto pit = parent_of.find(nbr.id);\n"
                   "                tl->parents.push_back(pit == parent_of.end() ? "
                   "std::numeric_limits<uint32_t>::max() - 1 "
                   ": pit->second);\n            }\n")
    # a round ends once its beam is formed
    E.replace(T, CPP, "        // read nhoods of frontier ids\n        if (!frontier.empty())\n",
              "        if (tl)\n            tl->round_end.push_back((uint32_t)tl->nodes.size());\n"
              "        // read nhoods of frontier ids\n        if (!frontier.empty())\n")
    # the sector of every issued read
    E.replace(T, CPP, "                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * "
                      "defaults::SECTOR_LEN,\n",
              "                if (tl)\n                    "
              "tl->read_sectors.push_back((uint32_t)get_node_sector((size_t)id));\n"
              "                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,\n")
    # parents: the first insertion into `visited`, from a cached record or from a record read this round
    visit = ("                uint32_t id = node_nbrs[m];\n                if (visited.insert(id).second)\n          "
             "      {\n")
    E.insert_after_from(T, CPP, '// process cached nhoods', visit,
                        "                    if (tl)\n                        parent_of[id] = "
                        "(uint32_t)cached_nhood.first;\n")
    E.insert_after_from(T, CPP, '// compute node_nbrs <-> query dist in PQ space', visit,
                        "                    if (tl)\n                        parent_of[id] = "
                        "(uint32_t)frontier_nhood.first;\n")

    E.copy_source(T, 'search_and_log.cpp', 'apps')
    E.add_app(T, 'search_and_log', 'search_and_log.cpp')
    print('logger added to', T)


if __name__ == '__main__':
    main()
