"""Add the node-to-(page, slot) layout map: loaded from <disk index>.layout when that file exists, it sets
get_node_sector(id) = 1 + page[id] and offset_to_node(buf, id) = buf + slot[id] * max_node_len."""
# Without the .layout file the stock code path runs. DiskANN is MIT-licensed (https://github.com/microsoft/DiskANN);
# the anchors below are excerpts of its pq_flash_index.{h,cpp} at the pinned commit, used to locate the edits.
import argparse

import edit_util as E

H = 'include/pq_flash_index.h'
CPP = 'src/pq_flash_index.cpp'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='DiskANN source tree with the traversal logger (add_logger.py)')
    T = ap.parse_args().tree
    if '_has_layout' in E.read(T, H):
        raise SystemExit(f'{H}: the layout map is present already')

    E.insert_after(T, H, "    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors"
                         "\n",
                   "    // node -> (page, slot) map loaded from <disk index>.layout (absent = stock layout)\n"
                   "    bool _has_layout = false;\n    std::vector<uint32_t> _layout_page, _layout_slot;\n")
    # [Paper] Section VI-E: a relayout stores a map from node to page and slot; page p starts at sector 1 + p
    E.replace(T, CPP, """    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));""",
              """    if (_has_layout)
        return 1 + (uint64_t)_layout_page[node_id];
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));""")
    E.replace(T, CPP, "    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * "
                      "_max_node_len);",
              """    if (_has_layout)
        return sector_buf + (uint64_t)_layout_slot[node_id] * _max_node_len;
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);""")
    E.replace(T, CPP, """    diskann::cout << "Disk-Index File Meta-data: ";""",
              """    {   // layout map (absent = stock layout): u64 n, u32 page[n], u32 slot[n]
        std::string lf = std::string(_disk_index_file) + ".layout";
        std::ifstream lin(lf, std::ios::binary);
        if (lin.good())
        {
            uint64_t ln = 0;
            lin.read((char *)&ln, sizeof(uint64_t));
            if (ln != this->_num_points || _nnodes_per_sector == 0)
                throw ANNException("relayout map: size mismatch or multi-sector nodes", -1, __FUNCSIG__, __FILE__, __LINE__);
            _layout_page.resize(ln);
            _layout_slot.resize(ln);
            lin.read((char *)_layout_page.data(), sizeof(uint32_t) * ln);
            lin.read((char *)_layout_slot.data(), sizeof(uint32_t) * ln);
            if (!lin)
                throw ANNException("relayout map: truncated file", -1, __FUNCSIG__, __FILE__, __LINE__);
            for (uint64_t i = 0; i < ln; i++)
                if (_layout_slot[i] >= _nnodes_per_sector)
                    throw ANNException("relayout map: slot out of range", -1, __FUNCSIG__, __FILE__, __LINE__);
            _has_layout = true;
            diskann::cout << "Loaded relayout map " << lf << " (" << ln << " nodes)" << std::endl;
        }
    }
    diskann::cout << "Disk-Index File Meta-data: ";""")
    E.prepend_if_missing(T, CPP, '#include <fstream>')
    print('layout map added to', T)


if __name__ == '__main__':
    main()
