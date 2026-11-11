"""Lower the per-thread asynchronous I/O queue depth (MAX_EVENTS) in a copy of a source tree, so that 64 to 256 threads
fit under the kernel's aio-max-nr limit; the search path is unchanged."""
# Used for the runs with 64, 128 and 256 concurrent sub-queries (Section VII-B, MARGO's page-search engine).
# The edit is located by a regular expression that names the macro MAX_EVENTS in src/linux_aligned_file_reader.cpp
# (DiskANN's tree and MARGO's tree); the script stops unless it matches exactly once, and the whole definition line is
# replaced by ours.
import argparse

import edit_util as E

READER = 'src/linux_aligned_file_reader.cpp'
EVENTS_DEF = r'^[ \t]*#[ \t]*define[ \t]+MAX_EVENTS\b[^\n]*$'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', help='copy of the hybrid-cache tree or of the page-search tree')
    ap.add_argument('--events', type=int, default=128, help='queue depth per thread')
    a = ap.parse_args()
    s, m = E.find_once(a.tree, READER, EVENTS_DEF, 'definition of MAX_EVENTS')
    E.splice(a.tree, READER, s, m.start(), m.end(),
             f'#define MAX_EVENTS {a.events}   // queue depth per thread, lowered so that many threads fit')
    print('MAX_EVENTS set to', a.events, 'in', a.tree)


if __name__ == '__main__':
    main()
