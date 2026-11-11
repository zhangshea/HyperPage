"""Anchored text edits of a source tree, shared by the edit scripts of this directory: exact anchors, and regular
expressions on identifier names. An anchor found a different number of times than expected stops the script."""
import os
import re
import shutil
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
APP_LINK = '${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options'


def read(root, rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


def _write(root, rel, s):
    with open(os.path.join(root, rel), 'w') as f:
        f.write(s)


def replace(root, rel, anchor, new, count=1):
    """Replace every occurrence of anchor, which must occur exactly count times."""
    s = read(root, rel)
    n = s.count(anchor)
    if n != count:
        sys.exit(f'{rel}: anchor found {n} times, expected {count}: {anchor[:70]!r}')
    _write(root, rel, s.replace(anchor, new))


def insert_after(root, rel, anchor, text, count=1):
    replace(root, rel, anchor, anchor + text, count)


def insert_after_first(root, rel, anchor, text):
    """Insert text after the first occurrence of anchor (the anchor may occur more than once)."""
    s = read(root, rel)
    if anchor not in s:
        sys.exit(f'{rel}: anchor not found: {anchor[:70]!r}')
    _write(root, rel, s.replace(anchor, anchor + text, 1))


def insert_after_from(root, rel, marker, anchor, text):
    """Insert text after the first occurrence of anchor that follows the first occurrence of marker."""
    s = read(root, rel)
    m = s.find(marker)
    i = s.find(anchor, m) if m >= 0 else -1
    if i < 0:
        sys.exit(f'{rel}: anchor {anchor[:50]!r} not found after {marker!r}')
    i += len(anchor)
    _write(root, rel, s[:i] + text + s[i:])


def find_once(root, rel, pattern, what):
    """Text of the file and the match of a multi-line regular expression; stops unless it matches exactly once."""
    s = read(root, rel)
    ms = list(re.finditer(pattern, s, re.M))
    if len(ms) != 1:
        sys.exit(f'{rel}: {what}: pattern matched {len(ms)} times, expected exactly 1; the tree is not at the pinned '
                 f'commit or was edited already')
    return s, ms[0]


def splice(root, rel, s, start, end, text):
    """Write s with s[start:end] replaced by text."""
    _write(root, rel, s[:start] + text + s[end:])


def prepend_if_missing(root, rel, line):
    s = read(root, rel)
    if line not in s:
        _write(root, rel, line + '\n' + s)


def copy_source(root, name, dest_dir):
    """Copy one of our source files (engine/src) into the tree."""
    shutil.copy(os.path.join(SRC, name), os.path.join(root, dest_dir, name))


def add_app(root, target, source):
    """Append an application target to apps/CMakeLists.txt, linked like DiskANN's own applications."""
    s = read(root, 'apps/CMakeLists.txt')
    if f'add_executable({target} ' in s:
        sys.exit(f'apps/CMakeLists.txt: target {target} exists already')
    _write(root, 'apps/CMakeLists.txt',
           s + f'\nadd_executable({target} {source})\ntarget_link_libraries({target} {APP_LINK})\n')
