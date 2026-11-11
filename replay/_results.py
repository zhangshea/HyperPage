"""Loading of replay result files for the selection and table scripts.
Helper module of the scripts in replay/, imported by name: Python puts a script's directory on the import path."""
import json


def load_rows(paths):
    """Merge the rows of several replay files; a row that occurs twice must agree."""
    rows = {}
    meta = None
    for path in paths:
        data = json.load(open(path))
        meta = meta or data
        for key, row in data['rows'].items():
            if key in rows:
                assert rows[key]['per_step'] == row['per_step'], (path, key)
            rows[key] = row
    return rows, meta


def layouts_in(rows, budget):
    return sorted({key.split('|')[0] for key in rows if key.split('|')[2] == str(budget)})
