"""Data read/write and io functions"""

import csv

def read_tsv(input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))