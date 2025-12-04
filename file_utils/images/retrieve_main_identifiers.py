#!/usr/bin/env python3
"""
Retrieve and normalize main identifiers from image filename bases.

This script processes a CSV file containing image filename bases and extracts
normalized main identifiers by:
- Replacing split markers: (1)% -> split_1, (2)% -> split_2, -2-%(stitched) -> split_2
- Removing stitched markers: (stitched) -> removed
- Normalizing case: Ba -> BA
- Stripping trailing '%' characters

The normalized identifiers are sorted and saved to a JSON file.

Usage:
    python retrieve_main_identifiers.py --csv-file input.csv --out-file output.json

Example:
    Input filename base: "Ba4 96_1 Dy28 C12(1)%"
    Output identifier: "BA4 96_1 Dy28 C12 split_1"
"""
import argparse
import json
import logging
import pathlib

import pandas as pd

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Retrieve main identifiers from a CSV file')
    parser.add_argument('--csv-file', type=pathlib.Path, help='The CSV file to retrieve main identifiers from')
    parser.add_argument('--out-file', type=pathlib.Path, help='The file to save the main identifiers to')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    filename_bases = df['filename base'].tolist()

    main_ids = []
    for fb in filename_bases:
        # Check for specific patterns first (before general patterns)
        if '-2-%(stitched)' in fb:
            fb = fb.replace('-2-%(stitched)', ' split_2')
            logging.debug(f"Matched -2-%(stitched): {fb}")
        elif '(1)%' in fb:
            fb = fb.replace('(1)%', ' split_1')
        elif '(2)%' in fb:
            fb = fb.replace('(2)%', ' split_2')
        elif '(stitched)' in fb:
            fb = fb.replace('(stitched)', '')

        fb = fb.replace('Ba', 'BA')
        fb = fb.rstrip('%')
        main_ids.append(fb)

    logging.info(f"Found {len(main_ids)} main identifiers")

    main_ids.sort()
    with open(args.out_file, 'w') as jf:
        json.dump(main_ids, jf, indent=2)
    logging.info(f"Saved main identifiers to: {args.out_file}")


if __name__ == '__main__':
    main()