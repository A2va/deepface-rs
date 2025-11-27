#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "pandas>=2.3",
# ]
# ///
#
# Learn more about Astral's UV tool at
# https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies
#
# Documentation:
#   Build the master.csv file for a face recognition dataset.
#
#   Input:
#       A folder structured as:
#           dataset_root/
#               person1/
#                   img1.jpg
#                   img2.jpg
#               person2/
#                   imgA.jpg
#                   imgB.jpg
#   Output:
#       master.csv in the same folder, containing:
#         - All positive pairs  (same identity)
#         - All negative pairs  (different identities)
#         - A "decision" column ("Yes" = same person, "No" = different person)
#
#   Usage:
#       uv run generate-csv dataset_root/
#
#   This CSV is later used by the generate-model-distance rust binary.

import argparse
import itertools
from pathlib import Path

import pandas as pd


def main(args):
    folder = Path(args.folder)

    idendities = {}

    for item in folder.iterdir():
        if not item.is_dir():
            continue

        idendities[item.name] = [
            str(file.relative_to(folder)) for file in folder.joinpath(item).iterdir()
        ]
    print(idendities)

    positives = []

    for key, values in idendities.items():
        for i in range(0, len(values) - 1):
            for j in range(i + 1, len(values)):
                positive = []
                positive.append(values[i])
                positive.append(values[j])
                positives.append(positive)

    positives = pd.DataFrame(positives, columns=["file_x", "file_y"])
    positives["decision"] = "1"

    samples_list = list(idendities.values())
    negatives = []

    for i in range(0, len(idendities) - 1):
        for j in range(i + 1, len(idendities)):
            cross_product = itertools.product(samples_list[i], samples_list[j])
            cross_product = list(cross_product)

            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                negatives.append(negative)

    negatives = pd.DataFrame(negatives, columns=["file_x", "file_y"])
    negatives["decision"] = "0"
    df = pd.concat([positives, negatives]).reset_index(drop=True)

    df.to_csv(folder.joinpath("master.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates master.csv for a dataset",
    )

    parser.add_argument("folder")

    args = parser.parse_args()
    main(args)
