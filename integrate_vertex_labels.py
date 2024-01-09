"""
When you have mapped the MaSIF vertex to a pdb file, this script integrates
the predictions associated to each verted into the generated mapping file.

Usage:
python integrate_vertex_labels.py \
       --mapping-file 1cz8_V.mapped \
       --feature-file /path/to/masif/output/pred_1cz2_V.npy \
       --output-file 1cz8_V.pred
"""

from pathlib import Path
import pandas as pd
import numpy as np


def run(mapping_file: Path,
        feature_file: Path,
        output_file: Path,
        feature_name: str = "pred") -> None:
    assert mapping_file.is_file()
    assert feature_file.is_file()
    vertex_map = pd.read_table(mapping_file)
    features = np.load(feature_file).flatten()
    assert vertex_map.shape[0] == features.shape[0]
    vertex_map[feature_name] = features
    vertex_map.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Integrates a MaSIF feature npy file with a vertex map.")
    parser.add_argument("-p", "--mapping-file", required=True,
                        help="Path to a vertex mapping file.")
    parser.add_argument("-f", "--feature-file", required=True,
                        help="Path to a feature file compatible with the "
                             "mapping file.")
    parser.add_argument("-n", "--feature-name", type=str,
                        default="pred",
                        help="Name to use as the feature column name.")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Path to the output file")
    args = parser.parse_args()
    run(Path(args.mapping_file),
        Path(args.feature_file),
        Path(args.output_file),
        args.feature_name)
