"""
Modified version of the following script
https://github.com/LPDI-EPFL/masif/issues/70#issuecomment-1823238474
but without filtering by the predicted interface probability, this script
simply matches the MaSIF vertices to PDB residues.

It also has a more "manual" interface that is independent of the "job name"
used by MaSIF.

Usage:

python map_vertices_to_residues.py \
       --pdb-file 1cz8_V.pdb \
       --precomputation-directory /path/to/masif/precomputation/1cz8_V \
       --radius 1.2 \
       --output-file 1cz8_V.mapped

Output format is a TSV file with columns:
    - vertex-index
    - x
    - y
    - z
    - chain_id:residue_name:residue_number[:insertion]
    - distance between atom and vertex

"""


from pathlib import Path
from typing import List
import numpy as np
import numpy.typing as npt
from biopandas.pdb import PandasPdb
from scipy.spatial import KDTree


def match_atoms(target_coords: npt.NDArray,
                query_coords: npt.NDArray,
                radius: float) -> List[int]:
    """
    Find atoms that are within a radius from the target vertices

    Parameters
    ----------
    target_coords : np.ndarray shape (N, 3)
        coordinates of the vertices
    query_coords : np.ndarray shape (N, 3)
        coordinates of the atoms
    radius : float
        radius in Å cutoff to retrieve atoms close to vertices

    Returns
    -------
    idx : List[int]
        indices of the atoms in `query_coords` that fall within a radius from
        the vertices
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(query_coords)  # indexing the atom coordinates
    # get atoms that are within a radius from the vertices
    idx = tree.query_ball_point(target_coords, r=radius)
    # flatten the list of lists
    idx = [item for sublist in idx for item in sublist]
    # remove duplicates
    idx = list(set(idx))

    return idx


def run(pdb_file: Path,
        precomputation_dir: Path,
        radius: float,
        output_file: Path):
    assert pdb_file.is_file()
    assert precomputation_dir.is_dir()

    pdb = PandasPdb().read_pdb(str(pdb_file))
    atoms = pdb.df["ATOM"]
    # add node_id in the format of
    # [chain_id]:[residue_name]:[residue_number]:[insertion]
    atoms['node_id'] = (atoms['chain_id'] + ':'
                        + atoms['residue_name'] + ':'
                        + atoms['residue_number'].astype(str) + ':'
                        + atoms['insertion'])
    # remove the tailing space and colon in the node_id if insertion is empty
    atoms['node_id'] = atoms['node_id'].str.replace(r':\s*$', '', regex=True)

    atom_coords = atoms.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy()

    p1_X = np.load(precomputation_dir / 'p1_X.npy')
    p1_Y = np.load(precomputation_dir / 'p1_Y.npy')
    p1_Z = np.load(precomputation_dir / 'p1_Z.npy')

    vertices = np.concatenate([p1_X.reshape(-1, 1),
                               p1_Y.reshape(-1, 1),
                               p1_Z.reshape(-1, 1)], axis=1)

    # match vertices to atoms

    tree = KDTree(atom_coords)  # indexing the atom coordinates
    # get atoms that are within a radius from the vertices
    dists, idx = tree.query(vertices, k=1)

    with output_file.open("w") as of:
        of.write("vertex index\tx\ty\tz"
                 "\tchain_id:residue_name:residue_number[:insertion]"
                 "\tdistance\n")
        for i, v in enumerate(vertices):
            dist = dists[i]
            node_idx = idx[i]
            if radius > 0 and dist > radius:
                continue
            of.write(f"{i}\t{v[0]}\t{v[1]}\t{v[2]}"
                     f"\t{atoms['node_id'][node_idx]}\t{dist}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Map a MaSIF mesh to the closest aminoacids in a PDB")
    parser.add_argument("-p", "--pdb-file", required=True,
                        help="Path to the PDB file")
    parser.add_argument("-d", "--precomputation-directory", required=True,
                        help="Path to the MaSIF precomputation directory")
    parser.add_argument("-r", "--radius", type=float, default=-1,
                        help="radius in Å to retrieve atoms close to vertices")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Path to the output file")
    args = parser.parse_args()
    run(Path(args.pdb_file),
        Path(args.precomputation_directory),
        args.radius,
        Path(args.output_file))
