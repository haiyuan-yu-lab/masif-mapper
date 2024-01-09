import requests


def download_pdb(pdb_id_with_chain, output_directory):
    try:
        pdb_id, chain = pdb_id_with_chain.split('_')
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)

        if response.status_code == 200:
            file_path = f'{output_directory}/{pdb_id_with_chain}.pdb'
            with open(file_path, 'w') as file:
                file.write(response.text)
            return f"PDB file {pdb_id_with_chain} saved as {file_path}"
        else:
            return (f"Error: Unable to download PDB file {pdb_id_with_chain}."
                    f" Status code {response.status_code}.")
    except Exception as e:
        return f"Error: {str(e)}"


def main(input_file, output_directory):
    with open(input_file, 'r') as file:
        pdb_ids_with_chains = file.read().splitlines()

    for pdb_id_with_chain in pdb_ids_with_chains:
        result = download_pdb(pdb_id_with_chain, output_directory)
        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download PDB files from a list of PDB IDs with chains.")
    parser.add_argument("input_file", type=str,
                        help="File with PDB IDs + chains (one per line).")
    parser.add_argument("output_directory", type=str,
                        help="Directory to save the downloaded PDB files.")

    args = parser.parse_args()
    main(args.input_file, args.output_directory)
