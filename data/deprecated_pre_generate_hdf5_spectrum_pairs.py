import argparse
import gc
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def generate_pairs_hdf5(pairs_path, output_path):
    print("Reading pairs file", flush=True)
    pairs = pd.read_feather(pairs_path)
    print("Finished reading pairs file", flush=True)
    
    # Check if pairs is symmetric
    # pair_set = set(map(tuple, pairs[['spectrum_id_1', 'spectrum_id_2']].values))
    # reverse_pair_set = set(map(tuple, pairs[['spectrum_id_2', 'spectrum_id_1']].values))
    # if pair_set != reverse_pair_set:
    if True:
        print("Pairs file is not symmetric. Making it symmetric.", flush=True)
        pairs = pd.concat([pairs, pairs.rename(columns={'spectrum_id_1':'spectrum_id_2',
                                                                'spectrum_id_2':'spectrum_id_1',})], axis=0)
        print("Pairs are now symmetric.", flush=True)
    # del pair_set, reverse_pair_set
    gc.collect()

    # Make an hdf5 file that's indexible by inchikey_1
    pairs = pairs.groupby('inchikey_1')
    
    # If the output file already exists, delete it
    if output_path.exists():
        output_path.unlink()

    with pd.HDFStore(output_path, mode='w') as store:
        for inchikey_1, group in tqdm(pairs, desc="Writing groups to HDF5"):
            store.put(f'{inchikey_1}', group, format='table')
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Generate hdf5 file for pairs")
    parser.add_argument("--pairs_path", type=str, help="Path to pairs file")
    parser.add_argument("--output_path", type=str, help="Path to output hdf5 file")
    args = parser.parse_args()

    pairs_path = Path(args.pairs_path)
    output_path = Path(args.output_path)

    # Validate input/output paths/dirs
    if not pairs_path.is_file():
        raise ValueError(f"Path {pairs_path} is not a file")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_pairs_hdf5(pairs_path, output_path)

if __name__ == "__main__":
    main()