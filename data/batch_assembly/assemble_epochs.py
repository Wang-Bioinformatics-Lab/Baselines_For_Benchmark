import argparse
from pathlib import Path
import pandas as pd
from glob import glob
from tqdm import tqdm

def merge_samples(output_file: Path):
    # Get all of the epoch samples
    all_samples = glob("./hdf5_file_*.hdf5")
    print(f"Got {len(all_samples)} samples")

    # Each sample will be an hdf5 file where the keys correspond to epochs. 
    # We will create a new hdf5 file, iterate over the samples, and store them
    # in the new hdf5 file, based on the epoch key.

    print(f"Saving to {output_file}")
    output_df = pd.HDFStore(output_file, mode="w")
    epoch_num = 0

    for sample in tqdm(all_samples):
        sample_df = pd.HDFStore(sample, mode="r")
        for key in sample_df.keys():
            output_df.put(f'epoch_{epoch_num}', sample_df.get(key))
            epoch_num += 1
        sample_df.close()

    output_df.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    output_file = Path(args.output_name)

    # Ensure it does not exist
    if output_file.exists():
        raise ValueError(f"Output file {output_file} already exists")

    # Make all directories for it
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Combine Epoch Samples
    merge_samples(output_file)

if __name__ == "__main__":
    main()