import argparse
from glob import glob
import numpy as np
import os
import torch

def process_to_torch(data_glob, output_path):
    files = glob(data_glob)
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        data = np.load(file)
        torch.save(os.path.join(output_path, file.split('/')[-1]), data)

def main():
    parser = argparse.ArgumentParser(description='Convert data points to pytorch.')
    parser.add_argument("--data_glob", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    process_to_torch(args.data_glob, args.output_path)

if __name__ == "__main__":
    main()