import argparse

def subsample_test_ids(test_spectra, output_id_path, spectra_per_structure):
    raise NotImplementedError("This function is not implemented yet")

def main():
    """The purpose of this script to subsample the spectra for the test dataset.
    The reason we do this is that by the time we have sufficent structure sampling, we have so many 
    spectral pairs that it's not feasible to fit them into memory.
    """

    parser = argparse.ArgumentParser(description='Generate all pairs of test ids')
    parser.add_argument('--test_spectra', type=str, help='Path to the test ids file')
    parser.add_argument('--output_id_path', type=str, help='Path to the output file')
    parser.add_argument('--spectra_per_structure', type=int, help='Number of spectra per structure', default=5)
    args = parser.parse_args()

    subsample_test_ids(args.test_spectra, args.output_id_path, args.spectra_per_structure)

if __name__ == "__main__":
    main()