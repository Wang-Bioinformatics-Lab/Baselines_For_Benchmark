import argparse
import os
import pandas as pd
from pathlib import Path
from glob import glob
import pyteomics.mgf as py_mgf

import matchms

def filter_data(input_dir, output_dir, NIST_Only=False):
    
    input_csv = glob(os.path.join(input_dir, "*.csv"))
    input_mgf = glob(os.path.join(input_dir, "*.mgf"))
    
    print("Input CSV:", input_csv)
    print("Input MGF:", input_mgf)
    
    assert len(input_csv) == 1
    assert len(input_mgf) == 1
    input_csv = Path(list(input_csv)[0])
    input_mgf = Path(list(input_mgf)[0])

    
    # New libraries could also inlcude: "MSMS-Pos-bmdms-np_20200811", but we already have this library in the current data
    # Strictly speaking, none of the negative library spectra will have been included in our data, but this is all of the new libraries
    new_libs = {"MSMS-Neg-RikenOxPLs", "MSMS-Pos-PlaSMA", "MSMS-Neg-PlaSMA", "BioMSMS-Pos-PlaSMA", "BioMSMS-Neg-PlaSMA", "MSMS-Neg-PFAS_20200806",
                "CMMC-REFRAME-NEGATIVE-LIBRARY", "CMMC-REFRAME-POSITIVE-LIBRARY", } # These ones are not from Riken, but are new
    
    if NIST_Only:
        new_libs = {"NIST"}
    
    # Read the inputs csv, get all rows who are a member of these libraries
    summary = pd.read_csv(input_csv)
    new_subset = summary.loc[summary.GNPS_library_membership.isin(new_libs)]
    del summary
    
    # Export the new subset
    output_path = os.path.join(output_dir, input_csv.stem + "_new.csv")
    new_subset.to_csv(output_path, index=False)
    
    # Get the spectra for these rows
    new_ids = set(new_subset.spectrum_id)
    del new_subset
    
    # Read the mgf file
    spectra = list(matchms.importing.load_from_mgf(str(input_mgf)))
    
    # Filter the spectra
    print("Original number of spectra:", len(spectra))
    spectra = [spectrum for spectrum in spectra if spectrum.get("spectrum_id") in new_ids]
    print("Number of new spectra:", len(spectra))
    
    assert len(spectra) == len(new_ids)
    
    # Export MGF
    output_path = os.path.join(output_dir, input_mgf.stem + "_new.mgf")
    # For some reason matchms.exporting.save_as_mgf will only output one spectra
    # This is a workaround for that code, but the above line should be used once working
    def output_gen():
        for spectrum in spectra:
            spectrum_dict = {"m/z array": spectrum.peaks.mz,
                                "intensity array": spectrum.peaks.intensities,
                                "params": spectrum.metadata}
            if 'fingerprint' in spectrum_dict["params"]:
                del spectrum_dict["params"]["fingerprint"]
            yield spectrum_dict
    py_mgf.write(output_gen(), output_path, file_mode="w")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--NIST_Only", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    assert input_dir.exists()
    
    filter_data(input_dir, output_dir, args.NIST_Only)
    
if __name__ == "__main__":
    main()