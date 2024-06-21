import argparse
import matchms
from glob import glob
import os
import pyteomics.mgf as py_mgf

from matchms.importing import load_from_mgf

def main():
    parser = argparse.ArgumentParser(description='Get positive spectra from .mgf file')
    parser.add_argument('--input_dir', type=str, help='input .mgf file')
    parser.add_argument('--output_dir', type=str, help='output .mgf file')
    args = parser.parse_args()
    
    if args.output_dir is None:
        if args.input_dir.endswith('/'): 
            args.input_dir = args.input_dir.rstrip('/')
        args.output_dir = args.input_dir + "_positive"
    print(f"Output directory: {args.output_dir}")    
    
    for path in glob(args.input_dir + "/*mgf"):
        print("Processing file:", path)
        spectra = list(load_from_mgf(path))
        print(f"Got {len(spectra)} spectra from {path}")
        # Assign polarities
        for spectrum in spectra:
            # if '+' in str(spectrum.get("charge")) and '-' in str(spectrum.get("charge")):
            #     raise ValueError("Spectrum has both positive and negative charges")
            # elif '+' in str(spectrum.get("charge")):
            #     spectrum.set("ionmode", "positive")
            # # If spectrum charge < 0, set ion mode to negative
            # elif '-' in str(spectrum.get("charge")):
            #     spectrum.set("ionmode", "negative")
            # else:
            #     print("Unable to determine ion mode for spectrum:", spectrum.get("title"), spectrum.get("charge"))

            if spectrum.get('charge') is None:
                print("Unable to determine ion mode for spectrum:", spectrum.get("title"), spectrum.get("charge"))
            elif int(spectrum.get('charge')) > 0:
                spectrum.set("ionmode", "positive")
            # If spectrum charge < 0, set ion mode to negative
            elif int(spectrum.get('charge')) < 0:
                spectrum.set("ionmode", "negative")
            else:
                print("Unable to determine ion mode for spectrum:", spectrum.get("title"), spectrum.get("charge"))            
            
        # Filter spectra by polarity
        spectra_positive = [spectrum for spectrum in spectra if spectrum.get("ionmode") == "positive"]
        if len(spectra_positive) ==0:
            print("No positive spectra found for file:", path)
            print([s.get("ionmode") for s in spectra])
            continue
        # Save spectra
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        print("Saving file:", args.output_dir + "/" + path.split("/")[-1])
        # matchms.exporting.save_as_mgf(spectra_positive, args.output_dir + "/" + path.split("/")[-1])
        
        # For some reason matchms.exporting.save_as_mgf will only output one spectra
        # This is a workaround for that code, but the above line should be used once working
        output_path = args.output_dir + "/" + path.split("/")[-1]
        def output_gen():
            for spectrum in spectra_positive:
                spectrum_dict = {"m/z array": spectrum.peaks.mz,
                                    "intensity array": spectrum.peaks.intensities,
                                    "params": spectrum.metadata}
                if 'fingerprint' in spectrum_dict["params"]:
                    del spectrum_dict["params"]["fingerprint"]
                yield spectrum_dict
        py_mgf.write(output_gen(), output_path, file_mode="w")

    
if __name__ == "__main__":
    main()