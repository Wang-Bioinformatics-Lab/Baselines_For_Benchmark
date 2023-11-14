from spec2vec import Spec2Vec
import argparse
from runner import spectrum_document_generator_parallel
import gensim
from matchms import calculate_scores
import numpy as np
import os

num_cpus = 8

def calculate_similarity(spectra_document_path, model_path, num_cpus=num_cpus):
    # load documents  
    documents = sorted(os.listdir(spectra_document_path))
    spectrum_documents=[np.load(os.path.join(spectra_document_path, file),allow_pickle=True).item()  for file in documents]
    spectrum_documents_names = [file.split('.')[0] for file in documents]
    
    # Load model    
    model = gensim.models.Word2Vec.load(model_path)
    
    # Define similarity_function
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5,
                        allowed_missing_percentage=100)                     # We'll allow 100% missing here because that's necessairily part of our results

    # Calculate scores on all combinations of reference spectrums and queries
    scores = calculate_scores(spectrum_documents, spectrum_documents, spec2vec, is_symmetric=True)
    
    # Return highest scores
    np.save('./spec2vec_predictions.npy', scores.to_array())
    np.save('./spec2vec_predictions_names.npy', spectrum_documents_names)

def main():
    parser = argparse.ArgumentParser(description='Calculate similarities with a spec2vec model.')
    parser.add_argument('--spectra', type=str, required=False)
    parser.add_argument('--metadata', type=str, required=False)
    parser.add_argument('--spectral_documents', type=str, required=False)
    parser.add_argument('--model_save_name', type=str, default="word2vec.model")
    parser.add_argument('--num_cpus', type=int, required=False)
    args = parser.parse_args()
    
    if args.num_cpus:
        global num_cpus
        num_cpus = args.num_cpus
    
    if args.spectral_documents:
        if args.spectra and args.metadata:
            print("Spectra and metadata provided in addition to documetns. Using existing documents.",flush=True)
        spectra_document_path = args.spectral_documents    
    elif args.spectra and args.metadata:
        spectrum_document_generator_parallel(args.spectra, args.metadata, output_path="./spectrum_documents_similarity_calculations/", num_cpus=num_cpus)
        spectra_document_path = "./spectrum_documents_similarity_calculations/"
    else:
        print("No spectra, metadata, or documents provided. Exiting.",flush=True)
        exit()
    
    calculate_similarity(spectra_document_path, args.model_save_name, num_cpus=num_cpus)

if __name__ == '__main__':
    main()