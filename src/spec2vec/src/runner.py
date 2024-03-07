import os
import gensim
import pandas as pd
import numpy as np
import vaex
from tqdm import tqdm
from joblib import Parallel, delayed
import matchms
import matchms.filtering as filter
from pyteomics.mgf import IndexedMGF
from spec2vec import SpectrumDocument, Spec2Vec, calc_vector
from spec2vec.model_building import train_new_word2vec_model
import argparse

num_cpus = 1

def spectrum_document_generator_parquet(spectra_path, metadata_path, indices, spectrum_ids,output_path="./spectrum_documents"):
    df = vaex.open(spectra_path)
    df = df[df.spectrum_id.isin(spectrum_ids)]
    
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df.spectrum_id.isin(spectrum_ids)]
    
    for idx, id in enumerate(spectrum_ids):
        spectrum_df = df[df.spectrum_id == id]
        metadata = metadata_df[metadata_df.spectrum_id == id]
        prec_mz = spectrum_df.prec_mz.values[0][0].as_py()
        
        # Precleaining
        s = matchms.Spectrum(mz=spectrum_df.mz.values.to_numpy(), 
                             intensities=spectrum_df.i.values.to_numpy(), 
                             metadata={"id": id,
                                       "precursor_mz":prec_mz,
                                       "adduct":metadata.Adduct.item(),
                                       "collision_energy":metadata.collision_energy.item(),
                                       "charge":metadata.Charge.item(),
                                       "compound_name":None})
        s = filter.default_filters(s)
        s = filter.add_parent_mass(s)
        s = filter.normalize_intensities(s)
        s = filter.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
        s = filter.select_by_mz(s, mz_from=0, mz_to=10000)
        s = filter.add_losses(s, loss_mz_from=5.0, loss_mz_to=200.0)
        s = filter.require_minimum_number_of_peaks(s, n_required=10)
        if s is None: continue # Sometimes things don't get passed cleaning
        
        spectrum_document = SpectrumDocument(s, n_decimals=2)
        
        np.save(output_path + str(id) + '.npy', spectrum_document)

def spectrum_document_generator_mgf(spectra_path, metadata_path, spectrum_ids,output_path="./spectrum_documents"):
    spectra = IndexedMGF(spectra_path)
    spectra = spectra.get_by_ids(spectrum_ids)
    
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df.spectrum_id.isin(spectrum_ids)]
    
    for idx, spectrum in enumerate(spectra):
        id = spectrum['params']['title']
        metadata = metadata_df[metadata_df.spectrum_id == id]
        prec_mz = spectrum['params']['precursor_mz'][0]
        
        # Precleaining
        ordered_indices = np.argsort(spectrum['m/z array'])
        s = matchms.Spectrum(mz=spectrum['m/z array'][ordered_indices], 
                             intensities=spectrum['intensity array'][ordered_indices], 
                             metadata={"id": id,
                                       "precursor_mz":prec_mz,
                                       "adduct":metadata.Adduct.item(),
                                       "collision_energy":metadata.collision_energy.item(),
                                       "charge":metadata.Charge.item(),
                                       "compound_name":None})
        s = filter.default_filters(s)
        s = filter.add_parent_mass(s)
        s = filter.normalize_intensities(s)
        s = filter.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
        s = filter.select_by_mz(s, mz_from=0, mz_to=10000)
        s = filter.add_losses(s, loss_mz_from=5.0, loss_mz_to=200.0)
        s = filter.require_minimum_number_of_peaks(s, n_required=10)
        if s is None: continue
        
        spectrum_document = SpectrumDocument(s, n_decimals=2)
        
        np.save(output_path + str(id) + '.npy', spectrum_document)

def spectrum_document_generator_parallel_parquet(spectra_path, metadata_path, output_path="./spectrum_documents/", num_cpus=1):
    print("Generating spectrum documents with {} cpus".format(num_cpus),flush=True)
    df = vaex.open(spectra_path)
    spectrum_ids = df.spectrum_id.unique()

    indices = np.arange(len(spectrum_ids))
    
    num_blocks = 1000
    
    indices = [list(x) for x in np.array_split(indices, num_blocks)]
    splits  = [list(x) for x in np.array_split(spectrum_ids, num_blocks)] # vaex doesn't like comparing to numpy arrays for some reason

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    # with parallel_backend('multiprocessing'):
    Parallel(n_jobs=num_cpus)(delayed(spectrum_document_generator_parquet)(spectra_path, metadata_path, indices[i], split_ids, output_path=output_path) for i, split_ids in enumerate(tqdm(splits)))
    print("Finished Dataset Preprocessing")
    
def spectrum_document_generator_parallel_mgf(spectra_path, metadata_path, output_path="./spectrum_documents/", num_cpus=1):
    print("Generating spectrum documents with {} cpus".format(num_cpus),flush=True)
    spectra = IndexedMGF(spectra_path)
    spectrum_ids = [spectrum['params']['title'] for spectrum in spectra]
    
    indices = np.arange(len(spectrum_ids))
    
    indices = [list(x) for x in np.array_split(indices, num_cpus)]
    splits  = [list(x) for x in np.array_split(spectrum_ids, num_cpus)] 
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    Parallel(n_jobs=num_cpus)(delayed(spectrum_document_generator_mgf)(spectra_path, metadata_path, splits[i], output_path=output_path) for i, split_ids in enumerate(tqdm(splits)))
    print("Finished Dataset Preprocessing")
        
def process(train_spectra_path,  
            train_metadata_path, 
            test_spectra_path,
            test_metadata_path,
            language_embedding_model, 
            test, 
            train_spectrum_doc_path= "./spectrum_documents/", 
            test_spectrum_doc_path="./spectrum_documents_test/" ,
            train_embedded_path="./embedded_spectra/", 
            test_embedded_path="./embedded_spectra_test/", 
            withold_test=True):
    
    train_spectrum_doc_path = train_spectrum_doc_path + train_spectra_path.split('/')[-1].rsplit('.',1)[0] + '/'
    test_spectrum_doc_path = test_spectrum_doc_path + test_spectra_path.split('/')[-1].rsplit('.',1)[0] + '/'
    train_embedded_path = train_embedded_path + train_spectra_path.split('/')[-1].rsplit('.',1)[0] + '/'
    test_embedded_path = test_embedded_path + test_spectra_path.split('/')[-1].rsplit('.',1)[0] + '/'

    
    if not os.path.isdir(train_spectrum_doc_path):
        os.makedirs(train_spectrum_doc_path, exist_ok=True)
    if not os.path.isdir(test_spectrum_doc_path):
        os.makedirs(test_spectrum_doc_path, exist_ok=True)
    
    if not os.path.isdir(train_embedded_path):
        os.makedirs(train_embedded_path, exist_ok=True)
    if not os.path.isdir(test_embedded_path):
        os.makedirs(test_embedded_path, exist_ok=True)
            
    
    for spectra_path, metadata_path, doc_path in zip([train_spectra_path, test_spectra_path], [train_metadata_path, test_metadata_path], [train_spectrum_doc_path, test_spectrum_doc_path]):
        if spectra_path == None:
            continue
        if spectra_path.endswith('parquet'):
            print("Performing preprocessing on a parquet files.")
            spectrum_document_generator_parallel_parquet(spectra_path, metadata_path, output_path=train_spectrum_doc_path, num_cpus=num_cpus)
        elif spectra_path.endswith('mgf'):
            print("Performing preprocessing on a mgf file.")
            print("Input Spectrum Path", spectra_path)
            print("Processed document Path", doc_path)
            spectrum_document_generator_parallel_mgf(spectra_path, metadata_path, output_path=doc_path, num_cpus=num_cpus)
        else:
            print("Unknown file type. Exiting.",flush=True)
            return

    if not os.path.isfile(language_embedding_model):
        if test:
            print("Language Model Not Found. Cannot Train on Test Set. Exiting.",flush=True)
            return
        else:
            print("Language Model Not Found. Training Embedding Model.",flush=True)
            # sentences=[list(np.load('./data/intermediate/' + file,allow_pickle=True)['words']) for file in os.listdir('./data/intermediate')]
            # print([train_spectrum_doc_path + file for file in os.listdir(train_spectrum_doc_path)])
            docs=[np.load(train_spectrum_doc_path + file,allow_pickle=True).item() for file in os.listdir(train_spectrum_doc_path)]
            # All docs will include test so that we can perform inference on unseen data. Note that these will be exceluded from the training data.
            all_docs = docs + [np.load(test_spectrum_doc_path + file,allow_pickle=True).item() for file in os.listdir(test_spectrum_doc_path)]
            if not withold_test:
                # If this is going to be used in downstream applications, we may want to train over all of the data
                docs = all_docs
            
            model = train_new_word2vec_model(docs, filename=language_embedding_model, iterations=50, vector_size=300, window=500, workers=8, learning_rate_initial=0.025, learning_rate_decay=0.00025, negative=5, sg=0, progress_logger=True)
            
            # model = gensim.models.Word2Vec(min_count=1, vector_size=512, window=500, workers=int(num_cpus), epochs=50)
            # model.build_vocab(all_docs)    # Includes test
            # model.train(tqdm(docs), total_examples=model.corpus_count, epochs=model.epochs)   # Does not include test
    
            print("Finished Language Model Training",flush=True)
    
    else:
        raise NotImplementedError("Loading of existing models is not yet implemented.")
        print("Language Model Found.",flush=True)
        return
    
    print("Embedding Spectra")
    for to_embed, embedded_path in zip([train_spectrum_doc_path, test_spectrum_doc_path],[train_embedded_path, test_embedded_path]):
        if to_embed == None:
            continue
        if not os.path.isdir(embedded_path):
            os.makedirs(embedded_path, exist_ok=True)
        for file in os.listdir(to_embed):
            spectrum_document=np.load(to_embed + file,allow_pickle=True).item()

            # precursor_mz = float(spectrum_document.metadata.get("precursor_mz"))     
            # ht = np.array(spectrum_document.peaks.intensities)
            # mz = spectrum_document.peaks.mz
            
            # Find closest peak to precursor mass and set intensity to 2
            # ht[np.argmin(np.abs(mz - precursor_mz))] = 2
            
            # Write molecule attributes as tensor
            # truncated_doc = np.array(list(spectrum_document))[:len(ht)]
            # max_ind = min(512, len(ht))
            # highest_intensity = np.argsort(ht)[:max_ind]
            # x = [np.append(model.wv.get_vector(word), [ht]) for (word, ht) in zip(np.take(list(spectrum_document),highest_intensity),np.take(ht,highest_intensity))]
            x = calc_vector(model, spectrum_document, allowed_missing_percentage=100)  # We allow full missing because it's either in the training set, or it's not.
            np.save(embedded_path + file, x)

def main():
    parser = argparse.ArgumentParser(description='Train a spec2vec model.')
    parser.add_argument('--train_spectra', type=str, default=None)
    parser.add_argument('--train_metadata', type=str, default=None)
    parser.add_argument('--test_spectra', type=str, default=None)
    parser.add_argument('--test_metadata', type=str, default=None)
    
    parser.add_argument('--model_save_name', type=str, default="./word2vec.model")
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--num_cpus', type=int, default=1)
    # parser.add_argument('--test', action='store_true')
    parser.add_argument('--withold_test', type=bool, default=False)
    args = parser.parse_args()
    
    train_spectra            = args.train_spectra 
    train_metadata           = args.train_metadata 
    test_spectra             = args.test_spectra
    test_metadata            = args.test_metadata
    language_embedding_model = args.model_save_name
    
    language_embedding_model_dir = os.path.dirname(language_embedding_model)
    if not os.path.isdir(language_embedding_model_dir):
        os.makedirs(language_embedding_model_dir, exist_ok=True)
    
    global num_cpus
    num_cpus = args.num_cpus
    
    train_spectrum_doc_path= "./spectrum_documents/"
    test_spectrum_doc_path="./spectrum_documents_test/" 
    train_embedded_path="./embedded_spectra/"
    test_embedded_path="./embedded_spectra_test/"
    
    if args.save_path:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
        train_spectrum_doc_path= os.path.join(args.save_path, train_spectrum_doc_path)
        test_spectrum_doc_path= os.path.join(args.save_path, test_spectrum_doc_path)
        train_embedded_path= os.path.join(args.save_path, train_embedded_path)
        test_embedded_path= os.path.join(args.save_path, test_embedded_path)

    process(train_spectra, 
            train_metadata, 
            test_spectra,
            test_metadata,
            language_embedding_model,
            False,
            withold_test=args.withold_test,
            train_spectrum_doc_path=train_spectrum_doc_path, 
            test_spectrum_doc_path=test_spectrum_doc_path,
            train_embedded_path=train_embedded_path, 
            test_embedded_path=test_embedded_path)
    
    # This is currently deprecated, everything can now be specified by the presence (or lack thereof) of training/test paths.
    # If it's not present, it gets skipped.
    # if args.test:
    print("Running in Test Mode. Will not train embedding model.",flush=True)

    # else:
        
    #     print("Running in Train Mode. Will train embedding model.",flush=True)
    #     process(train_spectra, 
    #             train_metadata, 
    #             language_embedding_model,
    #             test=args.test)

    
if __name__ == '__main__':
    main()