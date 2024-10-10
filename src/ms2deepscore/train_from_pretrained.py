import argparse
import os
import gc
import logging
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tempfile
import shutil

from custom_spectrum_binner import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys#, DataGeneratorCherrypicked
# from ms2deepscore.train_new_model.spectrum_pair_selection import SelectedCompoundPairs
from ms2deepscore.models import SiameseModel
from custom_cnn_encoded_siamese_model import SiameseModel as CNNSiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from train_generator_from_pairs import SelectedCompoundPairs, DataGeneratorCherrypickedInChi

import pickle

# DEBUG
from line_profiler import profile

class StrWrapper():
    """ A small wrapper to deal with .to_json() calls made on string
    metadata.
    """
    def __init__(self, s):
        self.s = s
    
    def to_json(self):
            return self.s
@profile
def main():
    parser = argparse.ArgumentParser(description='Train MS2DeepScore on the original data')
    parser.add_argument('--train_path', type=str, help='Path to the training data')
    parser.add_argument('--val_path', type=str, help='Path to the validation data')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to the pretrained model')
    parser.add_argument('--train_pairs_path', type=str, help='Path to the pairs', default=None)
    parser.add_argument('--val_pairs_path', type=str, help='Path to the pairs', default=None)
    parser.add_argument('--tanimoto_path', type=str, help='Path to the tanimoto scores')
    parser.add_argument('--save_dir', type=str, help='Path to the save directory')
    parser.add_argument('--hidden_dims', type=int, nargs='+', help='Hidden dimensions', default=[500, 500])
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension', default=200)
    parser.add_argument('--use_cnn_model', action='store_true', help='Use the CNN model instead of the dense model')
    parser.add_argument('--gpu', type=int, help='Which GPU to use', default=0)
    parser.add_argument('--test_sim_matrix', type=str, help="Test set similarity matrix. Used to remove common inchikeys")
    args = parser.parse_args()
        
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.isdir(os.path.join(args.save_dir, 'logs')):
        os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
        
    try:
        
        # Get current time in dd_mm_yyyy_hh_mm_ss format
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            
        logging.basicConfig(format='[%(levelname)s]: %(message)s',
                            level=logging.DEBUG,
                            handlers=[logging.FileHandler(os.path.join(args.save_dir, 'logs', f"training_{current_time}.log"),
                                                        mode='w'),
                                    logging.StreamHandler()]
                            )
        
        try:
            # TODO add scratch path as an argument
            if args.train_pairs_path is not None:
                if args.train_pairs_path.endswith('hdf5'):
                
                    # Check the file size
                    file_size = os.path.getsize(args.train_pairs_path)
                    
                    # Check available space in the destination directory
                    statvfs = os.statvfs('/')
                    available_space = statvfs.f_frsize * statvfs.f_bavail
                    
                    if file_size > available_space:
                        raise RuntimeError("Not enough space on the filesystem to copy the file.")
                    
                    # Create a temporary file in the root directory
                    _, file_extension = os.path.splitext(args.train_pairs_path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_pair_path = temp_file.name
                        
                    shutil.copyfile(args.train_pairs_path, temp_pair_path)
                    
                    train_pair_path = temp_pair_path
                    logging.info("Successfully copied train pairs to scratch")
                else:
                    train_pair_path = args.train_pairs_path
        except Exception as e:
            logging.info("Failed to copy pairs to scratch")
            print(e, flush=True)
            train_pair_path = args.train_pairs_path   # don't change the pair

            

        
        # Log all args
        logging.info('All arguments:')
        for arg in vars(args):
            logging.info('%s: %s', arg, getattr(args, arg))
        logging.info('')
        
        # Check if GPU is available
        logging.info("GPU is %s", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
        if tf.config.list_physical_devices('GPU'):
            # Log which gpu is being used
            logging.info("GPU in use: %s", tf.test.gpu_device_name())
        
        
        logging.info("Loading data...")
        spectrums_training = pickle.load(open(args.train_path, "rb"))
        spectrums_val = pickle.load(open(args.val_path, "rb"))
        tanimoto_df = pd.read_csv(args.tanimoto_path, index_col=0)
        logging.info("\tDone.")
        
        
        logging.info("Loading model...")
        model = load_model(args.pretrained_model_path)
        spectrum_binner = model.spectrum_binner
        logging.info("\tDone.")
        
        logging.info("Binning Spectra...")
        binned_spectrums_training = spectrum_binner.transform(spectrums_training)
        binned_spectrums_val = spectrum_binner.transform(spectrums_val)
        logging.info("\tDone.")

        training_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_training])
        logging.info(len(training_inchikeys))

        # tanimoto_scores_df = pickle.load(open("./data/similarities_train_pos.pkl", "rb"))
        # logging.info("Total number of inchikey14s:", len(tanimoto_scores_df.columns))

        same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
        dimension = len(spectrum_binner.known_bins)
        validation_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_val])
        
        if args.train_pairs_path is None:
            training_generator = DataGeneratorAllInchikeys(
                binned_spectrums_training, tanimoto_df, spectrum_binner=spectrum_binner, selected_inchikeys=training_inchikeys,
                same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)
            del binned_spectrums_training
            gc.collect()
        else:
            # Using the given pairs, we can create a generator that only uses these pairs
            if train_pair_path.endswith('feather'):
                test_sim_matrix = pd.read_csv(args.test_sim_matrix, index_col=0)
                assert set(test_sim_matrix.index) == set(test_sim_matrix.columns)
                test_inchis = set(test_sim_matrix.columns)
                del test_sim_matrix
                
                train_pairs = pd.read_feather(train_pair_path)
                start_len = len(train_pairs)
                train_pairs = train_pairs.loc[~((train_pairs['inchikey_1'].isin(test_inchis)) | (train_pairs['inchikey_2'].isin(test_inchis)))]
                logging.info("Removed %i pairs that had a structure in the test set.", (start_len - len(train_pairs)))
                
                # Print all cols and dtypes
                for col in train_pairs.columns:
                    logging.info('%s: %s', col, train_pairs[col].dtype)
            elif train_pair_path.endswith('hdf5'):
                raise NotImplementedError()
                train_pairs = pd.HDFStore(train_pair_path, 'r')
            
            selected_compound_pairs_train = SelectedCompoundPairs(train_pairs, shuffle=True, same_prob_bins=same_prob_bins)
            del train_pairs
            gc.collect()
            training_generator = DataGeneratorCherrypickedInChi(
                binned_spectrums_training, selected_compound_pairs_train, spectrum_binner=spectrum_binner, selected_inchikeys=training_inchikeys,
                same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01
            )
            del selected_compound_pairs_train, binned_spectrums_training
            gc.collect()

        if args.val_pairs_path is None:
            validation_generator = DataGeneratorAllInchikeys(
                binned_spectrums_val, tanimoto_df, spectrum_binner=spectrum_binner, selected_inchikeys=validation_inchikeys, same_prob_bins=same_prob_bins,
                num_turns=10, augment_removal_max=0, augment_removal_intensity=0,
                augment_intensity=0, augment_noise_max=0, use_fixed_set=True)
            del binned_spectrums_val
            gc.collect()
        else:
            if args.val_pairs_path.endswith('feather'):               
                val_pairs = pd.read_feather(args.val_pairs_path)
                
            elif args.train_pairs_path.endswith('hdf5'):
                raise NotImplementedError()
                val_pairs = pd.HDFStore(args.val_pairs_path, 'r')
            
            selected_compound_pairs_val  = SelectedCompoundPairs(val_pairs, shuffle=False, same_prob_bins=same_prob_bins)
            del val_pairs
            gc.collect()
            validation_generator = DataGeneratorCherrypickedInChi(
                binned_spectrums_val, selected_compound_pairs_val, spectrum_binner=spectrum_binner, selected_inchikeys=validation_inchikeys, same_prob_bins=same_prob_bins,
                num_turns=10, augment_removal_max=0, augment_removal_intensity=0,
                augment_intensity=0, augment_noise_max=0, use_fixed_set=True
            )
            del selected_compound_pairs_val#, binned_spectrums_val
            gc.collect()
            
        # DEBUG
        logging.info("Training generator:")
        logging.info(training_generator)
        logging.info(training_generator[0])
        
        logging.info("Validation generator:")
        logging.info(validation_generator)
        logging.info(validation_generator[0])
        

        if not args.use_cnn_model:
            model = SiameseModel(spectrum_binner,
                                base_dims=tuple(args.hidden_dims),
                                embedding_dim=int(args.embedding_dim),
                                dropout_rate=0.2)
        elif args.use_cnn_model:
            model = CNNSiameseModel(spectrum_binner, embedding_dim=args.embedding_dim)
        else:
            raise ValueError("Model type not recognized")
        
        model.compile(loss='mse',
                    optimizer=Adam(learning_rate=0.001),
                    metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

        # Save best model and include earlystopping
        earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1, start_from_epoch=20, restore_best_weights=True)

        history = model.model.fit(training_generator, validation_data=validation_generator,
                                epochs=150, verbose=1, callbacks=[earlystopper_scoring_net])

        # history = model.model.fit(training_generator, validation_data=validation_generator,
        #                         epochs=50, verbose=1) 

        model_file_name = os.path.join(args.save_dir, f"model_{current_time}.hdf5")
        model.save(model_file_name)

        # Save history to pickle
        pickle.dump(history.history, open(os.path.join(args.save_dir, f"model_{current_time}_history.pkl"), "wb"))

        # Get current log level
        log_level = logging.getLogger().getEffectiveLevel()
        # Set log level to INFO to plot the loss
        logging.getLogger().setLevel(logging.INFO)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(args.save_dir, "loss.png"))
        
        # Reset log level
        logging.getLogger().setLevel(log_level)
    finally:
        os.remove(temp_pair_path)

if __name__ == "__main__":
    main()