import argparse
import gc
import os
import logging
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from typing import Iterator, List, NamedTuple, Optional
import warnings

# Low memory generator 
from generator_filtered_pairs import FilteredPairsGenerator

def main():
    parser = argparse.ArgumentParser(description='Train MS2DeepScore on the original data')
    parser.add_argument('--train_path', type=str, help='Path to the training data')
    parser.add_argument('--train_pairs_path', type=str, help='Path to the pairs', default=None)
    parser.add_argument('--tanimoto_scores_path', type=str, help='Path to the tanimoto scores', default=None)
    parser.add_argument('--save_dir', type=str, help='Path to the save directory')
    parser.add_argument('--save_format', type=str, choices=['pickle', 'hdf5'], default='hdf5')
    parser.add_argument('--num_epochs', type=int, default=1800)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_turns', type=int, default=2) # Default for MS2DeepScore Training is 2, Validation is 10
    parser.add_argument('--low_io', action='store_true', help='Use low io mode')
    parser.add_argument('--memory_efficent', action='store_true', help='Use memory efficient mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Parameters for filtering, only applys in memory efficent mode
    parser.add_argument('--merge_on_lst', type=str, help='A semicolon delimited list of criteria to merge on. \
                                                            Options are: ["ms_mass_analyzer", "ms_ionisation", "adduct", "library"]. Only applys in memory efficent mode.',
                                                    default=None)
    parser.add_argument('--mass_analyzer_lst', type=str, default=None,
                                                help='A semicolon delimited list of allowed mass analyzers. All mass analyzers are \
                                                      allowed when not specified. Only applys in memory efficent mode.')
    parser.add_argument('--collision_energy_thresh', type=float, default=5.0,
                                                help='The maximum difference between collision energies of two spectra to be considered\
                                                    a pair. Default is <= 5. "-1.0" means collision energies are not filtered. \
                                                    If no collision enegy is available for either spectra, both will be included. Only applys in memory efficent mode.')
    parser.add_argument('--no_pm_requirement', action='store_true', help='Do not require precursor mass difference to be less than 200. Only applys in memory efficent mode.', default=False)

    args = parser.parse_args()
    
    if args.train_pairs_path is None and args.tanimoto_scores_path is None:
        raise ValueError("Either train_pairs_path or tanimoto_scores_path must be provided.")
    elif args.train_pairs_path is not None and args.tanimoto_scores_path is not None:
        print("Both train_pairs_path and tanimoto_scores_path are provided. Using train_pairs_path.")

    if args.memory_efficent:
        if args.low_io:
            raise ValueError("Memory efficient mode is not compatible with low io mode.")

        if not args.train_pairs_path is None:
            raise ValueError("Memory efficient mode is not compatible with train_pairs_path.")
    else:
        # Ensure None of the filtering flags were set
        if not args.merge_on_lst is None:
            raise ValueError("merge_on_lst is only compatible with memory efficient mode.")
        if not args.mass_analyzer_lst is None:
            raise ValueError("mass_analyzer_lst is only compatible with memory efficient mode.")
        if args.collision_energy_thresh != 5.0:
            raise ValueError("collision_energy_thresh is only compatible with memory efficient mode.")
        if args.no_pm_requirement:
            raise ValueError("no_pm_requirement is only compatible with memory efficient mode.")

        

    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    
    logging.basicConfig(format='[%(levelname)s]: %(message)s',
                    level=logging.DEBUG,
                    handlers=[logging.FileHandler(os.path.join(args.save_dir, 'logs', f"presampling_{current_time}.log"),
                                                mode='w'),
                            logging.StreamHandler()]
                    )
    
    logging.info('All arguments:')
    logging.info('Note: merge_on_lst, mass_analyzer_lst, collision_energy_thresh, no_pm_requirement are only used in memory efficent mode.')
    for arg in vars(args):
        logging.info('%s: %s', arg, getattr(args, arg))
    logging.info('')
    
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    num_turns  = int(args.num_turns)

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
    
    if args.train_pairs_path is None:
        train_pairs = None
    elif args.train_pairs_path.endswith('feather'):
                train_pairs = pd.read_feather(args.train_pairs_path, use_threads=False) # Sacrifice IO speed to reduce initial memory hit
                # Print all cols and dtypes
                for col in train_pairs.columns:
                    logging.info('%s: %s', col, train_pairs[col].dtype)
    elif args.train_pairs_path.endswith('hdf5'):
        train_pairs = pd.HDFStore(args.train_pairs_path, 'r')
    else:
        raise ValueError("Unknown file format for train_pairs_path")
    
    logging.info("Loading data...")
    spectrums_training = pickle.load(open(args.train_path, "rb"))
    all_training_inchikeys = [s.get("inchikey")[:14] for s in spectrums_training]
    training_inchikeys = np.unique(all_training_inchikeys)
    training_ids = [s.get("spectrum_id") for s in spectrums_training]

    if train_pairs is not None:
        if not args.low_io:
            selected_compound_pairs_train = SelectedCompoundPairs(train_pairs, inchikeys=training_inchikeys, shuffle=True, same_prob_bins=same_prob_bins)
        else:
            selected_compound_pairs_train = SelectedCompoundPairsLowIO(train_pairs, inchikeys=training_inchikeys, shuffle=True, same_prob_bins=same_prob_bins)
    gc.collect()

    if train_pairs is not None:
        logging.info("Creating DataGeneratorCherrypickedInChi (Using Pre-generated Filtered Pairs)...")
        training_generator = DataGeneratorCherrypickedInChi(
                                                        training_inchikeys,
                                                        training_ids,
                                                        selected_compound_pairs_train,
                                                        same_prob_bins=same_prob_bins,
                                                        shuffle=True,
                                                        random_seed=args.seed,
                                                        num_turns=num_turns,
                                                        batch_size=batch_size)
        del selected_compound_pairs_train, training_inchikeys, training_ids
        gc.collect()
    else:
        reference_scores_df = pd.read_csv(args.tanimoto_scores_path, index_col=0)
        reference_scores_df = reference_scores_df.loc[training_inchikeys, training_inchikeys]

        if args.memory_efficent:
            metadata = pd.DataFrame([{'spectrum_id': s.get('spectrum_id'), 
                            'inchikey': s.get('inchikey'), 
                            'collisionEnergy': s.get('collision_energy'),
                            'Adduct': s.get('adduct'),
                            'msMassAnalyzer':s.get('ms_mass_analyzer'),
                            'msIonisation': s.get('ms_ionisation'),
                            'ExactMass': s.get('parent_mass')} for s in tqdm(spectrums_training)])
            metadata['inchikey_14'] = metadata['inchikey'].str[:14]

            logging.info("Creating FilteredPairsGenerator... (Computing Filtered Pairs on the Fly)")
            training_generator = FilteredPairsGenerator(metadata,
                                                        reference_scores_df,
                                                        same_prob_bins=same_prob_bins,
                                                        shuffle=True,
                                                        random_seed=args.seed,
                                                        num_turns=num_turns,
                                                        batch_size=batch_size,
                                                        use_fixed_set=False,
                                                        ignore_equal_pairs=True,
                                                        merge_on_lst=args.merge_on_lst,
                                                        mass_analyzer_lst=args.mass_analyzer_lst,
                                                        collision_energy_thresh=args.collision_energy_thresh,
                                                        skip_precursor_mz=args.no_pm_requirement)
                                                        

        else:
            logging.info("Creating DataGeneratorAllInchikeys... (Using All Pairs)")
            training_generator = DataGeneratorAllInchikeys(all_training_inchikeys, 
                                                           training_ids, 
                                                           reference_scores_df,
                                                           same_prob_bins=same_prob_bins,
                                                           shuffle=True,
                                                           random_seed=args.seed,
                                                           num_turns=num_turns,
                                                           batch_size=batch_size,
                                                           use_fixed_set=False,
                                                           ignore_equal_pairs=True)

    if args.save_format == 'pickle':
        for epoch_num in range(num_epochs):
            epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch_num}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            for batch_num, batch in enumerate(tqdm(training_generator, desc=f"Epoch {epoch_num}")):
                # Define the filename for this batch
                filename = os.path.join(epoch_dir, f'batch_{batch_num}.pkl')
                
                # Save the batch DataFrame to a pickle file
                with open(filename, 'wb') as f:
                    pickle.dump(tuple(batch), f)
    elif args.save_format == 'hdf5':
        hdf_path = os.path.join(args.save_dir, 'data.hdf5')
        with pd.HDFStore(hdf_path, 'w') as store:
            for epoch_num in range(num_epochs):
                group_name = f'epoch_{epoch_num}'
                batch_accumulator = []
                for batch_num, batch in enumerate(tqdm(training_generator, desc=f"Epoch {epoch_num}")):
                    # Save the batch DataFrame to the hdf5 file
                    batch_accumulator.extend(list(batch))
                store.put(group_name, pd.DataFrame(batch_accumulator, columns=['spectrumid1', 'spectrumid2', 'inchikey1', 'inchikey2', 'score']), format='table')
    else:
        raise ValueError("Unknown save_format")

class SelectedCompoundPairs:
    def __init__(self, pairs, inchikeys=None, shuffle=True, same_prob_bins=None, ignore_equal_pairs=True):
        
        """ Pairs is a dataframe comtaining:
        spectrum_id_1,spectrum_id_2,mass_analyzer,ionisation,
        precursor_mass_difference,structural_similarity,
        greedy_cosine,modified_cosine,matched_peaks
        """
        self.ignore_equal_pairs = ignore_equal_pairs

        # Check if pairs is symmetric
        if isinstance(pairs, pd.DataFrame):
            pair_set = set(map(tuple, pairs[['spectrum_id_1', 'spectrum_id_2']].values))
            reverse_pair_set = set(map(tuple, pairs[['spectrum_id_2', 'spectrum_id_1']].values))
            
            if pair_set != reverse_pair_set:
                print("Pairs file is not symmetric. Making it symmetric.")
                pairs = pd.concat([pairs, pairs.rename(columns={'spectrum_id_1':'spectrum_id_2',
                                                                        'spectrum_id_2':'spectrum_id_1',
                                                                        'inchikey_1':'inchikey_2',
                                                                        'inchikey_2':'inchikey_1'})], axis=0)
                print("Pairs are now symmetric.")
            del pair_set, reverse_pair_set
            gc.collect()
                
            # Make pairs a dict
            print("Building Pairs Dictionary", flush=True)
            self.pairs          = dict(tuple(pairs.groupby('inchikey_1')))
            self.pairs_keys     = set(self.pairs.keys()) # Cache for better performance

            del pairs

            # DEBUG
            print("pairs", self.pairs)

            print("Done", flush=True)
            gc.collect()
        
        # Otherwise it's an hdf pandas
        elif isinstance(pairs, pd.io.pytables.HDFStore):
            print("Recieved pairs as an HDF5 store. Assuming pairs is symmteric", flush=True)
            self.pairs = pairs
            self.pairs_keys     = set([x[1:] for x in self.pairs.keys()]) # Cache for better performance

            if shuffle:
                print("Shuffling pairs for HDF5 store is not implemented. Pairs will not be shuffled.", flush=True)
                shuffle=False
        else:
            raise ValueError(f"Expected pairs to be pd.DataFrame or pd.io.pytables.HDFStore but got {type(pairs)}")
        
        if inchikeys is None:
            print("inchikeys were not supplied, using inchikeys in pairs.")
            self.inchikeys = self.pairs.keys()
        else:
            self.inchikeys = [x[:14] for x in inchikeys]

        # No point in interating over anything we don't have pairs for
        self.inchikeys      = list(set(self.inchikeys) & self.pairs_keys)
        
        self.shuffle        = shuffle
        self.same_prob_bins = same_prob_bins
        self.curr_prob_bin  = np.random.choice(np.arange(len(self.same_prob_bins))) # We use a global one so it doesn't change for each inchikey
        
        if self.shuffle:
            # Shuffle each grouped matrix
            self._shuffle()
                
        # Group the symmetric pairs by inchikey_1, so that we can easily get all pairs for a given inchikey
        self.curr_pair_idx = {inchikey:0 for inchikey in self.inchikeys}
    
    def _shuffle(self,):
        if not self.shuffle:
            print("self.shuffle is set to false so we'll skip shuffling")
        else:
            for inchikey, group in self.pairs.items():
                self.pairs[inchikey] = group.sample(frac=1, replace=False)
    
    def reset_counts(self):
        self.curr_pair_idx = {inchikey:0 for inchikey in self.inchikeys}
    
    def next_pair_for_inchikey(self, inchikey:str):
        # Make sure we can't go off the end
        if self.ignore_equal_pairs is True:
            raise NotImplementedError("This function is only implemented for ignore_equal_pairs=False")
        idx = self.curr_pair_idx[inchikey] % len(self.pairs.get(inchikey))
        sampled_row = None
        if isinstance(self.pairs, dict):
            sampled_row = self.pairs.get(inchikey).iloc[idx]
        elif isinstance(self.pairs, pd.io.pytables.HDFStore):
            selected_data = self.pairs.select(inchikey)
            sampled_row = selected_data.iloc[idx]
            sampled_row['inchikey_1'] = inchikey
        else:
            raise ValueError("self.pairs is neither dict or pd.io.pytables.HDFStore.")
            
        return sampled_row['structural_similarity'], \
            (sampled_row['inchikey_1'], sampled_row['inchikey_2']), \
            (sampled_row['spectrum_id_1'], sampled_row['spectrum_id_2'])
    
    def _find_match_in_range(self, inchikey, target_score_range, verbose=False):
        if isinstance(self.pairs, dict):
            group = self.pairs.get(inchikey)
        elif isinstance(self.pairs, pd.io.pytables.HDFStore):
            if inchikey not in self.pairs_keys:   # There is a chance the inchi will not have pairs (e.g., weird instrument)
                print("No pairs for inchikey", inchikey, flush=True)
                return None
            group = self.pairs.select(inchikey)
            group['inchikey_1'] = inchikey
            # print(inchqikey)
            # print(group)
        else:
            raise ValueError(f"self.pairs is neither dict or pd.io.pytables.HDFStore, got type {type(self.pairs)}.")
        
        if group is None or len(group) == 0:
            # There were no pairs for this inchikey
            return None

        if self.ignore_equal_pairs:
            equal_pairs_mask = group['inchikey_1'] != group['inchikey_2']
        else:
            equal_pairs_mask = np.ones(len(group), dtype=bool)

        options = group[(group['structural_similarity'] >= target_score_range[0]) & (group['structural_similarity'] < target_score_range[1]) & equal_pairs_mask]
        if len(options) == 0:
            if verbose:
                print(f"Failed to find options for {inchikey} in range ({target_score_range[0]:.2f}, {target_score_range[1]:.2f})")
            return None
        return options.sample()

    def next_pair_for_inchikey_equal_prob(self, inchikey:str):
        same_prob_bins = self.same_prob_bins
        target_score_range = same_prob_bins[self.curr_prob_bin ]

        sampled_row = self._find_match_in_range(inchikey, target_score_range)
        if sampled_row is None:
            return None

        # We sampled something! Now we need to update the current prob bin
        self.curr_prob_bin = np.random.choice(np.arange(len(same_prob_bins)))
        return sampled_row['structural_similarity'].item(), \
                (sampled_row['inchikey_1'].item(), sampled_row['inchikey_2'].item()), \
                (sampled_row['spectrum_id_1'].item(), sampled_row['spectrum_id_2'].item())
                
    def idx_to_inchikey(self, idx):
        return list(self.inchikeys)[idx]

class SelectedCompoundPairsLowIO:
    def __init__(self, pairs, inchikeys=None, shuffle=True, same_prob_bins=None, ignore_equal_pairs=True):
        
        """ Pairs is a dataframe comtaining:
        spectrum_id_1,spectrum_id_2,mass_analyzer,ionisation,
        precursor_mass_difference,structural_similarity,
        greedy_cosine,modified_cosine,matched_peaks
        """
        self.ignore_equal_pairs = ignore_equal_pairs
        print("Initalizing SelectedCompoundPairsLowIO", flush=True)

        # Check if pairs is symmetric
        if isinstance(pairs, pd.DataFrame):
            # pair_set = set(map(tuple, pairs[['spectrum_id_1', 'spectrum_id_2']].values))
            # reverse_pair_set = set(map(tuple, pairs[['spectrum_id_2', 'spectrum_id_1']].values))
            
            # This is quite slow, and should be disabled if pairs are known not to be symmetric
            if True: #pair_set != reverse_pair_set:
                print("Pairs file is not symmetric. Making it symmetric.", flush=True)
                pairs = pd.concat([pairs, pairs.rename(columns={'spectrum_id_1':'spectrum_id_2',
                                                                        'spectrum_id_2':'spectrum_id_1',
                                                                        'inchikey_1':'inchikey_2',
                                                                        'inchikey_2':'inchikey_1'})], axis=0)
                print("Pairs are now symmetric.", flush=True)

            # del pair_set, reverse_pair_set
            gc.collect()
        
        else:
            raise ValueError(f"Expected pairs to be pd.DataFrame or pd.io.pytables.HDFStore but got {type(pairs)}")
        
        if inchikeys is None:
            print("inchikeys were not supplied, using inchikeys in pairs.", flush=True)
            self.inchikeys = np.unique(pairs.inchikey_1.values)
        else:
            self.inchikeys = [x[:14] for x in inchikeys]
        
        # Upcast pairs.structural_similarity to float 32 since float16 is not supported by pandas
        pairs['structural_similarity'] = pairs['structural_similarity'].astype(np.float32)

        # Split pairs by similarity bin
        print("Splitting pairs by similarity bin", flush=True)

        bins_for_cut = [x[0] for x in same_prob_bins] + [same_prob_bins[-1][1]]
        pair_grouped = pairs.groupby(pd.cut(pairs['structural_similarity'], bins=bins_for_cut))


        # self.pairs = pair_grouped
        self.pairs = {}
        for bin_name, group in pair_grouped:
            print(f"Processing bin {bin_name}", flush=True)
            simple_bin_name = f'({bin_name.left:.1f}, {bin_name.right:.1f}]'
            self.pairs[simple_bin_name] = group
            
        print("Done", flush=True)
        # del pair_grouped, pairs
        gc.collect()

        self.shuffle        = False
        if shuffle == True:
            print("Shuffle is set to True, but we will not shuffle the pairs as we are in low IO mode.", flush=True)
        self.same_prob_bins = same_prob_bins
        self.curr_prob_bin  = np.random.choice(np.arange(len(self.same_prob_bins))) # We use a global one so it doesn't change for each inchikey
        
        if self.shuffle:
            # Shuffle each grouped matrix
            self._shuffle()
    
    def _shuffle(self,):
        if not self.shuffle:
            print("self.shuffle is set to false so we'll skip shuffling")
        else:
            raise NotImplementedError("Shuffling is not implemented for low IO mode.")
            # self.pair = self.pairs.sample(frac=1, replace=False)
    
    # def next_pair_for_inchikey(self, inchikey:str):
    #     # Make sure we can't go off the end
    #     if self.ignore_equal_pairs is True:
    #         raise NotImplementedError("This function is only implemented for ignore_equal_pairs=False")
    #     idx = self.curr_pair_idx[inchikey] % len(self.pairs.get(inchikey))
    #     sampled_row = None
    #     if isinstance(self.pairs, dict):
    #         sampled_row = self.pairs.get(inchikey).iloc[idx]
    #     elif isinstance(self.pairs, pd.io.pytables.HDFStore):
    #         selected_data = self.pairs.select(inchikey)
    #         sampled_row = selected_data.iloc[idx]
    #         sampled_row['inchikey_1'] = inchikey
    #     else:
    #         raise ValueError("self.pairs is neither dict or pd.io.pytables.HDFStore.")
            
    #     return sampled_row['structural_similarity'], \
    #         (sampled_row['inchikey_1'], sampled_row['inchikey_2']), \
    #         (sampled_row['spectrum_id_1'], sampled_row['spectrum_id_2'])
    
    # def _find_match_in_range(self, inchikey, target_score_range, verbose=False):
    #     if isinstance(self.pairs, dict):
    #         group = self.pairs.get(inchikey)
    #     elif isinstance(self.pairs, pd.io.pytables.HDFStore):
    #         if inchikey not in self.pairs_keys:   # There is a chance the inchi will not have pairs (e.g., weird instrument)
    #             print("No pairs for inchikey", inchikey, flush=True)
    #             return None
    #         group = self.pairs.select(inchikey)
    #         group['inchikey_1'] = inchikey
    #         # print(inchqikey)
    #         # print(group)
    #     else:
    #         raise ValueError(f"self.pairs is neither dict or pd.io.pytables.HDFStore, got type {type(self.pairs)}.")
        
    #     if group is None or len(group) == 0:
    #         # There were no pairs for this inchikey
    #         return None

    #     if self.ignore_equal_pairs:
    #         equal_pairs_mask = group['inchikey_1'] != group['inchikey_2']
    #     else:
    #         equal_pairs_mask = np.ones(len(group), dtype=bool)

    #     options = group[(group['structural_similarity'] >= target_score_range[0]) & (group['structural_similarity'] < target_score_range[1]) & equal_pairs_mask]
    #     if len(options) == 0:
    #         if verbose:
    #             print(f"Failed to find options for {inchikey} in range ({target_score_range[0]:.2f}, {target_score_range[1]:.2f})")
    #         return None
    #     return options.sample()

    def next_pair(self, count=1):
        """The general idea here is that we can achieve the same effect as iterating over the alternative method.
        Before we:
        * Iterated over inchikeys
        * Looked for a match in a randomly sampled similarity range that also had spectral similarity pairs
        * If no match was found, continue on to the next key.
        The end result of this is that some inchis with sparse similarity bins will be sampled less often than
        those with complete bins.

        Instead we can:
        * Randomly sample a range
        * View which pairs fall into this range
        * Randomly sample from the inchikeys that have spectral pairs in this range
        As with before, this will be biasd towards inchis with complete similarity bins, but it means that we won't have
        to iterate over the inchikeys to find a match.

        The caveat is that this method can't be supported by the hdf5 store, as we can't easily filter the 
        data for all inchis at once.
        
        This will not explicity enforce iteration over inchikeys, but it will be the same in the limit.
        """

        same_prob_bins = self.same_prob_bins
        target_score_range = same_prob_bins[self.curr_prob_bin]

        dict_key = f'({target_score_range[0]:.1f}, {target_score_range[1]:.1f}]'

        sampled_rows = self.pairs[dict_key].sample(count)
        # Get new index for next iteration
        self.curr_prob_bin = np.random.choice(np.arange(len(same_prob_bins)))

        return [(
            row['structural_similarity'],
            (row['inchikey_1'], row['inchikey_2']),
            (row['spectrum_id_1'], row['spectrum_id_2'])
        ) for _, row in sampled_rows.iterrows()]


    # def next_pair_for_inchikey_equal_prob(self, inchikey:str):
    #     same_prob_bins = self.same_prob_bins
    #     target_score_range = same_prob_bins[self.curr_prob_bin ]

    #     sampled_row = self._find_match_in_range(inchikey, target_score_range)
    #     if sampled_row is None:
    #         return None

    #     # We sampled something! Now we need to update the current prob bin
    #     self.curr_prob_bin = np.random.choice(np.arange(len(same_prob_bins)))
    #     return sampled_row['structural_similarity'].item(), \
    #             (sampled_row['inchikey_1'].item(), sampled_row['inchikey_2'].item()), \
    #             (sampled_row['spectrum_id_1'].item(), sampled_row['spectrum_id_2'].item())
                
    # def idx_to_inchikey(self, idx):
    #     return list(self.inchikeys)[idx]
    def reset_counts(self):
        pass

class DataGeneratorAllInchikeys():
    """Generates data for training a siamese Keras model
    This generator will provide training data by picking each training InchiKey
    listed in *selected_inchikeys* num_turns times in every epoch. It will then randomly
    pick one the spectra corresponding to this InchiKey (if multiple) and pair it
    with a randomly chosen other spectrum that corresponds to a reference score
    as defined in same_prob_bins.
    """

    def __init__(self, spectrum_inchikeys: List,
                spectrum_ids: List,
                reference_scores_df: pd.DataFrame,
                same_prob_bins:np.ndarray=None,
                shuffle:bool=True,
                random_seed:int=42,
                num_turns:int=2,
                batch_size:int=32,
                use_fixed_set:bool=False,
                ignore_equal_pairs:bool=True):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        binned_spectrums : List
            List of BinnedSpectrum objects with the binned peak positions and intensities.
            respective similarity scores.
        same_prob_bins : np.ndarray, optional
            List of tuples with the same probability bins, by default None.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.
        random_seed : int, optional
            Random seed for reproducibility, by default 42.
        num_turns : int, optional
            Number of turns to generate, by default 2.
        batch_size : int, optional
            Batch size, by default 32.
        use_fixed_set : bool, optional
            Whether to use a fixed set of data, by default False.
        """
        self.spectrum_inchikeys = np.array([x[:14] for x in spectrum_inchikeys])
        self.spectrum_ids       = np.array(spectrum_ids)
        assert len(self.spectrum_inchikeys) == len(self.spectrum_ids), "Inchikeys and spectrum ids must have the same length."
        print(f"Got {len(self.spectrum_inchikeys)} inchikeys and {len(self.spectrum_ids)} spectrum ids.")
        print(f"Got {len(np.unique(self.spectrum_inchikeys))} unique inchikeys.")
        self.shuffle = shuffle
        self.reference_scores_df = reference_scores_df
        self.same_prob_bins = same_prob_bins
        self.random_seed = random_seed
        self.num_turns = num_turns
        self.batch_size = batch_size
        self.use_fixed_set = use_fixed_set
        self.ignore_equal_pairs = ignore_equal_pairs

        self.fixed_set = {}

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        NB1: self.reference_scores_df only contains 'selected' inchikeys, see `self._data_selection`.
        NB2: We don't see all data every epoch, because the last half-empty batch is omitted.
        This is expected behavior, with the shuffling this is OK.
        """
        return int(self.num_turns) * int(np.floor(len(self.reference_scores_df) / self.batch_size))

    def _find_match_in_range(self, inchikey1, target_score_range):
        """Randomly pick ID for a pair with inchikey_id1 that has a score in
        target_score_range. When no such score exists, iteratively widen the range
        in steps of 0.1.

        Parameters
        ----------
        inchikey1
            Inchikey (first 14 characters) to be paired up with another compound within
            target_score_range.
        target_score_range
            lower and upper bound of label (score) to find an ID of.
        """
        # Part 1 - find match within range (or expand range iteratively)
        extend_range = 0
        low, high = target_score_range
        inchikey2 = None
        while inchikey2 is None:
            matching_inchikeys = self.reference_scores_df.index[
                (self.reference_scores_df[inchikey1] > low - extend_range)
                & (self.reference_scores_df[inchikey1] <= high + extend_range)]
            # We will not use this setting
            if self.ignore_equal_pairs:
                matching_inchikeys = matching_inchikeys[matching_inchikeys != inchikey1]
            if len(matching_inchikeys) > 0:
                inchikey2 = np.random.choice(matching_inchikeys)
            extend_range += 0.1
        return inchikey2
        
    def _get_spectrum_with_inchikey(self, inchikey: str) -> str:
        """
        Get a random spectrum matching the `inchikey` argument. NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        matching_spectrum_id = np.where(self.spectrum_inchikeys == inchikey)[0]
        assert len(matching_spectrum_id) > 0, f"No matching inchikey found (note: expected first 14 characters) {inchikey}"
        return self.spectrum_ids[np.random.choice(matching_spectrum_id)]

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator:
        """
        Generate spectrum pairs for batch. For each 'source' inchikey pick an inchikey in the
        desired target score range. Then randomly get spectrums for this pair of inchikeys.
        """
        same_prob_bins = self.same_prob_bins
        batch_size = self.batch_size
        # Go through all indexes
        indexes = self.indexes[batch_index * batch_size:(batch_index + 1) * batch_size]

        for index in indexes:
            inchikey1 = self.reference_scores_df.index[index]
            # Randomly pick the desired target score range and pick matching inchikey
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
            inchikey2 = self._find_match_in_range(inchikey1, target_score_range)
            spectrumid1 = self._get_spectrum_with_inchikey(inchikey1)
            spectrumid2 = self._get_spectrum_with_inchikey(inchikey2)

            # Get the score from the reference scores
            score = self.reference_scores_df.loc[inchikey1, inchikey2]

            yield spectrumid1, spectrumid2, inchikey1, inchikey2, score

    @ staticmethod
    def _data_selection(reference_scores_df, selected_inchikeys):
        """
        Select labeled data to generate from based on `selected_inchikeys`
        """
        return reference_scores_df.loc[selected_inchikeys, selected_inchikeys]

    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
        
        return spectrum_pairs
    
    # def __next__(self):
    #     if self.curr_batch >= len(self):
    #         raise StopIteration()
    #     return self.__getitem__(self.curr_batch)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.reference_scores_df)), int(self.num_turns))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class DataGeneratorCherrypickedInChi():
    """Generates data for training a siamese Keras model.

    This class extends DataGeneratorBase to provide a data generator specifically
    designed for training a siamese Keras model with a curated set of compound pairs.
    It uses pre-selected compound pairs, allowing more control over the training process,
    particularly in scenarios where certain compound pairs are of specific interest or
    have higher significance in the training dataset.
    """
    def __init__(self, spectrum_inchikeys: List,
                spectrum_ids: List,
                 selected_compound_pairs: SelectedCompoundPairs,
                 same_prob_bins:np.ndarray=None,
                 shuffle:bool=True,
                 random_seed:int=42,
                 num_turns:int=2,
                 batch_size:int=32,
                 use_fixed_set:bool=False,
                 ignore_equal_pairs:bool=True):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        binned_spectrums : List
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        selected_compound_pairs : SelectedCompoundPairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        same_prob_bins : np.ndarray, optional
            List of tuples with the same probability bins, by default None.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.
        random_seed : int, optional
            Random seed for reproducibility, by default 42.
        num_turns : int, optional
            Number of turns to generate, by default 2.
        batch_size : int, optional
            Batch size, by default 32.
        use_fixed_set : bool, optional
            Whether to use a fixed set of data, by default False.
        """
        self.spectrum_inchikeys = np.array([x[:14] for x in spectrum_inchikeys])
        self.spectrum_ids       = np.array(spectrum_ids)
        self.shuffle = shuffle
        self.ignore_equal_pairs = ignore_equal_pairs

        self.selected_compound_pairs = selected_compound_pairs
        
        self.same_prob_bins = same_prob_bins
        self.random_seed = random_seed
        self.num_turns = num_turns
        self.batch_size = batch_size
        self.use_fixed_set = use_fixed_set

        # Set all other settings to input (or otherwise to defaults):
        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        if len(unique_inchikeys) < self.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.fixed_set = {}
        self.on_epoch_end()

        # Check that the settings are consistent with the SpectrumBinner
        assert self.ignore_equal_pairs == self.selected_compound_pairs.ignore_equal_pairs, "Ignore equal pairs must be the same as the spectrum binner."
        assert self.same_prob_bins == self.selected_compound_pairs.same_prob_bins, "Same prob bins must be the same as the spectrum binner."
        if self.use_fixed_set and self.shuffle:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if self.random_seed is not None:
            assert isinstance(self.random_seed, int), "Random seed must be integer number."
            np.random.seed(self.random_seed)
        
        self.curr_batch = 0
        self.curr_index = 0
        self.curr_batch_size = 0
    
    def __len__(self):
        return int(self.num_turns)\
            * int(np.floor(len(self.selected_compound_pairs.inchikeys) / self.batch_size))

    # def _get_spectrum_with_spectrumid(self, spectrumid: str):
    #     """
    #     Get a random spectrum matching the 'spectrumid' argument. NB: A compound (identified by an
    #     inchikey) can have multiple measured spectrums in a binned spectrum dataset.
    #     """
    #     # print("DEBUG")
    #     # print("Spectrum ids", self.spectrum_ids)
    #     # print("sepctrum ids shapw", self.spectrum_ids.shape)
    #     # print("spectrumid", spectrumid)
        
    #     matching_spectrum_id = np.where(self.spectrum_ids == spectrumid)[0]
        
    #     # print("Matching index:", matching_spectrum_id)
    #     assert len(matching_spectrum_id) == 1, "No matching spectrumid found "
    #     return self.binned_spectrums[matching_spectrum_id.item()]

    def _standard_spectrum_pair_generator(self, batch_index: int) -> Iterator:
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        
        batch_size = self.batch_size
        indexes = np.arange(len(self.selected_compound_pairs.inchikeys))
        indexes = indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        
        # Safety to ensure the loop terminates
        max_attempts = max(batch_size * 5, 500)
        attempt = 0
        
        while self.curr_batch_size < batch_size and attempt < max_attempts:
            if isinstance(self.selected_compound_pairs, SelectedCompoundPairs):
                inchikey1 = self.selected_compound_pairs.idx_to_inchikey(self.curr_index)
                if self.same_prob_bins is not None:
                    out = self.selected_compound_pairs.next_pair_for_inchikey_equal_prob(inchikey1)
                else:
                    out = self.selected_compound_pairs.next_pair_for_inchikey(inchikey1)
                self.curr_index += 1
                if self.curr_index >= len(self.selected_compound_pairs.inchikeys):
                    # If we fail to create a batch, just loop over to the front of the inchikeys
                    # Because during training we will loop over the dataset multiple times, with random ordering
                    # this shouldn't cause an issues in the limit
                    self.curr_index = 0
            elif isinstance(self.selected_compound_pairs, SelectedCompoundPairsLowIO):
                out = self.selected_compound_pairs.next_pair()
            else:
                raise ValueError("Unknown type of selected_compound_pairs")
            attempt += 1
            if out is None:
                # print("No pair found for inchikey", inchikey1)  # Best thing to do here is just continue to maintain equal prob
                continue
            if out is not None:
                score, (inchikey1, inchikey2), (spectrumid1, spectrumid2) = out
                self.curr_batch_size += 1
                yield spectrumid1, spectrumid2, inchikey1, inchikey2, score
        if attempt >= max_attempts:
            raise StopIteration("Unable to find a match in range.")

    def _spectrum_pair_generator_low_io(self, batch_index: int) -> Iterator:
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        
        batch_size = self.batch_size
        indexes = np.arange(len(self.selected_compound_pairs.inchikeys))
        indexes = indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        
        output_batch = []

        while self.curr_batch_size < batch_size:
            out = self.selected_compound_pairs.next_pair(int(batch_size/5))
            for o in out:
                score, (inchikey1, inchikey2), (spectrumid1, spectrumid2) = o
                output_batch.append((spectrumid1, spectrumid2, inchikey1, inchikey2, score))
                self.curr_batch_size += 1

        # Truncate output batch to the correct size
        output_batch = output_batch[:batch_size]

        for item in output_batch:
            yield item

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator:
        if isinstance(self.selected_compound_pairs, SelectedCompoundPairs):
            return self._standard_spectrum_pair_generator(batch_index)
        elif isinstance(self.selected_compound_pairs, SelectedCompoundPairsLowIO):
            return self._spectrum_pair_generator_low_io(batch_index)
       
    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        self.curr_batch_size = 0
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
        
        return spectrum_pairs
    
    def __next__(self):
        if self.curr_batch >= len(self):
            raise StopIteration()
        return self.__getitem__(self.curr_batch)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)
        
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.selected_compound_pairs._shuffle()
        self.selected_compound_pairs.reset_counts()
        self.curr_index = 0
        self.curr_batch = 0

if __name__ == "__main__":
    main()
