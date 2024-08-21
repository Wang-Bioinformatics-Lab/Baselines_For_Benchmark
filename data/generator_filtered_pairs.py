""" Data generators for training/inference with siamese Keras model.
"""
import warnings
from typing import List, Iterator, NamedTuple, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence  # pylint: disable=import-error

from ms2deepscore.typing import BinnedSpectrumType
from ms2deepscore.data_generators import DataGeneratorBase

from tqdm import tqdm
from joblib import Parallel, delayed

class FilteredPairsGenerator(DataGeneratorBase):
    """Generates data for training a siamese Keras model
    This generator will provide training data by picking each training InchiKey
    listed in *selected_inchikeys* num_turns times in every epoch. It will then randomly
    pick one the spectra corresponding to this InchiKey (if multiple) and pair it
    with a randomly chosen other spectrum that corresponds to a reference score
    as defined in same_prob_bins.
    """

    def __init__(self, metadata_table: pd.DataFrame,
                reference_scores_df: pd.DataFrame,
                same_prob_bins:np.ndarray=None,
                shuffle:bool=True,
                random_seed:int=42,
                num_turns:int=2,
                batch_size:int=32,
                use_fixed_set:bool=False,
                ignore_equal_pairs:bool=True,
                merge_on_lst=None,
                mass_analyzer_lst=None,
                collision_energy_thresh:float=5.0,
                strict_collision_energy=False):
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

        skip_precursor_mz=False

        # Set parameters for filtering
        if merge_on_lst is not None:
            for x in merge_on_lst.split(';').lower():
                assert x in ['ms_mass_analyzer', 'ms_ionisation', 'adduct', 'library']
        else:
            merge_on_lst = ['ms_mass_analyzer', 'ms_ionisation', 'adduct']
        self.merge_on_lst = merge_on_lst
        print("Filtering Pairs by:", merge_on_lst)
        self.collision_energy_thresh = collision_energy_thresh
        print("Collision Energy Threshold:", collision_energy_thresh)
        self.skip_precursor_mz = skip_precursor_mz
        print("Require Collision Energy:", strict_collision_energy)
        self.strict_collision_energy = strict_collision_energy
        print("Skip precursor m/z:", skip_precursor_mz)
        if mass_analyzer_lst is not None:
            mass_analyzer_lst = mass_analyzer_lst.split(';')
            mass_analyzer_lst = [str(x).strip().lower() for x in mass_analyzer_lst]
        else:
            mass_analyzer_lst = None
        self.mass_analyzer_lst = mass_analyzer_lst
        print("Allowed Mass Analyzers:", mass_analyzer_lst)

        # Filter metadata_table by criteria
        print(f"Begining with {metadata_table.shape[0]} spectra.")
        if 'ms_mass_analyzer' in merge_on_lst:
            metadata_table = metadata_table.dropna(subset=['msMassAnalyzer'])
            print(f"Filtered by mass analyzer (was nan): {metadata_table.shape[0]} spectra.")
        if 'ms_ionisation' in merge_on_lst:
            metadata_table = metadata_table.dropna(subset=['msIonisation'])
            print(f"Filtered by ionisation (was nan): {metadata_table.shape[0]} spectra.")
        # Right now, we don't have enough collision energy data to filter by it, therefore we allow pairs with nan collision energy
        if self.strict_collision_energy:
            metadata_table = metadata_table.dropna(subset=['collisionEnergy'])
            print(f"Filtered by collision energy (was nan): {metadata_table.shape[0]} spectra.")

        if mass_analyzer_lst is not None:
            metadata_table = metadata_table[metadata_table['msMassAnalyzer'].isin(mass_analyzer_lst)]
            print(f"Filtered by mass analyzer type: {metadata_table.shape[0]} spectra.")
        print(f"{metadata_table.shape[0]} spectra left after filtering.")
       
        self.spectrum_inchikeys = metadata_table["inchikey_14"]
        self.spectrum_ids       = metadata_table["spectrum_id"]

        if self.skip_precursor_mz:
            unique_inchikeys = np.unique(self.spectrum_inchikeys)
            self.precursor_mass_diff_df = pd.DataFrame(np.ones((len(unique_inchikeys),len(unique_inchikeys)), dtype=bool), index=unique_inchikeys, columns=unique_inchikeys)
        else:
            # Compute precursor_mass_diff_df, based on inchikey_14
            mass_df = metadata_table.groupby('inchikey_14')
            # Sanity check to make sure ExactMass does not differ significantly
            exact_mass_std = mass_df['ExactMass'].std()
            if not np.any(exact_mass_std < 0.1):    # equal_nan=True for single instances
                raise ValueError(f"ExactMass differs significantly: {exact_mass_std.sort_values(ascending=False).head(5)}")
            # Assign average mass for each inchikey_14
            precursor_mass_diff_df = mass_df['ExactMass'].mean()
            del mass_df
            # Get square dataframe of mass differences for all pairs
            mass_diff_arr = np.abs(np.subtract.outer(precursor_mass_diff_df.values, precursor_mass_diff_df.values))
            precursor_mass_diff_df = pd.DataFrame(mass_diff_arr, index=precursor_mass_diff_df.index, columns=precursor_mass_diff_df.index)
            self.precursor_mass_diff_df = precursor_mass_diff_df<200
            del mass_diff_arr, precursor_mass_diff_df

            print(f"Total number of pairs with precursor mass difference < 200: {int(np.sum(self.precursor_mass_diff_df.values))}")

        # Create MultiIndex for faster lookup
        metadata_table.set_index(['msMassAnalyzer', 'msIonisation', 'Adduct'], inplace=True)
        # metadata_table.sort_index(inplace=True)

        # Create a dictonary of the metadata table by inchikey for faster lookup by single inchikey
        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        self.metadata_dict = {inchikey: metadata_table.loc[metadata_table.inchikey_14 == inchikey] for inchikey in unique_inchikeys}

        assert len(self.spectrum_inchikeys) == len(self.spectrum_ids), "Inchikeys and spectrum ids must have the same length."
        print(f"Got {len(self.spectrum_inchikeys)} inchikeys and {len(self.spectrum_ids)} spectrum ids.")
        print(f"Got {len(np.unique(self.spectrum_inchikeys))} unique inchikeys.")
        self.shuffle = shuffle
        self.reference_scores_df = self._data_selection(reference_scores_df, self.spectrum_inchikeys.unique())
        self.same_prob_bins = same_prob_bins
        self.random_seed = random_seed
        # Set random seed
        np.random.seed(random_seed)
        self.num_turns = num_turns
        self.batch_size = batch_size
        self.use_fixed_set = use_fixed_set
        self.ignore_equal_pairs = ignore_equal_pairs
        self.curr_index = 0

        self.reference_scores_dict = self._precompute_valid_reference_scores()

        self.fixed_set = {}

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        NB1: self.reference_scores_df only contains 'selected' inchikeys, see `self._data_selection`.
        NB2: We don't see all data every epoch, because the last half-empty batch is omitted.
        This is expected behavior, with the shuffling this is OK.
        """
        return int(self.num_turns) * int(np.floor(len(self.reference_scores_df) / self.batch_size))

    def _precompute_valid_reference_scores(self,) -> dict:
        """It turns out doing these lookups takes ~ 0.006 seconds, which since they're repeated frequently
        can become quite time consuming. Therefore, we precompute and cache them with the object.
        """
        output_dict = {}
        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        print(f"Precomputing valid reference scores for {len(unique_inchikeys)} inchikeys.")
        for inchikey in tqdm(unique_inchikeys):
            matching_inchikeys_exact_mass = self.precursor_mass_diff_df[self.precursor_mass_diff_df.loc[:, inchikey]].index
            if len(matching_inchikeys_exact_mass) == 0:
                continue
            selected_reference_scores_df = self.reference_scores_df.loc[matching_inchikeys_exact_mass, inchikey]
            
            output_dict[inchikey] = selected_reference_scores_df

        return output_dict

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

        local_reference_scores_df = self.reference_scores_dict[inchikey1]

        while inchikey2 is None:
            matching_inchikeys = self.reference_scores_df.index[
                (self.reference_scores_df[inchikey1] > low - extend_range)
                & (self.reference_scores_df[inchikey1] <= high + extend_range)]

            matching_inchikeys = local_reference_scores_df.loc[(local_reference_scores_df > low - extend_range) &
                                                               (local_reference_scores_df <= high + extend_range)].index

            if self.ignore_equal_pairs:
                matching_inchikeys = matching_inchikeys[matching_inchikeys != inchikey1]
            if len(matching_inchikeys) > 0:
                inchikey2 = np.random.choice(matching_inchikeys)
            # DEBUG
            # else:
            #     return None

            extend_range += 0.1
        return inchikey2

    def get_spectrum_pair_with_inchikey(self, inchikey1:str, inchikey2:str,
                                        return_type='ids',
                                        return_all = False):

        # A short cut for the decision version of this problem
        if return_type not in ['bool', 'ids']:
            raise ValueError(f"Invalid return_type: {return_type}, must be 'bool' or 'ids'.")
        if return_type != 'ids' and return_all:
            raise ValueError(f"return_all=True only works for return_type='ids'.")

        # Part 1 - Get all spectra that match this criteria
        inchikey1_matching_spectra = self.metadata_dict.get(inchikey1)
        if inchikey1_matching_spectra is None:
            return None, None
            raise ValueError(f"No matching inchikey found for {inchikey1}")

        # Part 2 - Get the matching spectrum ID for inchikey2
        inchikey2_matching_spectra = self.metadata_dict.get(inchikey2)
        if inchikey2_matching_spectra is None:
            return None, None
            raise ValueError(f"No matching inchikey found for {inchikey2}")

        # Part 3 - Get the intersection of the two instrument metadata values
        matching_conditions = inchikey1_matching_spectra.index.intersection(inchikey2_matching_spectra.index).unique()

        if len(matching_conditions) == 0:
            if return_type == 'bool':
                return False
            return None, None

        
        inchikey1_matching_spectra = inchikey1_matching_spectra.loc[matching_conditions]
        inchikey2_matching_spectra = inchikey2_matching_spectra.loc[matching_conditions]

        # Part 4 - For each intersection, get matches with collision energe +/- collision_energy_thresh eV
        possible_matches = []

        # print(f"Got {len(matching_conditions)} matching conditions for {inchikey1} and {inchikey2}")

        for idx in matching_conditions:
            _inchikey1_matching_spectra = inchikey1_matching_spectra.loc[[idx]]
            _inchikey2_matching_spectra = inchikey2_matching_spectra.loc[[idx]]

            ce1 = _inchikey1_matching_spectra.loc[:, 'collisionEnergy'].values
            ce2 = _inchikey2_matching_spectra.loc[:, 'collisionEnergy'].values

            if return_type == 'bool':
                if np.any(np.isnan(ce1)) or np.any(np.isnan(ce2)):
                    # Shortcut out without doing other calculations for decision to save time
                    return True

            # Add pairs with missing collsion energy to the valid pairs
            missing1 = np.isnan(ce1).nonzero()[0]
            missing2 = np.isnan(ce2).nonzero()[0]

            # Outer subtraction to get the absolute difference
            diff = np.abs(np.subtract.outer(ce1, ce2)) <= self.collision_energy_thresh
            valid_pairs = np.argwhere(diff)
            if return_type == 'bool':
                if len(valid_pairs) > 0:
                    return True

            missing_pairs = np.array(np.meshgrid(missing1, missing2)).T.reshape(-1, 2)
            valid_pairs = np.concatenate([valid_pairs, missing_pairs], axis=0)

            # Add all pairs to the possible matches
            if len(valid_pairs) > 0:
                if return_type == 'bool':
                    return True
                for pair in valid_pairs:
                    spectrum_id1 = _inchikey1_matching_spectra.iloc[pair[0]]['spectrum_id']
                    spectrum_id2 = _inchikey2_matching_spectra.iloc[pair[1]]['spectrum_id']
                    possible_matches.append((spectrum_id1, spectrum_id2))

        # Randomly pick one of the possible matches
        if len(possible_matches) == 0:
            if return_type == 'bool':
                return False
            return None, None

        if return_all:
            return possible_matches

        spectrum_id1, spectrum_id2 = possible_matches[np.random.choice(len(possible_matches))]

        return spectrum_id1, spectrum_id2

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
        curr_batch_size = 0
        batch_data = []
        # Go through all indexes
        # indexes = self.indexes[batch_index * batch_size:(batch_index + 1) * batch_size]

        target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
        while curr_batch_size < batch_size:
            index = self.indexes[self.curr_index]
            self.curr_index += 1
            if self.curr_index >= len(self.reference_scores_df):
                self.curr_index = 0
            inchikey1 = self.reference_scores_df.index[index]
            # Randomly pick the desired target score range and pick matching inchikey
            inchikey2 = self._find_match_in_range(inchikey1, target_score_range)
            if inchikey2 is None:
                # DEBUG
                continue

            spectrumid1, spectrumid2 = self.get_spectrum_pair_with_inchikey(inchikey1, inchikey2)
            if spectrumid2 is None:
                # These inchikeys have no match, continue through the batch with the same target_score_range
                continue

            curr_batch_size += 1
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]

            # Get the score from the reference scores
            score = self.reference_scores_df.loc[inchikey1, inchikey2]

            batch_data.append((spectrumid1, spectrumid2, inchikey1, inchikey2, score))
        
        # Since we're not using a subprocess, it's probably slightly more efficent to get the data in one chunk
        # then yield it in the generator
        for data in batch_data:
            yield data

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
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.reference_scores_df)), int(self.num_turns))
        if self.shuffle:
            np.random.shuffle(self.indexes)