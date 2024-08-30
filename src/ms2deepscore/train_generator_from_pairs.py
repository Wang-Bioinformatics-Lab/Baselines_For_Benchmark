import gc
import warnings
import numpy as np
import pandas as pd

from typing import Iterator, List, NamedTuple, Optional
from custom_spectrum_binner import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorBase, SpectrumPair
from prefetch_generator import background
import pickle

from glob import glob
import re
import os

# DEBUG
from line_profiler import profile

class SelectedCompoundPairs:
    def __init__(self, pairs, inchikeys=None, shuffle=True, same_prob_bins=None):
        
        """ Pairs is a dataframe comtaining:
        spectrum_id_1,spectrum_id_2,mass_analyzer,ionisation,
        precursor_mass_difference,structural_similarity,
        greedy_cosine,modified_cosine,matched_peaks
        """
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
            del pairs
            print("Done", flush=True)
            gc.collect()
        
        # Otherwise it's an hdf pandas
        elif isinstance(pairs, pd.io.pytables.HDFStore):
            print("Recieved pairs as an HDF5 store. Assuming pairs is symmteric", flush=True)
            self.pairs = pairs
            if shuffle:
                print("Shuffling pairs for HDF5 store is not implemented. Pairs will not be shuffled.", flush=True)
                shuffle=False
        else:
            raise ValueError(f"Expected pairs to be pd.DataFrame or pd.io.pytables.HDFStore but got {type(pairs)}")
        
        if inchikeys is None:
            print("inchikeys were not supplied, using inchikeys in pairs.")
            self.inchikeys = self.pairs.keys()
        else:
            self.inchikeys = inchikeys
        
        self.shuffle        = shuffle
        self.same_prob_bins = same_prob_bins
        
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
            group = self.pairs.select(inchikey)
            group['inchikey_1'] = inchikey
            # print(inchikey)
            # print(group)
        else:
            raise ValueError(f"self.pairs is neither dict or pd.io.pytables.HDFStore, got type {type(self.pairs)}.")
        
        options = group[(group['structural_similarity'] >= target_score_range[0]) & (group['structural_similarity'] < target_score_range[1])]
        if len(options) == 0:
            if verbose:
                print(f"Failed to find options for {inchikey} in range ({target_score_range[0]:.2f}, {target_score_range[1]:.2f})")
            return None
        return options.sample()
         
    def next_pair_for_inchikey_equal_prob(self, inchikey:str):
        same_prob_bins = self.same_prob_bins
        target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
        sampled_row = self._find_match_in_range(inchikey, target_score_range)
        if sampled_row is None:
            return None
        return sampled_row['structural_similarity'].item(), \
                (sampled_row['inchikey_1'].item(), sampled_row['inchikey_2'].item()), \
                (sampled_row['spectrum_id_1'].item(), sampled_row['spectrum_id_2'].item())
                
    def idx_to_inchikey(self, idx):
        return list(self.inchikeys)[idx]
    
class PrebatchedGenerator(DataGeneratorBase):
    # This generator uses a feather file, which already contains the spectra
    @staticmethod
    def _extract_epoch_number(epoch_dir):
        match = re.search(r'epoch_(\d+)', epoch_dir)
        if match:
            return int(match.group(1))
        raise ValueError()
    @staticmethod
    def _extract_batch_number(batch_dir):
        match = re.search(r'batch_(\d+)', batch_dir)
        if match:
            return int(match.group(1))
        raise ValueError()
    
    def __init__(self, prebatched_data_dir):
        self.data_dir = prebatched_data_dir
        
        all_epochs = glob(prebatched_data_dir + '/epoch_*')
        print(f"Found {len(all_epochs)} epochs of data.")
        # Sort epochs by number
        all_epochs = sorted(all_epochs, key=self._extract_epoch_number)
        
        self.epoch_lst = all_epochs
        self.curr_epoch = 0
        curr_epoch_batches = glob(self.epoch_lst[self.curr_epoch] + '/*.pkl')
        self.curr_epoch_batches = sorted(curr_epoch_batches, key=self._extract_batch_number)
    
    def __len__(self,):
        # Return length of first batch
        return len(self.curr_epoch_batches)
    
    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        
        return pickle.load(open(self.curr_epoch_batches[batch_index], 'rb'))

class PrebatchedGeneratorHDF5(DataGeneratorBase):
    # This generator uses and HDF5 file, which does not contain the spectra. The spectra must
    # be loaded from a separate source.
    @staticmethod
    def _extract_epoch_number(epoch_dir):
        match = re.search(r'epoch_(\d+)', epoch_dir)
        if match:
            return int(match.group(1))
        raise ValueError()
    # @staticmethod
    # def _extract_batch_number(batch_dir):
    #     match = re.search(r'batch_(\d+)', batch_dir)
    #     if match:
    #         return int(match.group(1))
    #     raise ValueError()
    
    def __init__(self, prebatched_data_path, binned_spectra, spectrum_binner,  **settings):
        self.data_dir = prebatched_data_path

        # Attributes required for _data_generation
        self.dim = len(spectrum_binner.known_bins)
        additional_metadata = spectrum_binner.additional_metadata
        if len(additional_metadata) > 0:
            self.additional_metadata = \
                [additional_feature_type.to_json() for additional_feature_type in additional_metadata]
        else:
            self.additional_metadata = ()

        # Convert spectra to a dict by spectrum_id
        spectra_dict = {s.get("spectrum_id"): s for s in binned_spectra}

        self.spec_dict = spectra_dict
        
        self.hdf_store = pd.HDFStore(prebatched_data_path)
        keys = self.hdf_store.keys()
        # Filter keys by regex 'epoch_<number>'
        keys = [key for key in keys if re.match(r'^/epoch_\d+$', key)]
        # Sort keys by number
        self.keys = sorted(keys, key=self._extract_epoch_number)
        print(f"Found {len(self.keys)} epochs of data.")
        
        self.curr_epoch = 0
        self.curr_loaded_epoch = None
        self.epoch_data = None

        self._set_generator_parameters(**settings)
    
    def __len__(self,):
        # Return length of first batch
        self._load_epoch_data(self.curr_epoch)
        return int(len(self.epoch_data)/self.settings['batch_size'])

    def _load_epoch_data(self, epoch_index: int):
        """Loads the epoch data into memory."""
        if self.curr_loaded_epoch is None or self.curr_loaded_epoch != epoch_index:
            self.epoch_data = self.hdf_store[self.keys[epoch_index]]
            self.curr_loaded_epoch = epoch_index

    
    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        self._load_epoch_data(self.curr_epoch)  # Only load the hdf table once per epoch

        start_index = batch_index * self.settings['batch_size']
        end_index = (batch_index + 1) * self.settings['batch_size']

        batch = self.epoch_data.iloc[start_index:end_index]
        
        # batch is a dataframe with the columns spectrumid1, spectrumid2, inchikey1, inchikey2, and score
        # We need to transform this to a list of tuples where the first and second elements are spectra
        # and the third element is the score

        output = []
        for _, row in batch.iterrows():
            if row['spectrumid1'] not in self.spec_dict:            
                # print(f"Failed to find spectrum with id {row['spectrumid1']}")
                continue
            if row['spectrumid2'] not in self.spec_dict:
                # print(f"Failed to find spectrum with id {row['spectrumid2']}")
                continue
            spectrum1 = self.spec_dict[row['spectrumid1']]
            spectrum2 = self.spec_dict[row['spectrumid2']]
            output.append((spectrum1, spectrum2, row['score']))

        return self._data_generation(output)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if not self.settings['use_fixed_set']:
            self.curr_epoch +=1

    def _set_generator_parameters(self, **settings):
        """Set parameter for data generator. Use below listed defaults unless other
        input is provided.

        Parameters
        ----------
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey14 during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_removal_max
            Maximum fraction of peaks (if intensity < below augment_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_removal_intensity
            will be removed.
        augment_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].
        augment_noise_max
            Max number of 'new' noise peaks to add to the spectrum, between 0 to `augment_noise_max`
            of peaks are added.
        augment_noise_intensity
            Intensity of the 'new' noise peaks to add to the spectrum
        use_fixed_set
            Toggles using a fixed dataset, if set to True the same dataset will be generated each
            epoch. Default is False.
        random_seed
            Specify random seed for reproducible random number generation.
        additional_inputs
            Array of additional values to be used in training for e.g. ["precursor_mz", "parent_mass"]
        """
        defaults = {
            "batch_size": 32,
            "num_turns": 1,
            "ignore_equal_pairs": True,
            "shuffle": True,
            "same_prob_bins": None,
            "augment_removal_max": 0.3,
            "augment_removal_intensity": 0.2,
            "augment_intensity": 0.4,
            "augment_noise_max": 10,
            "augment_noise_intensity": 0.01,
            "use_fixed_set": False,
            "random_seed": None,
        }

        # Set default parameters or replace by **settings input
        for key, value in defaults.items():
            if key in settings:
                print(f"The value for {key} is set from {value} (default) to {settings[key]}")
            else:
                settings[key] = defaults[key]
                
        assert 0.0 <= settings["augment_removal_max"] <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= settings["augment_removal_intensity"] <= 1.0, "Expected value within [0,1]"
        if settings["use_fixed_set"] and settings["shuffle"]:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if settings["random_seed"] is not None:
            assert isinstance(settings["random_seed"], int), "Random seed must be integer number."
            np.random.seed(settings["random_seed"])
        self.settings = settings
        
        self.curr_batch = 0
        self.curr_index = 0
        self.curr_batch_size = 0

class DataGeneratorCherrypickedInChi(DataGeneratorBase):
    """Generates data for training a siamese Keras model.

    This class extends DataGeneratorBase to provide a data generator specifically
    designed for training a siamese Keras model with a curated set of compound pairs.
    It uses pre-selected compound pairs, allowing more control over the training process,
    particularly in scenarios where certain compound pairs are of specific interest or
    have higher significance in the training dataset.
    """
    def __init__(self, binned_spectrums: List,
                 selected_compound_pairs: SelectedCompoundPairs,
                 spectrum_binner: SpectrumBinner,
                 **settings):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        binned_spectrums
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        selected_compound_pairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        spectrum_binner
            SpectrumBinner which was used to convert the data to the binned_spectrums.
        settings
            The available settings can be found in GeneratorSettings
        """
        self.binned_spectrums = binned_spectrums
        # Collect all inchikeys
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.binned_spectrums])
        self.spectrum_ids       = np.array([s.get("spectrum_id") for s in self.binned_spectrums])

        self.spectrum_binner = spectrum_binner
        self.selected_compound_pairs = selected_compound_pairs
        
        # Set all other settings to input (or otherwise to defaults):
        self._set_generator_parameters(**settings)
        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        if len(unique_inchikeys) < self.settings['batch_size']:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.dim = len(spectrum_binner.known_bins)
        additional_metadata = spectrum_binner.additional_metadata
        if len(additional_metadata) > 0:
            self.additional_metadata = \
                [additional_feature_type.to_json() for additional_feature_type in additional_metadata]
        else:
            self.additional_metadata = ()
        self.fixed_set = {}
        self.on_epoch_end()
    
    def _set_generator_parameters(self, **settings):
        """Set parameter for data generator. Use below listed defaults unless other
        input is provided.

        Parameters
        ----------
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey14 during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_removal_max
            Maximum fraction of peaks (if intensity < below augment_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_removal_intensity
            will be removed.
        augment_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].
        augment_noise_max
            Max number of 'new' noise peaks to add to the spectrum, between 0 to `augment_noise_max`
            of peaks are added.
        augment_noise_intensity
            Intensity of the 'new' noise peaks to add to the spectrum
        use_fixed_set
            Toggles using a fixed dataset, if set to True the same dataset will be generated each
            epoch. Default is False.
        random_seed
            Specify random seed for reproducible random number generation.
        additional_inputs
            Array of additional values to be used in training for e.g. ["precursor_mz", "parent_mass"]
        """
        defaults = {
            "batch_size": 32,
            "num_turns": 1,
            "ignore_equal_pairs": True,
            "shuffle": True,
            "same_prob_bins": None,
            "augment_removal_max": 0.3,
            "augment_removal_intensity": 0.2,
            "augment_intensity": 0.4,
            "augment_noise_max": 10,
            "augment_noise_intensity": 0.01,
            "use_fixed_set": False,
            "random_seed": None,
        }

        # Set default parameters or replace by **settings input
        for key, value in defaults.items():
            if key in settings:
                print(f"The value for {key} is set from {value} (default) to {settings[key]}")
            else:
                settings[key] = defaults[key]
                
        assert settings["same_prob_bins"] == self.selected_compound_pairs.same_prob_bins, "Same prob bins must be the same as the spectrum binner."
        assert 0.0 <= settings["augment_removal_max"] <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= settings["augment_removal_intensity"] <= 1.0, "Expected value within [0,1]"
        if settings["use_fixed_set"] and settings["shuffle"]:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if settings["random_seed"] is not None:
            assert isinstance(settings["random_seed"], int), "Random seed must be integer number."
            np.random.seed(settings["random_seed"])
        self.settings = settings
        
        self.curr_batch = 0
        self.curr_index = 0
        self.curr_batch_size = 0
    
    def __len__(self):
        return int(self.settings['num_turns'])\
            * int(np.floor(len(self.selected_compound_pairs.inchikeys) / self.settings['batch_size']))

    def _get_spectrum_with_spectrumid(self, spectrumid: str):
        """
        Get a random spectrum matching the 'spectrumid' argument. NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        # print("DEBUG")
        # print("Spectrum ids", self.spectrum_ids)
        # print("sepctrum ids shapw", self.spectrum_ids.shape)
        # print("spectrumid", spectrumid)
        
        matching_spectrum_id = np.where(self.spectrum_ids == spectrumid)[0]
        
        # print("Matching index:", matching_spectrum_id)
        assert len(matching_spectrum_id) == 1, "No matching spectrumid found "
        return self.binned_spectrums[matching_spectrum_id.item()]

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator[SpectrumPair]:
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        # batch_size = self.settings['batch_size']
        # indexes = np.arange(len(self.selected_compound_pairs.pairs.keys()))
        # indexes = indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        # for index in indexes:
        #     inchikey1 = self.selected_compound_pairs.idx_to_inchikey(index)
        #     if self.settings['same_prob_bins'] is not None:
        #         out = self.selected_compound_pairs.next_pair_for_inchikey_equal_prob(inchikey1)
        #     else:
        #         out = self.selected_compound_pairs.next_pair_for_inchikey(inchikey1)
        #     if out is None:
        #         print("No pair found for inchikey", inchikey1)  # Best thing to do here is just continue to maintain equal prob
        #         continue
        #     if out is not None:
        #         score, (inchikey1, inchikey2), (spectrumid1, spectrumid2) = out
        #         spectrum1 = self._get_spectrum_with_spectrumid(spectrumid1)
        #         spectrum2 = self._get_spectrum_with_spectrumid(spectrumid2)
        #         print("Returning", spectrum1, spectrum2, score)
        #         yield SpectrumPair(spectrum1, spectrum2, score)
        # raise StopIteration()
        
        batch_size = self.settings['batch_size']
        indexes = np.arange(len(self.selected_compound_pairs.inchikeys))
        indexes = indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        
        # Safety to ensure the loop terminates
        max_attempts = batch_size * 5
        attempt = 0
        
        while self.curr_batch_size < batch_size and attempt < max_attempts:
            inchikey1 = self.selected_compound_pairs.idx_to_inchikey(self.curr_index)
            if self.settings['same_prob_bins'] is not None:
                out = self.selected_compound_pairs.next_pair_for_inchikey_equal_prob(inchikey1)
            else:
                out = self.selected_compound_pairs.next_pair_for_inchikey(inchikey1)
            self.curr_index += 1
            if self.curr_index >= len(self.selected_compound_pairs.inchikeys):
                # If we fail to create a batch, just loop over to the front of the inchikeys
                # Because during training we will loop over the dataset multiple times, with random ordering
                # this shouldn't cause an issues in the limit
                self.curr_index = 0
            attempt += 1
            if out is None:
                # print("No pair found for inchikey", inchikey1)  # Best thing to do here is just continue to maintain equal prob
                continue
            if out is not None:
                score, (inchikey1, inchikey2), (spectrumid1, spectrumid2) = out
                spectrum1 = self._get_spectrum_with_spectrumid(spectrumid1)
                spectrum2 = self._get_spectrum_with_spectrumid(spectrumid2)
                # print("Returning", spectrum1, spectrum2, score)
                self.curr_batch_size += 1
                yield SpectrumPair(spectrum1, spectrum2, score)
        if attempt >= max_attempts:
            raise StopIteration("Unable to find a match in range.")
     
    @profile           
    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.settings['use_fixed_set'] and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        self.curr_batch_size = 0
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
        
        X, y = self._data_generation(spectrum_pairs)
        if self.settings['use_fixed_set']:
            # Store batches for later epochs
            self.fixed_set[batch_index] = (X, y)
            
            
        # print("__getitem__", X, y)
        return X, y
    
    def __next__(self):
        if self.curr_batch >= len(self):
            raise StopIteration()
        
        return self.__getitem__(self.curr_batch)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)
        

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.settings['shuffle']:
            self.selected_compound_pairs._shuffle()
        self.selected_compound_pairs.reset_counts()
        self.curr_index = 0
        self.curr_batch = 0