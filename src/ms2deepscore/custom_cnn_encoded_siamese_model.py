from pathlib import Path
from typing import Tuple, Union
import h5py
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, concatenate  # pylint: disable=import-error

from ms2deepscore import SpectrumBinner

# Import for CNN encoder:
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Flatten

class SiameseModel:
    """
    Class for training and evaluating a siamese neural network, implemented in Tensorflow Keras.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.

    Mimics keras.Model API.

    For example:

    .. code-block:: python

        # Import data and reference scores --> spectrums & tanimoto_scores_df

        # Create binned spectrums
        spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
        binned_spectrums = spectrum_binner.fit_transform(spectrums)

        # Create generator
        dimension = len(spectrum_binner.known_bins)
        test_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                                   dim=dimension)

        # Create (and train) a Siamese model
        model = SiameseModel(spectrum_binner, base_dims=(600, 500, 400), embedding_dim=400,
                             dropout_rate=0.2)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        model.summary()
        model.fit(test_generator,
                  validation_data=test_generator,
                  epochs=50)

    """

    def __init__(self,
                 spectrum_binner: SpectrumBinner,
                 base_dims: Tuple[int, ...] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5,
                 dropout_in_first_layer: bool = False,
                 l1_reg: float = 1e-6,
                 l2_reg: float = 1e-6,
                 keras_model: keras.Model = None,
                 additional_input=0):
        """
        Construct SiameseModel

        Parameters
        ----------
        spectrum_binner
            SpectrumBinner which is used to bin the spectra data for the model training.
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model.
        dropout_in_first_layer
            Set to True if dropout should be part of first dense layer as well. Default is False.
        l1_reg
            L1 regularization rate. Default is 1e-6.
        l2_reg
            L2 regularization rate. Default is 1e-6.
        keras_model
            When provided, this keras model will be used to construct the SiameseModel instance.
            Default is None.
        additional_input
            Shape of additional inputs to be used in the model. Default is 0.
        """
        # pylint: disable=too-many-arguments
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.input_dim = len(spectrum_binner.known_bins)
        self.additional_input = additional_input

        if keras_model is None:
            # Create base model
            self.base = self.get_base_model(input_dim=self.input_dim,
                                            base_dims=base_dims,
                                            embedding_dim=embedding_dim,
                                            dropout_rate=dropout_rate,
                                            dropout_in_first_layer=dropout_in_first_layer,
                                            l1_reg=l1_reg,
                                            l2_reg=l2_reg,
                                            additional_input=additional_input)
            # Create head model
            self.model = self._get_head_model(input_dim=self.input_dim,
                                              additional_input=additional_input,
                                              base_model=self.base)
        else:
            self._construct_from_keras_model(keras_model)

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.

        Parameters
        ----------
        filename
            Filename to specify where to store the model.

        """
        self.model.save(filename, save_format="h5")
        with h5py.File(filename, mode='a') as f:
            f.attrs['spectrum_binner'] = self.spectrum_binner.to_json()
            f.attrs['additional_input'] = self.additional_input

    @staticmethod
    def get_base_model(input_dim: int,
                    filters: Tuple[int, ...] = (32, 64, 128, 256, 512),
                    kernel_sizes: Tuple[int, ...] = (3, 3, 5, 5, 5),
                    strides: Tuple[int, ...] = (1, 1, 2, 2, 2),
                    base_dims: Tuple[int, ...] = (600, 500, 500),
                    embedding_dim: int = 400,
                    dropout_rate: float = 0.25,
                    dropout_in_first_layer: bool = False,
                    l1_reg: float = 1e-6,
                    l2_reg: float = 1e-6,
                    dropout_always_on: bool = False,
                    additional_input=0) -> keras.Model:

        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        base_input = Input(shape=(input_dim, 1), name='base_input')
        flattened_base_input = Flatten()(base_input)
        
        # CNN Layers
        model_layer = base_input
        for i, (filter_size, kernel_size, stride) in enumerate(zip(filters, kernel_sizes, strides)):
            model_layer = Conv1D(filters=filter_size, kernel_size=kernel_size, strides=stride, activation='relu')(model_layer)
            model_layer = MaxPooling1D(pool_size=2)(model_layer)
        cnn_out = (model_layer)  # Output from CNN

        # Side-input to MLP
        if additional_input > 0:
            side_input = Input(shape=additional_input, name="additional_input")
            model_input = concatenate([cnn_out, flattened_base_input, side_input], axis=1)
        else:
            model_input = concatenate([cnn_out, flattened_base_input], axis=1)

        # MLP Layers
        for i, dim in enumerate(base_dims):
            model_layer = Dense(dim, activation='relu', name='dense'+str(i+1))(model_input if i==0 else model_layer)
            model_layer = BatchNormalization(name='normalization'+str(i+1))(model_layer)
            if dropout_always_on and i >= dropout_starting_layer:
                model_layer = Dropout(dropout_rate, name='dropout'+str(i+1))(model_layer, training=True)
            elif i >= dropout_starting_layer:
                model_layer = Dropout(dropout_rate, name='dropout'+str(i+1))(model_layer)

        embedding = Dense(embedding_dim, activation='relu', name='embedding')(model_layer)
        
        if additional_input > 0:
            return keras.Model(inputs=[base_input, side_input], outputs=[embedding], name='base')

        return keras.Model(inputs=[base_input], outputs=[embedding], name='base')

    @staticmethod
    def _get_head_model(input_dim: int,
                        additional_input: int,
                        base_model: keras.Model):
        input_a = Input(shape=input_dim, name="input_a")
        input_b = Input(shape=input_dim, name="input_b")

        if additional_input > 0:
            input_a_2 = Input(shape=additional_input, name="input_a_2")
            input_b_2 = Input(shape=additional_input, name="input_b_2")
            inputs = [input_a, input_a_2, input_b, input_b_2]

            embedding_a = base_model([input_a, input_a_2])
            embedding_b = base_model([input_b, input_b_2])
        else:
            embedding_a = base_model(input_a)
            embedding_b = base_model(input_b)
            inputs = [input_a, input_b]

        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])

        return keras.Model(inputs=inputs, outputs=[cosine_similarity], name='head')

    def _construct_from_keras_model(self, keras_model):
        def valid_keras_model(given_model):
            assert given_model.layers, "Expected valid keras model as input."
            assert len(given_model.layers) > 2, "Expected more layers"
            if self.additional_input > 0:
                assert keras_model.layers[4], "Expected more layers for base model"
            else: 
                assert len(keras_model.layers[2].layers) > 1, "Expected more layers for base model"

        valid_keras_model(keras_model)
        self.base = keras_model.layers[2]
        if self.additional_input > 0:
            self.base = keras_model.layers[4]
        self.model = keras_model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def summary(self):
        self.base.summary()
        self.model.summary()

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
