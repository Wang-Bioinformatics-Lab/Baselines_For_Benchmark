nohup: ignoring input
Performing Training for MS2DeepScore with simple sampling split.
Traceback (most recent call last):
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/ms2deepscore/train_presampled.py", line 13, in <module>
    from custom_spectrum_binner import SpectrumBinner
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/ms2deepscore/custom_spectrum_binner.py", line 3, in <module>
    from matchms.typing import SpectrumType
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/matchms/__init__.py", line 1, in <module>
    from matchms.filtering.SpectrumProcessor import SpectrumProcessor
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/matchms/filtering/__init__.py", line 75, in <module>
    from .default_filters import default_filters
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/matchms/filtering/default_filters.py", line 6, in <module>
    from .metadata_processing.derive_adduct_from_name import \
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/matchms/filtering/metadata_processing/derive_adduct_from_name.py", line 4, in <module>
    from matchms.filtering.filter_utils.interpret_unknown_adduct import \
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/matchms/filtering/filter_utils/interpret_unknown_adduct.py", line 10, in <module>
    from rdkit import Chem
  File "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/shared/conda_env/lib/python3.9/site-packages/rdkit/Chem/__init__.py", line 16, in <module>
    from rdkit.Chem import rdchem
KeyboardInterrupt
