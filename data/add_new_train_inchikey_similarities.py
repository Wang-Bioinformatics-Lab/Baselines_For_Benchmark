import argparse
import os
import pandas as pd
import numpy as np

def add_new_train_inchikey_similarities(old_train_inchikey_similarities_path,
                                        new_train_inchikey_similarities_path,
                                        output_train_inchikey_similarities_path):
    new_train_inchikey_similarities = pd.read_csv(new_train_inchikey_similarities_path, index_col=0)
    old_train_inchikey_similarities = pd.read_csv(old_train_inchikey_similarities_path, index_col=0)
    
    # Add new similarities to old similarities, we only have to add the new inchikeys
    # Because we're training on filtered pairs, we can leave pairs between old and new as nan
    # Note that both csvs should be square matrices
    
    old_rows    = list(old_train_inchikey_similarities.index)
    old_columns = list(old_train_inchikey_similarities.columns)
    old_rows_set = set(old_rows)
    old_columns_set = set(old_columns)
    assert len(old_rows_set) == len(old_rows)
    assert old_rows_set == old_columns_set
    new_rows = set(new_train_inchikey_similarities.index)
    new_columns = set(new_train_inchikey_similarities.columns)
    assert new_rows == new_columns
    
    new_ids = new_rows - old_rows_set
    
    new_matrix_shape = (len(old_rows_set) + len(new_ids), len(old_columns_set) + len(new_ids))
    
    new_matrix = np.ones(new_matrix_shape) * np.nan
    
    new_matrix[:len(old_rows_set), :len(old_columns_set)] = old_train_inchikey_similarities.values
    
    new_ids = list(new_ids)
    new_matrix[len(old_rows_set):, len(old_columns_set):] = new_train_inchikey_similarities.loc[new_ids, new_ids].values
    
    # Make a new dataframe
    new_matrix = pd.DataFrame(new_matrix, index=old_rows+new_ids, columns=old_rows+new_ids)
    new_matrix.to_csv(output_train_inchikey_similarities_path)

def add_new_train_test_similarities(old_pickle_path,
                                    new_pickle_path,
                                    old_train_test_similarities_path,
                                    new_train_test_similarities_path,
                                    output_train_test_similarities_path):
    
    new_train_test_similarities = pd.read_csv(new_train_test_similarities_path, index_col=0)
    old_train_test_similarities = pd.read_csv(old_train_test_similarities_path, index_col=0)
    
    # Add new similarities, test similarities (columns) should be identical
    # Train similarities should be different, we only have to add the new rows corresponding to the new inchikeys
    
    old_rows    = list(old_train_test_similarities.index)
    old_columns = list(old_train_test_similarities.columns)
    old_rows_set = set(old_rows)
    old_columns_set = set(old_columns)
    
    new_rows = list(new_train_test_similarities.index)
    new_columns = list(new_train_test_similarities.columns)
    new_rows_set = set(new_rows)
    new_columns_set = set(new_columns)
    
    assert old_columns_set == new_columns_set
    
    new_ids = new_rows_set - old_rows_set
    
    new_matrix_shape = (len(old_rows_set) + len(new_ids), len(old_columns_set))
    
    new_matrix = np.ones(new_matrix_shape) * np.nan
    
    new_matrix[:len(old_rows_set), :] = old_train_test_similarities.values
    
    new_ids = list(new_ids)
    
    new_matrix[len(old_rows_set):, :] = new_train_test_similarities.loc[new_ids, :].values
    
    # Make a new dataframe
    new_matrix = pd.DataFrame(new_matrix, index=old_rows+new_ids, columns=old_columns)
    
    # Assert no nans
    assert new_matrix.isna().sum().sum() == 0
    
    new_matrix.to_csv(output_train_test_similarities_path)
    
def main():
    parser = argparse.ArgumentParser(description='Add new data to existing data')
    parser.add_argument('--old_pickle_path', type=str, help='path to old pickle file')
    parser.add_argument('--new_pickle_path', type=str, help='path to new pickle file')
    parser.add_argument('--old_train_test_similarities_path', type=str, help='path to old train test similarities file')
    parser.add_argument('--new_train_test_similarities_path', type=str, help='path to new train test similarities file')
    parser.add_argument('--old_train_inchikey_similarities_path', type=str, help='path to old train inchikey similarities file')
    parser.add_argument('--new_train_inchikey_similarities_path', type=str, help='path to new train inchikey similarities file')
    parser.add_argument('--output_train_test_similarities_path', type=str, help='path to output train test similarities file')
    parser.add_argument('--output_train_inchikey_similarities_path', type=str, help='path to output train inchikey similarities file')
    args = parser.parse_args()
    
    # Make sure all argument paths exist
    if not os.path.exists(args.old_pickle_path):
        raise FileNotFoundError(f"Old pickle file not found: {args.old_pickle_path}")
    if not os.path.exists(args.new_pickle_path):
        raise FileNotFoundError(f"New pickle file not found: {args.new_pickle_path}")
    if not os.path.exists(args.old_train_test_similarities_path):
        raise FileNotFoundError(f"Old train test similarities file not found: {args.old_train_test_similarities_path}")
    if not os.path.exists(args.new_train_test_similarities_path):
        raise FileNotFoundError(f"New train test similarities file not found: {args.new_train_test_similarities_path}")
    if not os.path.exists(args.old_train_inchikey_similarities_path):
        raise FileNotFoundError(f"Old train inchikey similarities file not found: {args.old_train_inchikey_similarities_path}")
    if not os.path.exists(args.new_train_inchikey_similarities_path):
        raise FileNotFoundError(f"New train inchikey similarities file not found: {args.new_train_inchikey_similarities_path}")
    
    add_new_train_inchikey_similarities(args.old_train_inchikey_similarities_path, 
                                        args.new_train_inchikey_similarities_path,
                                        args.output_train_inchikey_similarities_path)

    add_new_train_test_similarities(args.old_pickle_path,
                                    args.new_pickle_path,
                                    args.old_train_test_similarities_path,
                                    args.new_train_test_similarities_path,
                                    args.output_train_test_similarities_path)

if __name__=="__main__":
    main()