import os
import pandas as pd
from settings import data_folder


def make_data_frame(csv_folder: str, train_df_name: str, validate_df_name: str, test_df_name: str):
    train_df_path = os.path.join(csv_folder, train_df_name)
    validate_df_path = os.path.join(csv_folder, validate_df_name)
    test_df_path = os.path.join(csv_folder, test_df_name)

    train_df = pd.read_csv(train_df_path)
    validate_df = pd.read_csv(validate_df_path)
    test_df = pd.read_csv(test_df_path)

    def abs_path(df):
        df['image_a'] = df.apply(lambda x: os.path.join(data_folder, x['image_a']), axis=1)
        df['image_b'] = df.apply(lambda x: os.path.join(data_folder, x['image_b']), axis=1)
        return df

    train_df = abs_path(train_df)
    validate_df = abs_path(validate_df)
    test_df = abs_path(test_df)

    dataset = {
        'dataset': [train_df_path, validate_df_path, test_df_path],
        'features': ['img1_path', 'img2_path', 'mask_path']
    }

    data = {
        'train': train_df.copy(),
        'validate': validate_df.copy(),
        'test': test_df.copy()
    }

    return dataset, data

