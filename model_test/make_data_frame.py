import os
import pickle

import pandas as pd
from settings import data_folder, data_path


def make_data_frame(csv_folder: str, train_df_name: str, validate_df_name: str, test_df_name: str):
    train_df_path = os.path.join(csv_folder, train_df_name)
    validate_df_path = os.path.join(csv_folder, validate_df_name)
    test_df_path = os.path.join(csv_folder, test_df_name)

    train_df = pd.read_csv(train_df_path)
    validate_df = pd.read_csv(validate_df_path)
    test_df = pd.read_csv(test_df_path)

    with open(os.path.join(data_path, 'id_bbox_array.pickle'), 'rb') as handle:
        d_id_bbox_array = pickle.load(handle)

    def meta_change(df):
        df['image_a'] = df.apply(lambda x: os.path.join(data_folder, x['image_a']), axis=1)
        df['image_b'] = df.apply(lambda x: os.path.join(data_folder, x['image_b']), axis=1)

        df['bbox_voc_a'] = df['image_a_id'].map(d_id_bbox_array)
        df['bbox_voc_b'] = df['image_b_id'].map(d_id_bbox_array)

        df = df[df['bbox_voc_a'].notna()].reset_index(drop=True)
        df = df[df['bbox_voc_b'].notna()].reset_index(drop=True)
        return df

    train_df = meta_change(train_df)
    validate_df = meta_change(validate_df)
    test_df = meta_change(test_df)

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

