from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


__all__ = [
    'split_data_for_single_location',
    'split_data_for_multiple_location',
    'concat_and_shuffle'
]


def load_data(csi_preprocess_id, location):

    project_dirpath = Path('/tf/workspace/deformation-prediction-multi-environment')
    data_dirpath = project_dirpath/'data'/'preprocessed'
    
    csi_data_dirpath = data_dirpath/'csi'/csi_preprocess_id
    shape_data_dirpath = data_dirpath/'shape'/'binarized'
    
    csi_filepath = csi_data_dirpath/location/'csi.npy'
    shape_filepath = shape_data_dirpath/f'shape_{location}.npy'
    
    csi = np.load(csi_filepath)
    shape = np.load(shape_filepath)

    return csi, shape


def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    
    # 比率が1になることを確認
    assert train_size + val_size + test_size == 1, "train_size, val_size, test_sizeの合計は1である必要があります。"
    
    # 訓練データと残りデータに分割、時系列を保ちシャッフルせずに分割する
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, shuffle=False)
    
    # 残りデータを検証データと評価データに分割
    val_ratio = val_size / (val_size + test_size)  # 残りデータに対する検証データの割合
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio, shuffle=False)

    train_data = (X_train, y_train)
    valid_data = (X_val, y_val)
    test_data = (X_test, y_test)

    return train_data, valid_data, test_data


def split_data_for_single_location(csi_preprocess_id, location, shuffle=False):
    csi, shape = load_data(csi_preprocess_id, location)
    # 分割は時系列で行う
    train_data, valid_data, test_data = split_data(csi, shape)

    if shuffle:
        train_data = shuffle_data(*train_data)
        print('train data shuffled.')

    return train_data, valid_data, test_data


def shuffle_data(csi, shape):
    indices = np.arange(csi.shape[0])
    return csi[indices], shape[indices]


def concat_and_shuffle(dataset_dict, location_list):
    # データを結合し、シャッフルする
    X_list = []
    y_list = []
    for location in location_list:
        X, y = dataset_dict[location]
        X_list.append(X)
        y_list.append(y)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return shuffle_data(X, y)


def split_data_for_multiple_location(csi_preprocess_id, location_list):
    train_data_dict, valid_data_dict, test_data_dict = {}, {}, {}

    for location in location_list:
        csi, shape = load_data(csi_preprocess_id, location)
    
        loc_train_data, loc_valid_data, loc_test_data = split_data(csi, shape)
    
        train_data_dict[location] = loc_train_data
        valid_data_dict[location] = loc_valid_data
        test_data_dict[location] = loc_test_data

    dataset_dict = train_data_dict, valid_data_dict, test_data_dict

    return dataset_dict

