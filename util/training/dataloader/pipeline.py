import numpy as np
from sklearn.decomposition import PCA

def average_over_T_dimension(X):
    return np.mean(X, axis=1)

def apply_pca(X_train, X_valid, X_test, n_components):
    """
    Apply PCA to X_train, X_valid, and X_test along the (F, 2) feature dimensions.

    Parameters:
    X_train, X_valid, X_test (numpy.ndarray): Input datasets of shape (N, F, 2).
    n_components (int): Number of principal components to keep.

    Returns:
    tuple: X_train_pca, X_valid_pca, X_test_pca, transformed into shape (N, n_components).
    """
    # Reshape each dataset to 2D: (N, F*2)
    N_train, F, _ = X_train.shape
    X_train_reshaped = X_train.reshape(N_train, F * 2)
    N_valid, _, _ = X_valid.shape
    X_valid_reshaped = X_valid.reshape(N_valid, F * 2)
    N_test, _, _ = X_test.shape
    X_test_reshaped = X_test.reshape(N_test, F * 2)
    
    # Fit PCA on X_train and apply to all datasets
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_valid_pca = pca.transform(X_valid_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)
    
    return X_train_pca, X_valid_pca, X_test_pca


def stack_frames(data, window_size):
    """
    フレームをウィンドウサイズ分スタックし、新たな第3次元として追加する関数。
    
    Parameters:
    data (numpy.ndarray): 入力データ。形状は (フレーム, サブキャリア, 2)。
    window_size (int): フレームをスタックするウィンドウサイズ。
    
    Returns:
    numpy.ndarray: スタック後のデータ。形状は (フレーム数 - window_size + 1, サブキャリア, window_size, 2)。
    """
    # フレームをウィンドウサイズ分スライディングしながらスタック
    stacked_data = np.array([data[i:i+window_size] for i in range(data.shape[0] - window_size + 1)])
    
    # 形状を変換
    stacked_data = stacked_data.reshape(-1, data.shape[1], window_size)
    
    return stacked_data


def preprocess_pipeline(dataset, n_components, window_size):

    train_data, valid_data, test_data = dataset

    X_train, y_train = train_data
    X_valid, y_valid = valid_data
    X_test, y_test = test_data
    
    X_train_avg = average_over_T_dimension(X_train)
    X_valid_avg = average_over_T_dimension(X_valid)
    X_test_avg = average_over_T_dimension(X_test)

    X_train_pca, X_valid_pca, X_test_pca = apply_pca(X_train_avg, X_valid_avg, X_test_avg, n_components)
    variance_ratio = calculate_explained_variance(X_train_avg, n_components)
    print(f"{n_components} 個の主成分で維持される分散: {variance_ratio:.2f}%")

    X_train = stack_frames(X_train_pca, window_size)
    X_valid = stack_frames(X_valid_pca, window_size)
    X_test = stack_frames(X_test_pca, window_size)

    y_train = y_train[window_size-1:]
    y_valid = y_valid[window_size-1:]
    y_test = y_test[window_size-1:]

    train_data = X_train, y_train
    valid_data = X_valid, y_valid
    test_data = X_test, y_test

    return train_data, valid_data, test_data


def calculate_explained_variance(X, n_components):
    """
    Calculate the cumulative explained variance for a specified number of components.

    Parameters:
    X (numpy.ndarray): Input dataset of shape (N, F, 2).
    n_components (int): Number of principal components to retain.

    Returns:
    float: Cumulative explained variance ratio as a percentage.
    """
    # Reshape to 2D: (N, F*2)
    N, F, _ = X.shape
    X_reshaped = X.reshape(N, F * 2)
    
    # Fit PCA with the specified number of components
    pca = PCA(n_components=n_components)
    pca.fit(X_reshaped)
    
    # Calculate cumulative explained variance ratio
    cumulative_variance_ratio = np.sum(pca.explained_variance_ratio_) * 100  # Convert to percentage
    
    return cumulative_variance_ratio


