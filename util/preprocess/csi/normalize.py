import numpy as np

__all__ = [
    'normalize_csi_amplitudes_by_frame_power', 
    'normalize_csi_amplitudes_by_frame_mean',
    'normalize_csi_data_by_frame_power'
]

def normalize_csi_amplitudes_by_frame_power(amplitudes):
    num_frames, num_subcarriers = amplitudes.shape

    # 正規化されたデータを格納するための空の配列を作成
    normalized_amplitudes = np.zeros_like(amplitudes)

    # 各フレームについて処理
    for i in range(num_frames):
        # 各サブキャリアの振幅と位相を取得
        amplitude = amplitudes[i, :]

        # フレームの平均電力を計算
        mean_power = np.mean(np.power(amplitude, 2))

        # 振幅をフレームの平均電力で割って正規化
        normalized_amplitudes[i, :] = amplitude / mean_power

    return normalized_amplitudes

def normalize_csi_amplitudes_by_frame_mean(amplitudes):
    amp_means = amplitudes.mean(axis=1, keepdims=True)
    amp_normalized = amplitudes / amp_means

    return amp_normalized

def normalize_csi_data_by_frame_power(data):
    """
    フレームの平均電力でCSIデータを正規化する関数。
    
    Parameters:
        data (numpy.ndarray): CSIデータを含む3次元配列。
                              形状は (フレーム数, サブキャリア数, 2) で、
                              最後の次元が実部と虚部を表す。
                              
    Returns:
        normalized_data (numpy.ndarray): 正規化されたデータを含む同形状の3次元配列。
    """
    num_frames, num_subcarriers, _ = data.shape

    # 正規化されたデータを格納するための空の配列を作成
    normalized_data = np.zeros_like(data)

    # 各フレームについて処理
    for i in range(num_frames):
        # 各サブキャリアの振幅と位相を取得
        real = data[i, :, 0]
        imag = data[i, :, 1]

        # フレームの平均電力を計算
        mean_power = np.mean(real**2 + imag**2)

        # 振幅をフレームの平均電力で割って正規化
        normalized_data[i, :, 0] = real / mean_power
        normalized_data[i, :, 1] = imag / mean_power

    return normalized_data

def normalize_csi_data_by_frame_mean(data):
    """
    フレームの平均振幅でCSIデータを正規化する関数。
    
    Parameters:
        data (numpy.ndarray): CSIデータを含む3次元配列。
                              形状は (フレーム数, サブキャリア数, 2) で、
                              最後の次元が実部と虚部を表す。
                              
    Returns:
        normalized_data (numpy.ndarray): 正規化されたデータを含む同形状の3次元配列。
    """
    num_frames, num_subcarriers, _ = data.shape

    # 正規化されたデータを格納するための空の配列を作成
    normalized_data = np.zeros_like(data)

    # 各フレームについて処理
    for i in range(num_frames):
        # 各サブキャリアの振幅と位相を取得
        real = data[i, :, 0]
        imag = data[i, :, 1]

        # フレームの平均電力を計算
        mean_amp = np.sqrt(np.mean(real**2 + imag**2))

        # 振幅をフレームの平均電力で割って正規化
        normalized_data[i, :, 0] = real / mean_amp
        normalized_data[i, :, 1] = imag / mean_amp

    return normalized_data