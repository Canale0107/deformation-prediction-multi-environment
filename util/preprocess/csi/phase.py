import numpy as np

__all__ = ['remove_timing_offset']

def remove_timing_offset(csi_phase_unwrapped):
    '''
    位相を変形する関数。理論は以下の論文に基づく
    https://ieeexplore.ieee.org/abstract/document/7218588
    '''

    F = csi_phase_unwrapped.shape[1] # サブキャリア数
    f = range(F)
    a = (csi_phase_unwrapped[:, -1] - csi_phase_unwrapped[:, 0]) / F
    b = (1/F) * np.sum(csi_phase_unwrapped, axis=1)
    csi_phase_transformed = np.empty_like(csi_phase_unwrapped)

    for t in range(csi_phase_unwrapped.shape[0]):
        csi_phase_transformed[t, :] = csi_phase_unwrapped[t, :] - a[t] * f - b[t]

    return csi_phase_transformed