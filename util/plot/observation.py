import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_random_images']

def plot_random_images(images, N, M, cmap='gray', random_seed=None):
    """
    画像データの中からランダムでN行M列に画像をプロットする関数

    Args:
    images (numpy.ndarray): 形状が (4500, 28, 28) の画像データ
    N (int): プロットする画像の行数
    M (int): プロットする画像の列数
    cmap (str): カラーマップ（デフォルトは 'gray'）
    random_seed (int or None): 乱数シード（デフォルトは None）
    """
    # 乱数シードを設定
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # プロットする画像の数を計算
    num_images = N * M
    
    # ランダムに画像を選択
    random_indices = np.random.choice(images.shape[0], num_images, replace=False)
    selected_images = images[random_indices]
    
    # プロットの設定
    fig, axes = plt.subplots(N, M, figsize=(M * 2, N * 2))
    
    # 画像をプロット
    for i in range(N):
        for j in range(M):
            ax = axes[i, j]
            ax.imshow(selected_images[i * M + j], cmap=cmap)
            ax.axis('off')
    
    plt.show()