import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_csi', 'plot_train_valid_test_csi']

# 目盛りのステップ候補を定数として定義
STEPS = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

CMAP = 'CMRmap'

def _choose_step(data_length, steps):
    """
    データの長さに基づいて適切な目盛りのステップを選択する関数
    """
    for step in steps:
        if data_length / step <= 12:
            return step
    return steps[-1]

def plot_csi(data, title, figsize=(6, 4)):
    """
    単一のCSIデータのヒートマップをプロットする関数
    """
    # データの最小値と最大値を取得
    vmin = data.min()
    vmax = data.max()
    
    # プロットを作成
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, ax=ax, vmin=vmin, vmax=vmax, cmap=CMAP, cbar=True)
    
    ax.set_title(f'{title}')
    
    # 適切な目盛りのステップを選択
    step = _choose_step(data.shape[0], STEPS)
    
    ax.set_yticks(np.arange(0, data.shape[0], step))
    ax.set_yticklabels(np.arange(0, data.shape[0], step))
    
    # プロットの表示
    plt.tight_layout()
    # plt.savefig(os.path.join(plot_dirpath, f'csi_{process_name.lower().replace(" ","_")}.png'), dpi=300)
    plt.show()
    plt.close()
    
def plot_train_valid_test_csi(train, valid, test, process_name):
    """
    train, valid, testそれぞれのCSIのヒートマップをプロットする関数
    """
    # データの最小値と最大値を取得
    vmin = min(train.min(), valid.min(), test.min())
    vmax = max(train.max(), valid.max(), test.max())
    
    # サブプロットを作成
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    
    sns.heatmap(train, ax=ax[0], vmin=vmin, vmax=vmax, cmap=CMAP, cbar=False)
    sns.heatmap(valid, ax=ax[1], vmin=vmin, vmax=vmax, cmap=CMAP, cbar=False)
    sns.heatmap(test, ax=ax[2], vmin=vmin, vmax=vmax, cmap=CMAP, cbar=False)
    
    ax[0].set_title(f'Train ({process_name})')
    ax[1].set_title(f'Valid ({process_name})')
    ax[2].set_title(f'Test ({process_name})')
    
    # 適切な目盛りのステップを選択
    train_step = _choose_step(train.shape[0], STEPS)
    valid_step = _choose_step(valid.shape[0], STEPS)
    test_step = _choose_step(test.shape[0], STEPS)
    
    ax[0].set_yticks(np.arange(0, train.shape[0], train_step))
    ax[1].set_yticks(np.arange(0, valid.shape[0], valid_step))
    ax[2].set_yticks(np.arange(0, test.shape[0], test_step))
    
    ax[0].set_yticklabels(np.arange(0, train.shape[0], train_step))
    ax[1].set_yticklabels(np.arange(0, valid.shape[0], valid_step))
    ax[2].set_yticklabels(np.arange(0, test.shape[0], test_step))
    
    # カラーバーのプロット
    cbar = fig.colorbar(ax[2].collections[0], cax=ax[3])
    
    # プロットの表示
    plt.tight_layout()
    # plt.savefig(os.path.join(plot_dirpath, f'amp_{process_name.lower().replace(" ","_")}.png'), dpi=300)
    plt.show()
    plt.close()