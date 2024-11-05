import numpy as np
import matplotlib.pyplot as plt

from . import plot_style


__all__ = ['plot_predictions']


def plot_predictions(model, X_data, Y_data, n_samples=6, threshold=0.5, binarize=True, title_prefix='Sample', seed=None, metrics={}):
    """
    予測結果と実際の値を比較するプロットを作成します。
    
    Parameters:
        model: 学習済みモデル
        X_data: 入力データ
        Y_data: 正解データ
        n_samples: プロットするサンプルの数
        threshold: IoUの計算に使用するしきい値
        title_prefix: プロットのタイトルに付けるプレフィックス
        seed: 乱数のシード値
        metrics: 名前をキーとした評価指標のインスタンスの辞書
    """
    # 乱数シードを設定
    if seed is not None:
        np.random.seed(seed)
    
    # 予測を実行
    Y_pred = model.predict(X_data)

    if binarize:
        Y_pred = (Y_pred >= threshold).astype(np.float32)
    
    # サンプルインデックスを選ぶ
    sample_indices = sorted(np.random.choice(range(len(Y_data)), n_samples, replace=False))
    
    # 選ばれたサンプルの画像を用意
    images_predicted = [Y_pred[i] for i in sample_indices]
    images_ground_truth = [Y_data[i] for i in sample_indices]

    # メトリクスを計算
    metrics_results = {name: [] for name in metrics}
    for idx in sample_indices:
        for name, metric in metrics.items():
            metric.reset_state()  # 状態をリセット
            metric.update_state(Y_data[idx:idx+1], Y_pred[idx:idx+1])
            metric_value = metric.result().numpy().item()
            metrics_results[name].append(metric_value)

    # プロットの作成
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4), gridspec_kw={'wspace': 0, 'hspace': 0.05})

    # 各サブプロットに画像を表示
    for i, (ax_pred, ax_gt, img_pred, img_gt) in enumerate(zip(axes[0], axes[1], images_predicted, images_ground_truth)):
        ax_pred.imshow(img_pred, cmap='gray')
        ax_pred.axis('off')  # 軸を非表示にする
        ax_gt.imshow(img_gt, cmap='gray')
        ax_gt.axis('off')  # 軸を非表示にする
        
        # 列にサンプル番号を表示
        sample_index = sample_indices[i]
        ax_pred.set_title(f'{title_prefix} {sample_index}', fontsize=18, pad=5)
        
        # 評価指標の値を表示
        metrics_text = '\n'.join([f'{name}: {metrics_results[name][i]:.4f}' for name in metrics])
        ax_gt.annotate(metrics_text, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top')

    # 'Predicted' と 'Ground Truth' を行の左端に表示
    fig.text(-0.02, 0.7, 'Predicted', va='center', ha='center', fontsize=18)
    fig.text(-0.02, 0.3, 'Ground Truth', va='center', ha='center', fontsize=18)

    # 余白を調整して画像をくっつける
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0, hspace=0)

    plt.show()