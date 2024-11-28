from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from . import plot_style


__all__ = ['plot_ecdf']


def plot_ecdf(metric, baseline_metrics, labels, x_label, y_label, filename=None, xlim=None, legend_loc='lower right'):
    """
    Generalized function to plot ECDF for a given metric and multiple baselines.
    
    Parameters:
    - metric: numpy array or list of the metric to plot (e.g., mse, iou)
    - baseline_metrics: list of numpy arrays or lists for the baseline metrics
    - labels: list of strings corresponding to the labels for the baselines and the metric
    - x_label: label for the x-axis
    - y_label: label for the y-axis
    - filename: file name to save the plot
    - xlim: tuple of (min, max) for x-axis limits, default is None
    """
    # Calculate ECDF for each metric
    ecdf_metric = ECDF(metric)
    ecdf_baselines = [ECDF(baseline) for baseline in baseline_metrics]

    # Plot ECDFs
    fig, ax = plt.subplots(figsize=(6, 5))
    lines = []
    
    line, = ax.step(ecdf_metric.x, ecdf_metric.y, where='post', label=labels[0], linewidth=3)
    lines.append(line)
    
    for ecdf, label in zip(ecdf_baselines, labels[1:]):
        line, = ax.step(ecdf.x, ecdf.y, where='pre', label=label, linewidth=3)
        lines.append(line)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if xlim:
        ax.set_xlim(xlim)
    
    # Set up the legend
    leg = ax.legend(
        loc=legend_loc,
        fontsize=14,
        ncol=1,
        fancybox=False,  # True gives rounded corners
        edgecolor='black',  # Color of the border
        handler_map={line: HandlerLine2D(numpoints=1) for line in lines}
    )
    leg.get_frame().set_alpha(1.0)  # Make legend box opaque

    ax.grid()
    ax.tick_params(direction='in')  # Make ticks point inward

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.show()