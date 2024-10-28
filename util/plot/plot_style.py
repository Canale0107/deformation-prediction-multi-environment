import matplotlib.pyplot as plt
import matplotlib as mpl

# 全体のスタイルを設定
plt.style.use('default')

# 色とフォントの設定
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])

# フォント設定
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titleweight'] = 'semibold'