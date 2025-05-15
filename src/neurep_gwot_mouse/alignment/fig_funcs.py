#%%
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.cluster.hierarchy import dendrogram

from typing import List


# %%
def swarm_same_areas(
    df: pd.DataFrame,
    metric_name: str,
    fig_dir: Path,
):
    n = df.shape[1]
    areas = df.columns.tolist()
    palette = sns.color_palette('bright', n_colors=n)

    plt.figure(figsize=(8, 8))
    plt.style.use('seaborn-darkgrid')
    sns.swarmplot(data=df, palette=palette, size=n)

    plt.xticks(ticks=range(len(areas)), labels=areas, size=25, rotation=90)
    plt.xlabel('Brain area', size=35, labelpad=10)

    ylabel = "rsa corr" if metric_name == "spearman_r" else "top1 acc"
    plt.ylabel(ylabel, size=30)
    plt.yticks(size=25)

    if metric_name in ['spearman_r']:
        plt.ylim(-0.05, 1.05)
    elif metric_name in ['ot_top1_matching_rate']:
        plt.ylim(-5, 105)

    # show chance level
    if metric_name == 'ot_top1_matching_rate':
        chance_level = 100/118
        plt.axhline(y=chance_level, color='black', linestyle='--')

    plt.tight_layout()
    plt.savefig(fig_dir / f'swarm_{metric_name}.svg')
    pass


def across_area_heatmap(
    v: np.ndarray,
    labels: List[str],
    title: str = "Spearman R",
    fmt: str = ".2f",
    vmax: float = 100,
    figsize=(10, 10),
    fig_dir: Path = None,
    fig_name: str = "heatmap"
):
    # plot heatmap
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        v,
        square=True,
        annot=True,
        annot_kws={"size": 18},
        cbar_kws={"shrink": .8},
        fmt=fmt,
        cmap="rocket_r",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=vmax
    )

    # label setting
    ax.set_xlabel("Target", fontsize=20, labelpad=10)
    ax.set_ylabel("Source", fontsize=20, labelpad=10)

    # tick setting
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for label in ax.get_xticklabels():
        label.set_fontsize(15)  # ここでフォントサイズを設定

    for label in ax.get_yticklabels():
        label.set_fontsize(15)  # ここでフォントサイズを設定

    ax.set_title(title, fontsize=20, pad=10)

    # colorabr setting
    cbar = ax.collections[0].colorbar
    cbar.set_label(title, size=20, labelpad=20)
    cbar.ax.tick_params(labelsize=20)

    # save fig
    if fig_dir:
        ax.figure.savefig(fig_dir / f"{fig_name}.svg", bbox_inches="tight")
    pass


def plot_clustering(
    Z: np.ndarray,
    labels: List[str],
    title: str = "Dendrogram",
    figsize=(8, 3),
    fontsize=16,
    fig_dir: Path = None,
    fig_name: str = "dendrogram"
):
    # plot
    plt.style.use("seaborn-darkgrid")
    _, ax = plt.subplots(figsize=figsize)
    dendrogram(Z, ax=ax, labels=labels, leaf_font_size=fontsize)
    ax.set_title(title, fontsize=fontsize + 5, pad=10)
    ax.set_xlabel('Area', fontsize=fontsize, labelpad=10)
    ax.set_ylabel('Distance', fontsize=fontsize, labelpad=10)
    ax.tick_params(axis='y', labelsize=14)
    # save fig
    if fig_dir:
        ax.figure.savefig(fig_dir / f"{fig_name}.svg", bbox_inches="tight")
    pass
