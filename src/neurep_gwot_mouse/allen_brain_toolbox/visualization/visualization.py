# Standard Library
from typing import List, Optional, Tuple, Union

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns


def plot_heatmap(
    corrs: np.ndarray,
    annot: bool = False,
    *,
    figsize: Tuple[int, int] = (15, 15),
    cmap: str = "rocket_r",
    cbar_label: str = "Spearman R",
    annot_size: int = 8,
    fmt: str = ".2f",
    shrink: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_size: int = 15,
    ylabel_size: int = 15,
    xticklabels: Union[List[str], bool] = True,
    yticklabels: Union[List[str], bool] = True,
    xticklabels_size: int = 10,
    yticklabels_size: int = 10,
    title: str = "RSA",
    title_size: int = 20,
    show_figure: bool = True,
) -> matplotlib.axes.Axes:
    """RDM間の相関をplotする

    Args:
        corrs (np.ndarray): RDM間の相関係数
        annot (bool, optional): アノテーションを表示するかどうか. Defaults to True.

    Keyword Args:
        figsize (Tuple[int, int], optional): figure size. Defaults to (15, 15).
        cmap (str, optional): colormap. Defaults to "rocket_r".
        cbar_label (str, optional): colorbar label. Defaults to "Spearman R".
        title (str, optional): title. Defaults to "RSA".
        title_size (int, optional): title fontsize. Defaults to 20.

    Returns:
        matplotlib.axes.Axes: heatmap
    """

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corrs,
        square=True,
        annot=annot,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": shrink},
        fmt=fmt,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    # label setting
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)

    # tick setting
    if yticklabels is not False:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # xticklabels と yticklabels のサイズを変更
    for label in ax.get_xticklabels():
        label.set_fontsize(xticklabels_size)  # ここでフォントサイズを設定

    for label in ax.get_yticklabels():
        label.set_fontsize(yticklabels_size)  # ここでフォントサイズを設定

    # title setting
    ax.set_title(title, fontsize=title_size)

    # colorabr setting
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, size=title_size)

    if show_figure:
        plt.show()

    return ax


def plot_distribution(
    corrs: np.ndarray,
    *,
    figsize: Tuple[int, int] = (8, 5),
    font_size: int = 20,
    bins: int = 100,
    color: str = "C0",
    title: str = "RDMs distribution",
    title_size: int = 20,
    xlabel_size: int = 15,
    ylabel_size: int = 15,
    show_figure: bool = True,
):
    """RDM間の相関のhistogramをplotする

    Args:
        corrs (np.ndarray): RDM間の相関係数

    Keyword Args:
        figsize (Tuple[int, int], optional): figure size. Defaults to (15, 15).
        cmap (str, optional): colormap. Defaults to "rocket_r".
        cbar_label (str, optional): colorbar label. Defaults to "Spearman R".
        title (str, optional): title. Defaults to "RSA".
        title_size (int, optional): title fontsize. Defaults to 20.

    Returns:
        matplotlib.axes.Axes: heatmap
    """

    corrs_vec = sp.spatial.distance.squareform(corrs, checks=False)

    # default setting
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(style="darkgrid")
    plt.rcParams["font.size"] = font_size

    _, ax = plt.subplots(figsize=figsize)
    sns.histplot(corrs_vec, bins=bins, color=color)
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("RDM value", fontsize=xlabel_size)
    ax.set_ylabel("Count", fontsize=ylabel_size)

    if show_figure:
        plt.show()

    return ax


def plot_all_gw(
    all_gw: List[np.ndarray],
    areas: List[str],
    *,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "rocket_r",
    shrink: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_size: int = 15,
    ylabel_size: int = 15,
    title: str = "GWOT",
    title_size: int = 20,
    linewidth: float = 0.8,
    show_figure: bool = True,
) -> matplotlib.axes.Axes:
    """すべてのGWOTをplotする.

    Args:
        all_gw (List[np.ndarray]):
            すべてのGWOTを格納したリスト.
        areas (List[str]):
            すべての領域名を格納したリスト.

    Keyword Args:
        figsize (Tuple[int, int], optional): figure size. Defaults to (8, 8).
        cmap (str, optional): colormap. Defaults to "rocket_r".
        shrink (float, optional): shrink of colorbar. Defaults to 0.8.
        xlabel (Optional[str], optional): xlabel. Defaults to None.
        ylabel (Optional[str], optional): ylabel. Defaults to None.
        xlabel_size (int, optional): xlabel fontsize. Defaults to 15.
        ylabel_size (int, optional): ylabel fontsize. Defaults to 15.
        title (str, optional): title. Defaults to "GWOT".
        title_size (int, optional): title fontsize. Defaults to 20.
        linewidth (float, optional): linewidth of grid. Defaults to 0.8.
        show_figure (bool, optional): show figure or not. Defaults to True.

    Returns:
        matplotlib.axes.Axes: heatmaps
    """

    # reshape GWOT
    len_matrix = len(areas)
    len_gw = all_gw[0].shape[0]
    gw = np.zeros((len_matrix * len_gw, len_matrix * len_gw))

    for i in range(len_matrix):
        for j in range(len_matrix):
            gw[i * len_gw : (i + 1) * len_gw, j * len_gw : (j + 1) * len_gw] = all_gw[i * len_matrix + j]

    # plot heatmap
    plt.rcParams["font.size"] = 12
    ticks = np.arange(len_gw // 2, len_gw * len_matrix, len_gw)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(gw, aspect="equal", cmap=cmap)
    fig.colorbar(im, shrink=shrink, ax=ax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(areas)
    ax.set_yticks(ticks)
    ax.set_yticklabels(areas)
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=ylabel_size, labelpad=10)
    ax.xaxis.set_label_position("top")

    for i in range(len_matrix):
        ax.axvline(len_gw * i - 0.5, color="gray", linewidth=linewidth)
        ax.axhline(len_gw * i - 0.5, color="gray", linewidth=linewidth)

    ax.set_title(title, fontsize=title_size, pad=10)

    if show_figure:
        plt.show()

    return ax
