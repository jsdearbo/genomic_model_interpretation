"""
Attribution logo plots with gene map overlays.

Combines sequence-level attribution logos (from tangermeme) with gene
structure diagrams and optional read-density tracks for comprehensive
motif visualization.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional tangermeme dependency
try:
    from tangermeme.plot import plot_logo

    _HAS_TANGERMEME = True
except ImportError:
    _HAS_TANGERMEME = False


def plot_attribution_logo(
    attribution: np.ndarray,
    start: int = 0,
    end: int = None,
    ax=None,
    title: str = "",
    annotations: pd.DataFrame = None,
    figsize: tuple = (12, 4),
    save_path: str = None,
):
    """
    Plot an attribution logo (sequence logo weighted by attribution scores).

    Parameters
    ----------
    attribution : np.ndarray
        Attribution array, shape (4, L).
    start, end : int
        Position range to display.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Creates a new figure if None.
    title : str
        Plot title.
    annotations : pd.DataFrame, optional
        Seqlet annotations to overlay (passed to tangermeme).
    figsize : tuple
        Figure size if creating a new figure.
    save_path : str, optional
        If set, save figure to this path.
    """
    if not _HAS_TANGERMEME:
        raise ImportError("tangermeme is required for logo plots.")

    attribution = np.asarray(attribution)
    if attribution.ndim != 2:
        raise ValueError(f"Expected 2D attribution, got shape {attribution.shape}")
    if attribution.shape[1] == 4:
        attribution = attribution.T
    if attribution.shape[0] != 4:
        raise ValueError(f"Attribution must have shape (4, L), got {attribution.shape}")

    if end is None:
        end = attribution.shape[1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot_logo(attribution, ax=ax, start=start, end=end, annotations=annotations)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution Score")

    if save_path is not None and own_fig:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Logo plot saved → %s", save_path)


def plot_logo_with_gene_map(
    attribution: np.ndarray,
    elements_df: pd.DataFrame,
    seqlet_row: pd.Series,
    seq_annotations: pd.DataFrame = None,
    tensor_center: int = 262_144,
    plot_window: int = 104,
    highlight_bounds: tuple = None,
    save_path: str = None,
    figsize: tuple = (20, 6),
):
    """
    Plot a combined attribution logo (top) and gene map (bottom).

    Places the attribution logo centered on a seqlet hit and aligns
    a gene structure diagram underneath with the motif region highlighted.

    Parameters
    ----------
    attribution : np.ndarray
        Attribution array, shape (4, L) or (L, 4).
    elements_df : pd.DataFrame
        Gene elements for the gene map.
    seqlet_row : pd.Series
        Row describing the seqlet hit. Must contain 'start', 'strand',
        'intron_name', and optionally 'pattern_label' / 'motif_name'.
    seq_annotations : pd.DataFrame, optional
        Additional annotations to overlay on the logo.
    tensor_center : int
        Center of the input tensor in sequence coordinates.
    plot_window : int
        Width of the logo window (centered on seqlet start).
    highlight_bounds : tuple, optional
        Explicit (start, end) genomic coords to highlight on the gene map.
        If None, computed from seqlet position and tensor center.
    save_path : str, optional
        If set, saves the figure.
    figsize : tuple
        Figure size.
    """
    if not _HAS_TANGERMEME:
        raise ImportError("tangermeme is required for logo plots.")

    from visualization.gene_map import plot_gene_map

    # Normalize attribution to (4, L)
    arr = np.asarray(attribution)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D attribution, got {arr.shape}")
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    seq_len = arr.shape[1]

    plt_start = max(0, int(seqlet_row.start) - plot_window // 2)
    plt_end = min(seq_len, int(seqlet_row.start) + plot_window // 2)

    motif_label = getattr(
        seqlet_row, "pattern_label",
        getattr(seqlet_row, "motif_name", "unknown"),
    )
    seq_name = seqlet_row.intron_name

    fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

    # Top: logo
    plot_logo(arr, ax=axs[0], start=plt_start, end=plt_end, annotations=seq_annotations)
    axs[0].set_title(f"Motif: {motif_label} | Seq: {seq_name}")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Attribution Score")

    # Bottom: gene map with highlight
    if highlight_bounds is None and not elements_df.empty:
        # Compute highlight from seqlet position
        tscript_start = elements_df["start"].min()
        tscript_end = elements_df["end"].max()
        tscript_mid = (tscript_start + tscript_end) // 2
        strand = seqlet_row.strand
        if strand == "+":
            hl_start = tscript_mid + (int(seqlet_row.start) - tensor_center)
            hl_end = tscript_mid + (int(seqlet_row.end) - tensor_center)
        else:
            hl_start = tscript_mid - (int(seqlet_row.end) - tensor_center)
            hl_end = tscript_mid - (int(seqlet_row.start) - tensor_center)
        highlight_bounds = (hl_start, hl_end)

    gene_name = (
        elements_df["gene_name"].iloc[0]
        if "gene_name" in elements_df.columns and not elements_df.empty
        else ""
    )
    plot_gene_map(
        elements_df=elements_df,
        ax=axs[1],
        title=gene_name,
        highlight_area=highlight_bounds,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Logo + gene map saved → %s", save_path)
    return fig
