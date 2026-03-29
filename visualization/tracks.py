"""
Model prediction and ISM track visualization.

Plot model output tracks with gene annotations, ISM heatmaps showing
the effect of point mutations, and read-density signal from BigWig files.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional grelu dependency
try:
    import grelu.visualize

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def plot_predictions(
    preds: np.ndarray,
    tasks: pd.DataFrame,
    output_start: int,
    output_end: int,
    task_names: list,
    elements_to_highlight: pd.DataFrame = None,
    transcript_exons: pd.DataFrame = None,
    gene_of_interest: str = "",
    filename: str = None,
    figsize: tuple = (20, 6),
):
    """
    Plot model prediction tracks with optional gene annotations.

    Parameters
    ----------
    preds : np.ndarray
        Prediction array from the model.
    tasks : pd.DataFrame
        Task metadata.
    output_start, output_end : int
        Genomic bounds of the output window.
    task_names : list of str
        Which tasks to display.
    elements_to_highlight : pd.DataFrame, optional
        Genomic intervals to shade on the plot.
    transcript_exons : pd.DataFrame, optional
        Exon coordinates for gene annotations.
    gene_of_interest : str
        Gene name for annotation label.
    filename : str, optional
        If set, save figure to this path.
    figsize : tuple
        Figure size.
    """
    if not _HAS_GRELU:
        raise ImportError("grelu is required for track plots.")

    annotations = {}
    if transcript_exons is not None:
        annotations[f"{gene_of_interest} exons"] = transcript_exons

    fig = grelu.visualize.plot_tracks(
        preds,
        start_pos=output_start,
        end_pos=output_end,
        titles=task_names,
        figsize=figsize,
        highlight_intervals=elements_to_highlight,
        facecolor="blue",
        annotations=annotations,
    )

    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Track plot saved → %s", filename)
    return fig


def plot_ism_heatmap(
    ism_results: dict,
    task_index: int,
    tasks: pd.DataFrame,
    chrom: str = "",
    coord_start: int = 0,
    coord_end: int = 0,
    gene_label: str = "",
    element_label: str = "",
    save_path: str = None,
    figsize: tuple = (20, 1.5),
):
    """
    Plot an ISM result as a heatmap.

    Parameters
    ----------
    ism_results : dict
        Mapping of task index → ISM DataFrame.
    task_index : int
        Which task to visualize.
    tasks : pd.DataFrame
        Task metadata for labeling.
    chrom : str
        Chromosome name for axis label.
    coord_start, coord_end : int
        Genomic coordinates of the mutation window.
    gene_label, element_label : str
        Labels for title and axis.
    save_path : str, optional
        If set, save figure to this path.
    figsize : tuple
        Figure size.
    """
    if not _HAS_GRELU:
        raise ImportError("grelu is required for ISM heatmaps.")

    ism_data = ism_results[task_index]

    grelu.visualize.plot_ISM(
        ism_data,
        start_pos=0,
        end_pos=ism_data.shape[1] - 1,
        method="heatmap",
        figsize=figsize,
        center=0,
    )
    plt.title(
        f"Track: {tasks['sample'].loc[task_index]}",
        fontsize=14,
    )
    plt.xlabel(
        f"Log2FC {gene_label} {element_label} expression: "
        f"mutations from {chrom} {coord_start:,} to {coord_end:,}"
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("ISM heatmap saved → %s", save_path)


def plot_read_density_tracks(
    elements_df: pd.DataFrame,
    bw_files: dict,
    gene: str,
    transcript_name: str,
    timepoints: list = None,
    highlight_area: tuple = None,
    save_path: str = None,
    orientation: str = "RNA",
    figsize: tuple = (20, 2.5),
    y_scale: str = "linear",
):
    """
    Plot gene map with RNA-seq signal density tracks from BigWig files.

    Creates a multi-panel figure with a gene structure diagram on top
    and one signal track per timepoint/sample below.

    Parameters
    ----------
    elements_df : pd.DataFrame
        Transcript elements table.
    bw_files : dict
        Mapping of sample name → BigWig file path.
    gene : str
        Gene name for the title.
    transcript_name : str
        Which transcript to display.
    timepoints : list, optional
        Subset of bw_files keys to plot. Defaults to all, sorted.
    highlight_area : tuple of (int, int), optional
        Genomic region to highlight across all panels.
    save_path : str, optional
        If set, save figure to this path.
    orientation : str
        'RNA' or 'genomic'.
    figsize : tuple
        Base figure dimensions.
    y_scale : str
        Y-axis transform: 'linear', 'log', or 'log1p'.
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required for signal track plots.")

    from visualization.gene_map import plot_gene_map

    sub = elements_df[elements_df["transcript_name"] == transcript_name]
    if sub.empty:
        logger.warning("No elements for transcript %s", transcript_name)
        return

    strand = sub["strand"].iloc[0]
    flip_x = orientation == "RNA" and strand == "-"
    chrom = sub["chrom"].iloc[0]
    plot_start = sub["start"].min()
    plot_end = sub["end"].max()

    if timepoints is None:
        timepoints = sorted(bw_files.keys())
    n_tracks = len(timepoints)

    height_ratios = [0.5] + [1] * n_tracks
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 2 * n_tracks))
    gs = fig.add_gridspec(n_tracks + 1, 1, height_ratios=height_ratios)

    ax_gene = fig.add_subplot(gs[0, 0])
    plot_gene_map(
        elements_df=sub,
        ax=ax_gene,
        title=gene,
        xlim=(plot_start, plot_end),
        highlight_area=highlight_area,
        orientation=orientation,
        start=plot_start,
        end=plot_end,
    )
    ax_gene.set_xticklabels([])

    colors = plt.cm.viridis(np.linspace(0, 1, n_tracks))
    for i, (bw_key, color) in enumerate(zip(timepoints, colors)):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=ax_gene)
        bw = pyBigWig.open(bw_files[bw_key])
        try:
            values = np.array(bw.values(str(chrom), plot_start, plot_end))
        except RuntimeError as e:
            logger.error("BigWig read error for %s: %s", bw_key, e)
            bw.close()
            continue

        if y_scale == "log":
            values = np.log(values + 1e-9)
        elif y_scale == "log1p":
            values = np.log1p(values)

        if flip_x:
            values = values[::-1]
            x_range = range(plot_end - 1, plot_start - 1, -1)
        else:
            x_range = range(plot_start, plot_end)

        if highlight_area is not None:
            left, right = sorted(highlight_area)
            ax.axvspan(left, right, color="orange", alpha=0.4, zorder=0)

        ax.plot(x_range, values, color=color)
        ax.fill_between(x_range, values, color=color, alpha=0.3)
        ax.set_ylabel(bw_key, fontsize=12)
        ax.grid(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if i < n_tracks - 1:
            ax.set_xticklabels([])
        if flip_x:
            ax.set_xlim(plot_end, plot_start)

        bw.close()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Signal track plot saved → %s", save_path)
    else:
        plt.tight_layout()
