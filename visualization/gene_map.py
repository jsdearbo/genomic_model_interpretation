"""
Gene structure visualization.

Renders exons, introns, and UTRs as a compact gene map with optional
region highlighting. Supports strand-aware RNA orientation (5'→3').
"""

import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _plot_gene_elements(
    ax,
    elements: pd.DataFrame,
    start: int = None,
    end: int = None,
    color_exon: str = "blue",
    color_intron: str = "black",
    exon_text_color: str = "white",
    intron_text_color: str = "black",
):
    """
    Render gene elements (exons, introns, UTRs) on a matplotlib axis.

    Exons are drawn as filled rectangles, introns as thin lines,
    and UTRs as shorter rectangles. Text labels are placed at each
    element's center.
    """
    elements_list = list(elements.itertuples())

    def _maybe_text(x, y, s, color, **kwargs):
        if color is None:
            return
        ax.text(x, y, s, **kwargs)

    # Draw introns
    for elem in elements_list:
        name = str(elem.element).lower()
        if "intron" not in name:
            continue
        ps = max(elem.start, start) if start is not None else elem.start
        pe = min(elem.end, end) if end is not None else elem.end
        if elem.end < (start or elem.start) or elem.start > (end or elem.end):
            continue
        ax.plot([ps, pe], [0, 0], color=color_intron, linewidth=2)
        _maybe_text(
            (ps + pe) / 2, 0.01, elem.element, intron_text_color,
            ha="center", va="bottom",
        )

    # Draw exons (excluding UTRs)
    for exon in elements_list:
        name = str(exon.element).lower()
        if "exon" not in name or "utr" in name:
            continue
        ps = max(exon.start, start) if start is not None else exon.start
        pe = min(exon.end, end) if end is not None else exon.end
        if exon.end < (start or exon.start) or exon.start > (end or exon.end):
            continue

        # Check for UTR overlap → truncate exon at UTR boundaries
        overlapping_utrs = [
            u for u in elements_list
            if (
                "utr" in str(getattr(u, "element_type", "")).lower()
                or "utr" in str(u.element).lower()
            )
            and hasattr(u, "chrom") and u.chrom == exon.chrom
            and hasattr(u, "transcript_id") and u.transcript_id == exon.transcript_id
            and not (u.end < ps or u.start > pe)
        ]

        if not overlapping_utrs:
            ax.add_patch(
                plt.Rectangle(
                    (ps, -0.5), pe - ps + 1, 1.0,
                    color=color_exon, linewidth=2,
                )
            )
            _maybe_text(
                (ps + pe) / 2, 0, exon.element, exon_text_color,
                ha="center", va="center", fontsize=8,
            )
        else:
            # Split exon into non-UTR segments
            segments = [(ps, pe)]
            for utr in overlapping_utrs:
                new_segs = []
                for seg_s, seg_e in segments:
                    if utr.start > seg_s:
                        new_segs.append((seg_s, utr.start - 1))
                    if utr.end < seg_e:
                        new_segs.append((utr.end + 1, seg_e))
                segments = [(s, e) for s, e in new_segs if s <= e]

            for seg_s, seg_e in segments:
                ax.add_patch(
                    plt.Rectangle(
                        (seg_s, -0.5), seg_e - seg_s + 1, 1.0,
                        color=color_exon, linewidth=2,
                    )
                )
                _maybe_text(
                    (seg_s + seg_e) / 2, 0, exon.element, exon_text_color,
                    ha="center", va="center", fontsize=8,
                )

    # Draw UTRs (shorter rectangles)
    for utr in elements_list:
        utr_name = str(utr.element).lower()
        utr_type = str(getattr(utr, "element_type", "")).lower()
        if "utr" not in utr_type and "utr" not in utr_name:
            continue
        ps = max(utr.start, start) if start is not None else utr.start
        pe = min(utr.end, end) if end is not None else utr.end
        if utr.end < (start or utr.start) or utr.start > (end or utr.end):
            continue
        ax.add_patch(
            plt.Rectangle(
                (ps, -0.25), pe - ps + 1, 0.5,
                color=color_exon, linewidth=1, edgecolor="black", zorder=10,
            )
        )
        _maybe_text(
            (ps + pe) / 2, 0, "utr", exon_text_color,
            ha="center", va="center", fontsize=8, zorder=11,
        )


def plot_gene_map(
    elements_df: pd.DataFrame,
    ax=None,
    title: str = "",
    xlim: tuple = None,
    start: int = None,
    end: int = None,
    color_exon: str = "#0068fa",
    color_intron: str = "#969696",
    exon_text_color: str = "white",
    intron_text_color: str = "black",
    highlight_area: tuple = None,
    orientation: str = "RNA",
    save_path: str = None,
    figsize: tuple = (20, 1),
    show_xaxis_label: bool = True,
):
    """
    Plot a gene structure map with exons, introns, and UTRs.

    Parameters
    ----------
    elements_df : pd.DataFrame
        Transcript elements with columns: chrom, start, end, element,
        element_type, strand, transcript_id.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Creates a new figure if None.
    title : str
        Plot title.
    xlim : tuple of (int, int), optional
        Explicit x-axis limits (start, end).
    start, end : int, optional
        Region bounds for element rendering.
    color_exon, color_intron : str
        Colors for exon bodies and intron lines.
    highlight_area : tuple of (int, int), optional
        Genomic region to highlight with a red shaded overlay.
    orientation : str
        'RNA' flips x-axis for minus-strand genes (5'→3' left-to-right).
        'genomic' preserves standard coordinate order.
    save_path : str, optional
        If set, saves the figure and closes it.
    figsize : tuple
        Figure size when creating a new figure.
    show_xaxis_label : bool
        Whether to display coordinate labels on the x-axis.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True

    if elements_df.empty:
        ax.text(0.5, 0.5, "No elements found", ha="center", va="center")
        ax.set_title(title)
        return

    strand = elements_df["strand"].iloc[0] if "strand" in elements_df.columns else "+"
    flip_x = orientation == "RNA" and strand == "-"

    if start is None or end is None:
        if xlim is not None:
            start, end = xlim
        else:
            start = elements_df["start"].min()
            end = elements_df["end"].max()

    chrom = elements_df["chrom"].iloc[0] if "chrom" in elements_df.columns else ""

    def _fmt(val):
        return f"{val:,}" if val is not None else "?"

    if highlight_area is not None:
        left, right = sorted(highlight_area)
        x_label = f"{chrom}:{_fmt(start)}-{_fmt(end)} | Highlight: {_fmt(left)}-{_fmt(right)}"
    else:
        left = right = None
        x_label = f"{chrom}:{_fmt(start)}-{_fmt(end)}"

    _plot_gene_elements(
        ax, elements_df,
        start=start, end=end,
        color_exon=color_exon, color_intron=color_intron,
        exon_text_color=exon_text_color, intron_text_color=intron_text_color,
    )

    if highlight_area is not None:
        ax.axvspan(left, right, color="red", alpha=0.35, zorder=10.5)

    ax.set_xlim(*(xlim or (start, end)))
    if flip_x:
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])

    ax.set_yticks([])
    ax.set_yticklabels([])
    if show_xaxis_label:
        ax.set_xlabel(x_label)
    ax.set_xticks([])
    ax.set_title(title, fontsize=24)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path is not None and own_fig:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
