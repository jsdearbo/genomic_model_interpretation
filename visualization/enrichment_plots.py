"""
Enrichment comparison plots.

Scatter plots and boxplots for comparing motif enrichment between
primary and control sets, including fold-change coloring and smart labeling.
"""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_motif_scatter(
    df: pd.DataFrame,
    x_col: str = "percent_match_ctrl",
    y_col: str = "percent_match_primary",
    label_col: str = "motif_name",
    title: str = "Motif Enrichment: Primary vs Control",
    save_path: Optional[str] = None,
    top_n: int = 5,
    dot_color: str = "royalblue",
    figsize: tuple = (8, 8),
):
    """
    Scatter plot of motif hit percentages in primary vs control sets.

    Points above the diagonal are enriched in the primary set; points
    below are depleted. Labels the top-N most divergent motifs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for x_col, y_col, and label_col.
    x_col : str
        Column for control-set percentages (x-axis).
    y_col : str
        Column for primary-set percentages (y-axis).
    label_col : str
        Column for motif name labels.
    title : str
        Plot title.
    save_path : str, optional
        If set, saves the figure to this path.
    top_n : int
        Number of most-divergent motifs to label.
    dot_color : str
        Color for scatter points.
    figsize : tuple
        Figure dimensions.
    """
    if df.empty:
        logger.warning("Empty DataFrame — nothing to plot.")
        return

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df[x_col], df[y_col],
        color=dot_color, edgecolor="white", s=90, alpha=0.85,
        label="Motifs", zorder=2,
    )

    # Diagonal (no-enrichment line)
    max_val = max(df[x_col].max(), df[y_col].max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="y = x", zorder=1)

    # Label top-N most divergent motifs
    deviation = (df[y_col] - df[x_col]).abs()
    top = df.loc[deviation.nlargest(top_n).index]

    texts = []
    for _, row in top.iterrows():
        texts.append(
            ax.text(
                row[x_col], row[y_col], str(row[label_col]),
                fontsize=12, fontweight="bold", color="black", alpha=0.95, zorder=3,
            )
        )

    try:
        from adjustText import adjust_text

        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.2, alpha=0.7),
        )
    except ImportError:
        pass

    ax.set_title(title, pad=15)
    ax.set_xlabel("% Sequences with Motif (Control)", labelpad=10)
    ax.set_ylabel("% Sequences with Motif (Primary)", labelpad=10)
    ax.grid(True, linestyle=":", linewidth=1, alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Motif scatter plot saved → %s", save_path)
    else:
        plt.show()


def plot_cosi_boxplot(
    df: pd.DataFrame,
    cosi_prefix: str = "CoSI",
    save_path: str = None,
    title: str = "CoSI",
    highlight_names: list = None,
    figsize: tuple = (12, 8),
):
    """
    Boxplot of CoSI (co-splicing index) values across timepoints.

    Columns starting with ``cosi_prefix`` are treated as timepoints.
    Overlay outlier points with jitter and optionally highlight specific elements.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with CoSI columns named like 'CoSI_<timepoint>'.
    cosi_prefix : str
        Column name prefix to identify CoSI value columns.
    save_path : str, optional
        If set, saves the figure.
    title : str
        Plot title.
    highlight_names : list, optional
        Element names to highlight in red.
    figsize : tuple
        Figure dimensions.
    """
    import re

    cosi_cols = [c for c in df.columns if c.startswith(cosi_prefix)]
    if not cosi_cols:
        raise ValueError(f"No columns starting with '{cosi_prefix}' found.")

    def _extract_minutes(tp):
        m = re.search(r"_(\d+)_", tp)
        return int(m.group(1)) if m else float("inf")

    cosi_cols = sorted(cosi_cols, key=_extract_minutes)

    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in cosi_cols],
        value_vars=cosi_cols,
        var_name="timepoint",
        value_name="CoSI",
    ).dropna(subset=["CoSI"])

    if highlight_names is not None and "intron_name" in long_df.columns:
        long_df["highlight"] = long_df["intron_name"].isin(highlight_names)
    else:
        long_df["highlight"] = False

    x_labels = [str(_extract_minutes(tp)) for tp in cosi_cols]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=long_df, x="timepoint", y="CoSI",
        order=cosi_cols, palette="coolwarm", ax=ax,
    )

    # Overlay outliers with jitter
    for i, tp in enumerate(cosi_cols):
        tp_data = long_df[long_df["timepoint"] == tp]
        q1 = tp_data["CoSI"].quantile(0.25)
        q3 = tp_data["CoSI"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = tp_data[(tp_data["CoSI"] < lower) | (tp_data["CoSI"] > upper)]

        x_jitter = i + np.random.uniform(-0.15, 0.15, size=len(outliers))
        ax.scatter(x_jitter, outliers["CoSI"], color="black", alpha=0.6, zorder=10)

        if highlight_names is not None:
            highlighted = outliers[outliers["highlight"]]
            if not highlighted.empty:
                xl = i + np.random.uniform(-0.15, 0.15, size=len(highlighted))
                ax.scatter(
                    xl, highlighted["CoSI"],
                    color="red", s=60, edgecolor="white", zorder=11,
                )

    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Timepoint", fontsize=20)
    ax.set_ylabel("CoSI Value", fontsize=20)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        logger.info("CoSI boxplot saved → %s", save_path)
