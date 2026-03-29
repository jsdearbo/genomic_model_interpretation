"""
TF-MoDISco integration for de novo motif discovery from attribution maps.

Extracts recurring patterns (seqlets) from attribution scores, clusters them
into motifs, and exports results in standard formats for downstream analysis.
"""

import logging
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional grelu dependency
try:
    import grelu.interpret.modisco

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def run_modisco(
    model: Any,
    input_seqs: Sequence[str],
    attributions: np.ndarray,
    output_dir: str,
    genome: str = "mm10",
    device: str = "cuda",
    window: int = 5000,
    sliding_window_size: int = 21,
    flank_size: int = 10,
    batch_size: int = 1024,
    meme_file: str = "CISBP_RNA_DNA_ENCODED",
) -> None:
    """
    Run TF-MoDISco on pre-computed attributions.

    Discovers recurring motif patterns in attribution maps by finding
    high-scoring subsequences (seqlets), clustering them, and matching
    against known motif databases.

    Parameters
    ----------
    model : LightningModel
        The grelu model used to generate attributions.
    input_seqs : sequence of str
        Input DNA sequences matching the attribution arrays.
    attributions : np.ndarray
        Pre-computed attributions, shape (N, 4, L).
    output_dir : str
        Directory for MoDISco output files (H5 report, MEME files, etc.).
    genome : str
        Genome assembly name.
    device : str
        Torch device.
    window : int
        Window size around attribution peaks for seqlet detection.
    sliding_window_size : int
        MoDISco sliding window (default 21; max seqlet = window + 2*flank).
    flank_size : int
        MoDISco flank size (default 10; yields 41bp max seqlet with default window).
    batch_size : int
        Batch size for ISM (used internally by MoDISco for some methods).
    meme_file : str
        Path or identifier for the motif database to match against.
    """
    if not _HAS_GRELU:
        raise ImportError(
            "grelu is required for MoDISco. Install with: pip install grelu"
        )

    grelu.interpret.modisco.run_modisco(
        model,
        seqs=input_seqs,
        genome=genome,
        meme_file=meme_file,
        method="completed",
        out_dir=output_dir,
        batch_size=batch_size,
        devices=device,
        num_workers=16,
        window=window,
        seed=0,
        attributions=attributions,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
    )
    logger.info("TF-MoDISco analysis completed → %s", output_dir)


def extract_seqlets_from_h5(
    h5_path: str,
    pattern_group: str = "both",
    indexing_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Extract seqlet coordinates from a modiscolite HDF5 report.

    Reads the seqlet start/end/strand from the H5 file and returns a DataFrame
    suitable for downstream coordinate mapping and visualization.

    Parameters
    ----------
    h5_path : str
        Path to the modiscolite H5 report file.
    pattern_group : str
        Which pattern groups to extract: 'both' (pos + neg), 'pos', or 'neg'.
    indexing_df : pd.DataFrame, optional
        If provided, used to resolve native strand for each example index.
        Must contain 'index' and 'strand' columns.

    Returns
    -------
    pd.DataFrame
        Columns: example_index, start, end, name, score, strand,
        pattern_label, is_revcomp.
    """
    bed_rows = []

    with h5py.File(h5_path, "r") as f:
        available = set(f.keys())

        def _resolve_groups(pg: str):
            if pg in ("both", "all", "patterns", None):
                groups = [g for g in ("pos_patterns", "neg_patterns") if g in available]
                if not groups:
                    groups = [g for g in available if g.endswith("_patterns")]
                return groups
            if pg in ("pos", "pos_patterns"):
                return [g for g in ("pos_patterns",) if g in available]
            if pg in ("neg", "neg_patterns"):
                return [g for g in ("neg_patterns",) if g in available]
            return [pg] if pg in available else []

        groups = _resolve_groups(pattern_group)
        if not groups:
            raise KeyError(
                f"No pattern groups found for '{pattern_group}'. "
                f"Available: {sorted(available)}"
            )

        for group_name in groups:
            group = f[group_name]
            prefix = "pos" if group_name.startswith("pos") else "neg"

            for pattern_name in group.keys():
                pattern = group[pattern_name]
                seqlets = pattern["seqlets"]
                starts = seqlets["start"][:]
                ends = seqlets["end"][:]
                strands = seqlets["is_revcomp"][:]
                example_idxs = seqlets["example_idx"][:]

                for j in range(len(starts)):
                    example = int(example_idxs[j])

                    # Determine native strand
                    if indexing_df is not None:
                        native_strand = indexing_df.loc[
                            indexing_df["index"] == example, "strand"
                        ].values[0]
                    else:
                        native_strand = "+"

                    strand = (
                        ("-" if native_strand == "+" else "+")
                        if strands[j]
                        else native_strand
                    )
                    name = f"{pattern_name}_seqlet_{j}"
                    pattern_label = f"{prefix}_{pattern_name}"
                    bed_rows.append(
                        [
                            example,
                            int(starts[j]),
                            int(ends[j]),
                            name,
                            0,
                            strand,
                            pattern_label,
                            bool(strands[j]),
                        ]
                    )

    return pd.DataFrame(
        bed_rows,
        columns=[
            "example_index",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "pattern_label",
            "is_revcomp",
        ],
    )


def shift_seqlet_coordinates(
    seqlets_df: pd.DataFrame,
    offset: int,
) -> pd.DataFrame:
    """
    Shift seqlet start/end coordinates by a fixed offset.

    Typically used to convert from local modisco-window coordinates to
    full-sequence tensor coordinates.

    Parameters
    ----------
    seqlets_df : pd.DataFrame
        Seqlet DataFrame with 'start' and 'end' columns.
    offset : int
        Value to add to start and end.

    Returns
    -------
    pd.DataFrame
        Copy with shifted coordinates.
    """
    df = seqlets_df.copy()
    df["start"] = df["start"] + offset
    df["end"] = df["end"] + offset
    return df
