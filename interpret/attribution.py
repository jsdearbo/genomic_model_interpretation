"""
Attribution computation, native-base filtering, and region-based masking for
genomic foundation models.

Designed for models like Borzoi that predict over long genomic windows with
binned outputs. Attribution methods (saliency, DeepSHAP, integrated gradients)
score each input base's contribution to a predicted quantity.

Masking modes allow focusing attributions on specific genomic features:
    - element_only: keep only the target element ± flank
    - context_only: mask the element, keep surrounding context
    - upstream_exon / downstream_exon: keep neighboring exon(s)
    - adjacent_exons: keep both flanking exons (disjoint windows)
    - intron_and_adjacent_exons: keep the full E-I-E span
"""

import logging
from typing import Any, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional grelu dependency
try:
    import grelu.transforms.prediction_transforms
    from grelu.interpret.score import get_attributions as _grelu_get_attributions

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------

def get_attributions_for_element(
    model: Any,
    input_seq: str,
    selected_bins: Sequence[int],
    tasks: List[str],
    device: str = "cuda",
    method: str = "saliency",
    genome: str = "mm10",
    n_shuffles: int = 10,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Compute input-level attributions for a genomic element.

    Uses ``grelu.transforms.prediction_transforms.Aggregate`` to focus the prediction
    on specific output bins (e.g. the bins covering a target intron or exon) before
    back-propagating to compute per-base attribution scores.

    Parameters
    ----------
    model : LightningModel
        A grelu model (pretrained or fine-tuned).
    input_seq : str
        DNA input sequence of length matching the model's expected input.
    selected_bins : sequence of int
        Output bin indices to aggregate over before computing gradients.
    tasks : list of str
        Task names to aggregate (e.g. RNA-seq sample names).
    device : str
        Torch device string.
    method : str
        Attribution method — 'saliency', 'deepshap', or 'inputxgradient'.
    genome : str
        Genome assembly name ('mm10', 'hg38', etc.).
    n_shuffles : int
        Number of dinucleotide shuffles for DeepSHAP background.
    batch_size : int
        Batch size for attribution computation.

    Returns
    -------
    np.ndarray
        Attribution scores, shape (4, L) or model-dependent.
    """
    if not _HAS_GRELU:
        raise ImportError(
            "grelu is required for attribution computation. "
            "Install with: pip install grelu"
        )

    aggregate_transform = grelu.transforms.prediction_transforms.Aggregate(
        tasks=tasks,
        positions=selected_bins,
        length_aggfunc="mean",
        task_aggfunc="mean",
        model=model,
    )

    attrs = _grelu_get_attributions(
        model,
        seqs=[input_seq],
        genome=genome,
        prediction_transform=aggregate_transform,
        device=device,
        method=method,
        seed=0,
        hypothetical=False,
        n_shuffles=n_shuffles,
        batch_size=batch_size,
    )
    return attrs


# ---------------------------------------------------------------------------
# Native-base filtering
# ---------------------------------------------------------------------------

def attribution_native_only(
    attrs: np.ndarray,
    seq: str,
    on_unknown: str = "zero",
) -> np.ndarray:
    """
    Zero out attribution channels that don't correspond to the native base.

    For each position, only the channel matching the actual genomic base
    retains its value; all other channels are set to zero. This converts
    hypothetical attributions into observed-sequence scores.

    Parameters
    ----------
    attrs : np.ndarray
        Attribution array, shape (4, L) or (L, 4).
    seq : str
        DNA sequence of length L.
    on_unknown : str
        How to handle non-ACGT bases:
        - 'zero': zero the entire column
        - 'ignore': keep all channels
        - 'error': raise ValueError

    Returns
    -------
    np.ndarray
        Filtered attributions in the same shape as input.
    """
    attrs = np.asarray(attrs)
    if attrs.ndim != 2:
        raise ValueError(f"Expected 2D attrs, got shape {attrs.shape}")

    L = len(seq)
    if attrs.shape == (4, L):
        channel_first = True
        A = attrs
    elif attrs.shape == (L, 4):
        channel_first = False
        A = attrs.T
    else:
        raise ValueError(
            f"Expected attrs shape (4, L) or (L, 4); got {attrs.shape} for seq len {L}"
        )

    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros_like(A)

    for i, base in enumerate(seq):
        idx = base_to_idx.get(base.upper())
        if idx is not None:
            out[idx, i] = A[idx, i]
        elif on_unknown == "zero":
            continue
        elif on_unknown == "ignore":
            out[:, i] = A[:, i]
        elif on_unknown == "error":
            raise ValueError(f"Unknown base '{base}' at position {i}")
        else:
            raise ValueError(f"Invalid on_unknown option: {on_unknown}")

    return out if channel_first else out.T


# ---------------------------------------------------------------------------
# Strand-aware helpers for exon-based masking
# ---------------------------------------------------------------------------

def _parse_intron_name(intron_name: str) -> tuple:
    """Split 'transcriptName_intronNumber' → (transcript_name, intron_number)."""
    tname, num = intron_name.rsplit("_", 1)
    return tname, int(num)


def _five_prime_end(row, strand: str) -> int:
    return int(row["start"] if strand == "+" else row["end"])


def _three_prime_end(row, strand: str) -> int:
    return int(row["end"] if strand == "+" else row["start"])


def _transcript_block(elements_df: pd.DataFrame, transcript_name: str) -> pd.DataFrame:
    return (
        elements_df.loc[elements_df["transcript_name"] == transcript_name]
        .sort_values(["start", "end"])
        .reset_index(drop=True)
    )


def _adjacent_rows_transcriptional(
    elements_df: pd.DataFrame,
    transcript_name: str,
    intron_number: int,
    strand: str,
):
    """
    Return (upstream_row, intron_row, downstream_row) in transcriptional order.
    """
    tdf = _transcript_block(elements_df, transcript_name)
    mask = (tdf["element_type"] == "intron") & (tdf["element_number"] == intron_number)
    if not mask.any():
        raise KeyError(
            f"intron_{intron_number} not found for transcript '{transcript_name}'"
        )
    pos = int(mask[mask].index[0])

    if strand == "+":
        return tdf.iloc[pos - 1], tdf.iloc[pos], tdf.iloc[pos + 1]
    elif strand == "-":
        return tdf.iloc[pos + 1], tdf.iloc[pos], tdf.iloc[pos - 1]
    else:
        raise ValueError(f"Unexpected strand: {strand!r}")


# ---------------------------------------------------------------------------
# Region-based attribution masking
# ---------------------------------------------------------------------------

def mask_attributions(
    attributions: np.ndarray,
    coords: pd.DataFrame,
    mode: str = "element_only",
    flank: int = 0,
    seq_len: int = 524288,
    elements_df: pd.DataFrame = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Mask attributions to focus on specific genomic regions.

    Supports both simple modes (element-centered) and complex exon-aware
    modes that find neighboring exons using transcript structure.

    Parameters
    ----------
    attributions : np.ndarray
        Attribution array, shape (N, 4, L) where N is the number of elements.
    coords : pd.DataFrame
        Element coordinates with 'start', 'end' columns.  Exon-aware modes
        also require 'intron_name' and 'strand'.
    mode : str
        Masking mode. One of:
        - 'element_only' / 'intron_only' / 'peak_only': keep element ± flank
        - 'context_only': mask element ± flank, keep surrounding context
        - 'upstream_exon': keep the upstream flanking exon
        - 'downstream_exon': keep the downstream flanking exon
        - 'adjacent_exons': keep both flanking exons (disjoint)
        - 'intron_and_adjacent_exons': keep the full E-I-E span
    flank : int
        Bases of flanking context to include around kept regions.
    seq_len : int
        Total sequence length (model input size).
    elements_df : pd.DataFrame
        Transcript element table (required for exon-aware modes). Must contain
        'transcript_name', 'element_type', 'element_number', 'start', 'end'.
    in_place : bool
        If True, modify ``attributions`` in place.

    Returns
    -------
    np.ndarray
        Masked attributions (same shape as input).
    """
    if not in_place:
        attributions = attributions.copy()

    if not np.issubdtype(attributions.dtype, np.floating):
        attributions = attributions.astype(np.float32, copy=True)

    masked_value = 1e-20

    def _clamp(lo, hi):
        lo = max(0, int(lo))
        hi = min(seq_len, int(hi))
        return (lo, hi) if lo < hi else (None, None)

    for i, row in coords.iterrows():
        elem_len = abs(int(row["end"]) - int(row["start"]))
        elem_attr_start = (seq_len - elem_len) // 2
        elem_attr_end = elem_attr_start + elem_len

        intron_5p_idx = elem_attr_start
        intron_3p_idx = elem_attr_end

        if mode in ("element_only", "intron_only", "peak_only"):
            keep_lo, keep_hi = _clamp(
                elem_attr_start - flank, elem_attr_end + flank
            )
            if keep_lo is None:
                attributions[i, :, :] = masked_value
            else:
                attributions[i, :, :keep_lo] = masked_value
                attributions[i, :, keep_hi:] = masked_value

        elif mode == "context_only":
            core_lo, core_hi = _clamp(
                elem_attr_start - flank, elem_attr_end + flank
            )
            if core_lo is not None:
                attributions[i, :, core_lo:core_hi] = masked_value

        elif mode in {
            "upstream_exon",
            "downstream_exon",
            "adjacent_exons",
            "intron_and_adjacent_exons",
        }:
            if elements_df is None:
                raise ValueError(
                    "elements_df is required for exon-based masking modes."
                )
            if "intron_name" not in row:
                raise ValueError(
                    f"Mode '{mode}' requires 'intron_name' in coord data."
                )

            tname, inum = _parse_intron_name(str(row["intron_name"]))
            strand = str(row.get("strand", "+"))

            try:
                up_row, _intr_row, dn_row = _adjacent_rows_transcriptional(
                    elements_df, tname, inum, strand
                )
            except Exception as e:
                logger.warning(f"Skipping {mode} mask for {tname}: {e}")
                attributions[i, :, :] = masked_value
                continue

            up_exon_len = abs(int(up_row["end"]) - int(up_row["start"]))
            dn_exon_len = abs(int(dn_row["end"]) - int(dn_row["start"]))

            up_lo = intron_5p_idx - up_exon_len
            up_hi = intron_5p_idx
            dn_lo = intron_3p_idx
            dn_hi = intron_3p_idx + dn_exon_len

            if mode == "upstream_exon":
                keep_lo, keep_hi = _clamp(up_lo - flank, up_hi + flank)
                if keep_lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :keep_lo] = masked_value
                    attributions[i, :, keep_hi:] = masked_value

            elif mode == "downstream_exon":
                keep_lo, keep_hi = _clamp(dn_lo - flank, dn_hi + flank)
                if keep_lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :keep_lo] = masked_value
                    attributions[i, :, keep_hi:] = masked_value

            elif mode == "intron_and_adjacent_exons":
                keep_lo, keep_hi = _clamp(
                    (intron_5p_idx - up_exon_len) - flank,
                    (intron_3p_idx + dn_exon_len) + flank,
                )
                if keep_lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :keep_lo] = masked_value
                    attributions[i, :, keep_hi:] = masked_value

            elif mode == "adjacent_exons":
                up_keep = _clamp(up_lo - flank, up_hi + flank)
                dn_keep = _clamp(dn_lo - flank, dn_hi + flank)

                keep = np.zeros(seq_len, dtype=bool)
                if up_keep[0] is not None:
                    keep[up_keep[0] : up_keep[1]] = True
                if dn_keep[0] is not None:
                    keep[dn_keep[0] : dn_keep[1]] = True

                attributions[i, :, ~keep] = masked_value

        else:
            raise ValueError(f"Unknown masking mode: {mode}")

    return attributions
