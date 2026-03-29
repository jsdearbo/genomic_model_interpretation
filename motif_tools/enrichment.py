"""
Motif enrichment analysis using FIMO and SEA from the MEME Suite.

Compares motif occurrence frequencies between primary and control sequence
sets to identify enriched regulatory motifs.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional tangermeme dependency for programmatic FIMO
try:
    from tangermeme.tools.fimo import fimo as _tangermeme_fimo

    _HAS_TANGERMEME = True
except ImportError:
    _HAS_TANGERMEME = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# SEA enrichment (MEME Suite CLI)
# ---------------------------------------------------------------------------

def run_sea(
    primary_fasta: str,
    control_fasta: str,
    meme_file: str,
    output_dir: str,
    thresh: float = 1.0e6,
) -> pd.DataFrame:
    """
    Run SEA (Simple Enrichment Analysis) from the MEME Suite.

    Compares motif enrichment between primary and control FASTA files
    using the SEA command-line tool.

    Parameters
    ----------
    primary_fasta : str
        Path to the primary (positive) FASTA file.
    control_fasta : str
        Path to the control (negative) FASTA file.
    meme_file : str
        Path to a MEME-format motif file.
    output_dir : str
        Directory for SEA output.
    thresh : float
        E-value threshold for reporting.

    Returns
    -------
    pd.DataFrame
        SEA results table.

    Raises
    ------
    RuntimeError
        If the ``sea`` command fails.
    FileNotFoundError
        If the expected output file is not created.
    """
    os.makedirs(output_dir, exist_ok=True)

    primary_fasta = os.path.abspath(primary_fasta)
    control_fasta = os.path.abspath(control_fasta)
    meme_file = os.path.abspath(meme_file)

    cmd = [
        "sea",
        "--verbosity", "1",
        "--oc", output_dir,
        "--thresh", str(thresh),
        "--align", "center",
        "--p", primary_fasta,
        "--n", control_fasta,
        "--m", meme_file,
    ]

    logger.info("Running SEA: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if res.returncode != 0:
        logger.error("SEA failed. STDERR: %s", res.stderr)
        raise RuntimeError(f"SEA failed (exit {res.returncode})")

    tsv_path = os.path.join(output_dir, "sea.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"SEA completed but {tsv_path} not found.")

    return pd.read_csv(tsv_path, sep="\t")


# ---------------------------------------------------------------------------
# FIMO enrichment (tangermeme)
# ---------------------------------------------------------------------------

def _count_seqs_in_fasta(fasta_path: str) -> int:
    """Count the number of sequences in a FASTA file."""
    count = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def _normalize_fimo_output(hits) -> pd.DataFrame:
    """Normalize FIMO output to a single DataFrame."""
    if hits is None:
        return pd.DataFrame()
    if isinstance(hits, list):
        valid_dfs = [h for h in hits if isinstance(h, pd.DataFrame)]
        if not valid_dfs:
            return pd.DataFrame()
        return pd.concat(valid_dfs, ignore_index=True)
    if isinstance(hits, pd.DataFrame):
        return hits
    raise TypeError(f"Unexpected FIMO output type: {type(hits)}")


def _get_hit_percentage(fimo_hits, total_seqs: int) -> Dict[str, float]:
    """Calculate per-motif percentage of sequences with at least one hit."""
    if total_seqs == 0:
        return {}

    fimo_df = _normalize_fimo_output(fimo_hits)
    if fimo_df.empty:
        return {}

    seq_col = next(
        (c for c in ("sequence_name", "sequence_id", "seq_name") if c in fimo_df.columns),
        None,
    )
    motif_col = next(
        (c for c in ("motif_name", "motif_id") if c in fimo_df.columns),
        None,
    )
    if not (seq_col and motif_col):
        return {}

    counts = fimo_df.groupby(motif_col)[seq_col].nunique()
    return {motif: (count / total_seqs) * 100.0 for motif, count in counts.items()}


def run_fimo_enrichment(
    motifs: Dict[str, Any],
    primary_fasta: str,
    control_fasta: str,
    output_dir: str,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """
    Run FIMO motif scanning on primary and control FASTA files.

    Uses tangermeme's FIMO implementation to scan for motif occurrences
    and computes the percentage of sequences containing each motif.

    Parameters
    ----------
    motifs : dict
        Motif name → PPM (numpy array or torch tensor, shape (4, L)).
    primary_fasta : str
        Path to the primary FASTA file.
    control_fasta : str
        Path to the control FASTA file.
    output_dir : str
        Directory for output CSV.
    threshold : float
        FIMO p-value threshold.

    Returns
    -------
    pd.DataFrame
        Columns: motif_name, percent_match_primary, percent_match_ctrl.
    """
    if not _HAS_TANGERMEME:
        raise ImportError("tangermeme is required for FIMO enrichment.")
    if not _HAS_TORCH:
        raise ImportError("torch is required for FIMO enrichment.")

    primary_fasta = os.path.abspath(primary_fasta)
    control_fasta = os.path.abspath(control_fasta)

    # Validate motifs → must be tensors for tangermeme
    validated_motifs = {}
    for name, motif in motifs.items():
        if isinstance(motif, np.ndarray):
            t = torch.tensor(motif, dtype=torch.float32)
        elif torch.is_tensor(motif):
            t = motif.float()
        else:
            t = torch.tensor(np.asarray(motif), dtype=torch.float32)

        if t.shape[0] != 4 and t.shape[1] == 4:
            t = t.T
        if t.shape[0] != 4:
            logger.warning("Skipping motif %s: invalid shape %s", name, tuple(t.shape))
            continue

        validated_motifs[name] = t

    total_primary = _count_seqs_in_fasta(primary_fasta)
    total_control = _count_seqs_in_fasta(control_fasta)
    logger.info("Primary: %d seqs | Control: %d seqs", total_primary, total_control)

    primary_results = _tangermeme_fimo(
        motifs=validated_motifs,
        sequences=primary_fasta,
        reverse_complement=False,
        dim=0,
        threshold=threshold,
    )
    pct_primary = _get_hit_percentage(primary_results, total_primary)

    control_results = _tangermeme_fimo(
        motifs=validated_motifs,
        sequences=control_fasta,
        reverse_complement=False,
        dim=0,
        threshold=threshold,
    )
    pct_control = _get_hit_percentage(control_results, total_control)

    all_motifs = set(pct_primary) | set(pct_control) | set(validated_motifs)

    rows = [
        {
            "motif_name": m,
            "percent_match_primary": pct_primary.get(m, 0.0),
            "percent_match_ctrl": pct_control.get(m, 0.0),
        }
        for m in sorted(all_motifs)
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "fimo_enrichment.csv")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info("FIMO enrichment results → %s", output_csv)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_enrichment_analysis(
    primary_fasta: str,
    control_fasta: str,
    meme_file: str,
    output_dir: str,
    motifs: Optional[Dict[str, Any]] = None,
    run_sea_tool: bool = True,
    run_fimo_tool: bool = True,
    fimo_threshold: float = 0.001,
) -> Dict[str, pd.DataFrame]:
    """
    Run SEA and/or FIMO enrichment analysis.

    Orchestrates both enrichment methods, caching results to avoid re-runs.

    Parameters
    ----------
    primary_fasta : str
        Path to the primary FASTA file.
    control_fasta : str
        Path to the control FASTA file.
    meme_file : str
        Path to MEME motif file (used for SEA and optionally for loading FIMO motifs).
    output_dir : str
        Root output directory. SEA and FIMO get separate subdirectories.
    motifs : dict, optional
        Pre-loaded motifs for FIMO. If None, loaded from ``meme_file``.
    run_sea_tool : bool
        Whether to run SEA analysis.
    run_fimo_tool : bool
        Whether to run FIMO analysis.
    fimo_threshold : float
        P-value threshold for FIMO.

    Returns
    -------
    dict
        Mapping of {'sea': DataFrame, 'fimo': DataFrame} for completed analyses.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    if run_sea_tool:
        sea_dir = os.path.join(output_dir, "sea")
        sea_tsv = os.path.join(sea_dir, "sea.tsv")
        if not os.path.exists(sea_tsv):
            try:
                sea_results = run_sea(primary_fasta, control_fasta, meme_file, sea_dir)
                sea_df = pd.DataFrame(
                    {
                        "motif_name": sea_results["ID"],
                        "percent_match_primary": sea_results["TP%"],
                        "percent_match_ctrl": sea_results["FP%"],
                    }
                )
                sea_df.to_csv(os.path.join(sea_dir, "sea_enrichment.csv"), index=False)
                results["sea"] = sea_df
            except Exception as e:
                logger.error("SEA analysis failed: %s", e)
        else:
            logger.info("SEA results already exist in %s", sea_dir)
            results["sea"] = pd.read_csv(os.path.join(sea_dir, "sea_enrichment.csv"))

    if run_fimo_tool:
        fimo_dir = os.path.join(output_dir, "fimo")
        fimo_csv = os.path.join(fimo_dir, "fimo_enrichment.csv")
        if not os.path.exists(fimo_csv):
            try:
                if motifs is None:
                    from motif_tools.pwm_utils import load_meme

                    motifs = load_meme(meme_file)

                fimo_df = run_fimo_enrichment(
                    motifs=motifs,
                    primary_fasta=primary_fasta,
                    control_fasta=control_fasta,
                    output_dir=fimo_dir,
                    threshold=fimo_threshold,
                )
                results["fimo"] = fimo_df
            except Exception as e:
                logger.error("FIMO analysis failed: %s", e)
        else:
            logger.info("FIMO results already exist in %s", fimo_dir)
            results["fimo"] = pd.read_csv(fimo_csv)

    return results
