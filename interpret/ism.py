"""
In silico mutagenesis (ISM) for genomic foundation models.

Systematically introduces point mutations across a region of interest and
measures the predicted effect (log2 fold-change) on output tracks, revealing
which bases contribute most to model predictions.
"""

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional grelu dependency
try:
    import grelu.interpret.score

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def evaluate_ism(
    model: Any,
    input_seq: Sequence[str],
    prediction_transform: Any,
    start_pos: int,
    end_pos: int,
    input_start: int = 0,
    batch_size: int = 8,
    compare_func: str = "log2FC",
) -> pd.DataFrame:
    """
    Run in silico mutagenesis over a genomic region.

    For each position in [start_pos, end_pos), substitutes every alternative
    base and computes the change in aggregated prediction relative to the
    wild-type sequence.

    Parameters
    ----------
    model : LightningModel
        A grelu genomic model.
    input_seq : sequence of str
        Wild-type input DNA sequence(s).
    prediction_transform : callable
        Aggregation transform applied to model output before comparison
        (e.g., an ``Aggregate`` that focuses on specific bins/tasks).
    start_pos : int
        Genomic start coordinate of the mutation window.
    end_pos : int
        Genomic end coordinate of the mutation window.
    input_start : int
        Genomic coordinate of the first base in ``input_seq``.
    batch_size : int
        Mutations per batch for inference.
    compare_func : str
        How to compare mutant vs wild-type — 'log2FC', 'diff', or 'ratio'.

    Returns
    -------
    pd.DataFrame
        ISM scores indexed by position and alternative allele.
    """
    if not _HAS_GRELU:
        raise ImportError(
            "grelu is required for ISM. Install with: pip install grelu"
        )

    logger.info(
        "Running ISM from %d to %d (relative %d–%d)",
        start_pos,
        end_pos,
        start_pos - input_start,
        end_pos - input_start,
    )
    t0 = time.time()

    ism = grelu.interpret.score.ISM_predict(
        seqs=input_seq,
        model=model,
        prediction_transform=prediction_transform,
        devices=[0],
        batch_size=batch_size,
        num_workers=1,
        start_pos=start_pos - input_start,
        end_pos=end_pos - input_start,
        return_df=True,
        compare_func=compare_func,
    )

    logger.info("ISM completed in %.1f s", time.time() - t0)
    return ism


def run_prediction(
    model: Any,
    input_seqs: Sequence[str],
    device: str = "cuda",
) -> np.ndarray:
    """
    Run model predictions on one or more input sequences.

    Parameters
    ----------
    model : LightningModel
        A grelu genomic model.
    input_seqs : sequence of str
        DNA input sequences.
    device : str
        Torch device for inference.

    Returns
    -------
    np.ndarray
        Prediction tensor with shape (n_seqs, n_bins, n_tasks).
    """
    t0 = time.time()
    preds = model.predict_on_seqs(input_seqs, device=device)
    logger.info("Prediction shape: %s (%.1f s)", preds.shape, time.time() - t0)
    return preds
