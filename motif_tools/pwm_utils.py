"""
Position weight matrix (PWM) utilities.

Read and write motifs in MEME format, compute reverse complements,
and convert MoDISco H5 reports to standard motif files.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def reverse_complement_ppm(ppm: np.ndarray) -> np.ndarray:
    """
    Compute the reverse complement of a position probability matrix.

    Reverses position order and swaps A↔T, C↔G channels.

    Parameters
    ----------
    ppm : np.ndarray
        PPM with shape (4, L) where rows are [A, C, G, T].

    Returns
    -------
    np.ndarray
        Reverse-complement PPM, shape (4, L).
    """
    complement_indices = [3, 2, 1, 0]  # A↔T, C↔G
    return ppm[complement_indices, ::-1]


def reverse_complement_motifs_dict(
    motifs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Generate reverse-complement versions of all motifs in a dictionary.

    Keys are suffixed with '_rc'.
    """
    return {f"{key}_rc": reverse_complement_ppm(val) for key, val in motifs.items()}


def write_motifs_to_meme(
    motifs: Dict[str, Any],
    output_file: str,
    background: Optional[Dict[str, float]] = None,
) -> None:
    """
    Write motifs to a MEME-format file.

    Handles numpy arrays, torch tensors, and lists. Automatically transposes
    (4, L) matrices to (L, 4) for MEME's row-per-position format.

    Parameters
    ----------
    motifs : dict
        Mapping of motif name → PPM (shape (4, L) or (L, 4)).
    output_file : str
        Path for the output MEME file.
    background : dict, optional
        Nucleotide background frequencies {A, C, G, T}. Defaults to uniform.
    """
    with open(output_file, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")

        if background is None:
            background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        bg_line = " ".join(f"{base} {background[base]}" for base in "ACGT")
        f.write("Background letter frequencies:\n")
        f.write(f"{bg_line}\n\n")

        for name, ppm in motifs.items():
            # Convert tensor → numpy
            if hasattr(ppm, "numpy"):
                ppm = ppm.numpy()
            elif isinstance(ppm, list):
                ppm = np.array(ppm)

            # Ensure (L, 4) for writing
            if ppm.shape[0] == 4 and ppm.shape[1] != 4:
                ppm = ppm.T

            L = ppm.shape[0]
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {L}\n")
            for i in range(L):
                column = np.clip(ppm[i], 1e-6, 1.0)
                column = column / column.sum()
                f.write(" ".join(f"{val:.6f}" for val in column) + "\n")
            f.write("\n")

    logger.info("Motifs written to MEME file: %s", output_file)


def load_meme(meme_file: str) -> Dict[str, np.ndarray]:
    """
    Load motifs from a MEME-format file.

    Parameters
    ----------
    meme_file : str
        Path to a MEME motif file.

    Returns
    -------
    dict
        Mapping of motif name → numpy array with shape (4, L).
    """
    with open(meme_file, "r") as f:
        lines = f.readlines()

    motifs = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            motif_name = line.split()[1]
            # Skip to letter-probability matrix header
            while not lines[i].strip().startswith("letter-probability matrix"):
                i += 1
            header = lines[i].strip()
            w = int(header.split("w=")[1].split()[0])
            pwm = []
            i += 1
            for _ in range(w):
                vals = [float(x) for x in lines[i].strip().split()]
                if len(vals) == 4:
                    pwm.append(vals)
                i += 1
            # Store as (4, L)
            pwm_arr = np.array(pwm, dtype=np.float32).T
            if pwm_arr.shape[0] == 4:
                motifs[motif_name] = pwm_arr
        else:
            i += 1

    logger.info("Loaded %d motifs from %s", len(motifs), meme_file)
    return motifs


def convert_modisco_h5_to_meme(
    report_file: str,
    output_dir: str,
) -> None:
    """
    Convert a MoDISco H5 report to MEME format files.

    Generates three files in ``output_dir``:
    - forward.meme: original motif orientations
    - combined.meme: forward + reverse complement

    Parameters
    ----------
    report_file : str
        Path to the modiscolite H5 report file.
    output_dir : str
        Directory for output MEME files.
    """
    import os

    try:
        from grelu.io import motifs as grelu_motifs
    except ImportError:
        logger.error("grelu is required to read MoDISco H5 reports.")
        return

    if not os.path.exists(report_file):
        logger.warning("MoDISco report not found: %s", report_file)
        return

    modisco_output = grelu_motifs.read_modisco_report(report_file)
    rc_motifs = reverse_complement_motifs_dict(modisco_output)

    os.makedirs(output_dir, exist_ok=True)

    write_motifs_to_meme(
        motifs=modisco_output,
        output_file=os.path.join(output_dir, "forward.meme"),
    )
    write_motifs_to_meme(
        motifs={**modisco_output, **rc_motifs},
        output_file=os.path.join(output_dir, "combined.meme"),
    )

    logger.info("MoDISco H5 → MEME conversion complete in %s", output_dir)
