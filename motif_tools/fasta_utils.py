"""
FASTA file utilities for motif discovery pipelines.

Provides DNA sequence validation, FASTA I/O, overlap removal between
primary and control sets, and header deduplication.
"""

import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

_VALID_DNA = set("ACGTNacgtn")


def validate_dna_sequences(seqs: List[str]) -> List[str]:
    """
    Validate that all sequences are DNA strings (not arrays or tensors).

    Catches a common mistake where attribution tensors are accidentally
    passed to FIMO instead of actual DNA sequences.

    Parameters
    ----------
    seqs : list of str
        Candidate DNA sequences.

    Returns
    -------
    list of str
        Validated DNA sequences.

    Raises
    ------
    TypeError
        If a sequence is not a string.
    ValueError
        If a string looks like a stringified array.
    """
    cleaned = []
    for i, s in enumerate(seqs):
        if not isinstance(s, str):
            raise TypeError(
                f"Sequence {i} is {type(s)}, expected str. "
                "You may be passing attribution tensors instead of DNA."
            )
        stripped = s.strip()
        if stripped.startswith("[") or (" " in stripped and not stripped.isalpha()):
            raise ValueError(
                f"Sequence {i} appears to be a stringified array: {stripped[:50]}..."
            )
        cleaned.append(s)
    return cleaned


def write_fasta(
    sequences: List[str],
    names: List[str],
    output_path: str,
) -> None:
    """
    Write DNA sequences to a FASTA file with validation.

    Parameters
    ----------
    sequences : list of str
        DNA sequences to write.
    names : list of str
        Corresponding sequence names (FASTA headers).
    output_path : str
        Path for the output FASTA file.
    """
    sequences = validate_dna_sequences(sequences)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for name, seq in zip(names, sequences):
            if not seq:
                continue
            clean_name = str(name).replace(" ", "_").replace("\t", "_")
            f.write(f">{clean_name}\n{seq.strip()}\n")


def load_fasta_as_dict(fasta_path: str) -> Dict[str, str]:
    """
    Load a FASTA file into a dictionary.

    Parameters
    ----------
    fasta_path : str
        Path to a FASTA file.

    Returns
    -------
    dict
        Mapping of sequence name → sequence string.
    """
    fasta_dict = {}
    with open(fasta_path, "r") as f:
        seq_name = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_name is not None:
                    fasta_dict[seq_name] = "".join(seq_chunks)
                seq_name = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_name is not None:
            fasta_dict[seq_name] = "".join(seq_chunks)
    return fasta_dict


def remove_fasta_overlaps(
    primary_fasta: str,
    control_fasta: str,
    output_fasta: str = None,
) -> str:
    """
    Remove sequences from a control FASTA that appear in the primary set.

    Removes by header name matching to ensure primary and control sets are
    non-overlapping for unbiased enrichment analysis.

    Parameters
    ----------
    primary_fasta : str
        Path to the primary (positive) FASTA file.
    control_fasta : str
        Path to the control FASTA file.
    output_fasta : str, optional
        Path for the filtered output. Defaults to
        ``<control_basename>_no_overlaps.fa``.

    Returns
    -------
    str
        Path to the output FASTA (with overlaps removed).
    """
    if output_fasta is None:
        base = os.path.splitext(control_fasta)[0]
        output_fasta = f"{base}_no_overlaps.fa"

    primary_headers = set()
    with open(primary_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                primary_headers.add(line[1:].strip())

    overlaps_found = 0
    kept = 0
    with open(control_fasta, "r") as fin, open(output_fasta, "w") as fout:
        current_header = None
        current_seq: List[str] = []

        def _flush():
            nonlocal current_header, current_seq, overlaps_found, kept
            if current_header is None:
                return
            if current_header in primary_headers:
                overlaps_found += 1
            else:
                fout.write(f">{current_header}\n")
                seq = "".join(current_seq)
                for i in range(0, len(seq), 80):
                    fout.write(seq[i : i + 80] + "\n")
                kept += 1
            current_header = None
            current_seq = []

        for line in fin:
            line = line.strip()
            if line.startswith(">"):
                _flush()
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        _flush()

    logger.info(
        "Overlap removal: %d removed, %d kept → %s",
        overlaps_found,
        kept,
        output_fasta,
    )
    return output_fasta


def dedupe_fasta_headers(
    input_fasta: str,
    output_fasta: str,
) -> int:
    """
    Remove duplicate FASTA records by header (first token).

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    output_fasta : str
        Path for the deduplicated output.

    Returns
    -------
    int
        Number of duplicate records dropped.
    """
    seen: set = set()
    dropped = 0

    with open(input_fasta) as fin, open(output_fasta, "w") as fout:
        hdr = None
        seq_lines: List[str] = []

        def _flush():
            nonlocal hdr, seq_lines, dropped
            if hdr is None:
                return
            key = hdr.split()[0]
            if key in seen:
                dropped += 1
            else:
                seen.add(key)
                fout.write(f">{hdr}\n")
                fout.write("".join(seq_lines))
            hdr = None
            seq_lines = []

        for line in fin:
            if line.startswith(">"):
                _flush()
                hdr = line[1:].strip()
            else:
                if hdr is not None:
                    seq_lines.append(line)
        _flush()

    logger.info("Deduplicated FASTA: %d duplicates dropped → %s", dropped, output_fasta)
    return dropped
