"""Tests for motif_tools.fasta_utils module."""

import os
import tempfile

import pytest

from motif_tools.fasta_utils import (
    validate_dna_sequences,
    write_fasta,
    load_fasta_as_dict,
    remove_fasta_overlaps,
    dedupe_fasta_headers,
)


def _write_fasta_file(records: dict, path: str) -> None:
    """Helper to write a dict of {header: seq} to a FASTA file."""
    with open(path, "w") as f:
        for name, seq in records.items():
            f.write(f">{name}\n{seq}\n")


class TestValidateDnaSequences:
    def test_valid_sequences_returned(self):
        seqs = ["ACGTACGT", "aaacccgggttt"]
        result = validate_dna_sequences(seqs)
        assert result == seqs

    def test_non_string_raises(self):
        import numpy as np
        with pytest.raises(TypeError):
            validate_dna_sequences([np.array([1, 2, 3])])

    def test_stringified_array_raises(self):
        with pytest.raises(ValueError):
            validate_dna_sequences(["[0.1 0.2 0.3 0.4]"])

    def test_n_characters_accepted(self):
        seqs = ["ACGTNNNacgt"]
        result = validate_dna_sequences(seqs)
        assert result == seqs


class TestFastaRoundTrip:
    def test_write_then_read(self):
        names = ["chr1:100-200", "chr2:300-400"]
        seqs = ["ACGTACGT", "TTTTAAAA"]

        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            path = f.name

        try:
            write_fasta(seqs, names, path)
            loaded = load_fasta_as_dict(path)
            assert loaded == dict(zip(names, seqs))
        finally:
            os.unlink(path)

    def test_empty_sequences_skipped(self):
        """Empty sequences should be skipped."""
        names = ["has_seq", "empty_seq"]
        seqs = ["ACGT", ""]

        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            path = f.name

        try:
            write_fasta(seqs, names, path)
            loaded = load_fasta_as_dict(path)
            assert "has_seq" in loaded
            assert "empty_seq" not in loaded
        finally:
            os.unlink(path)


class TestRemoveFastaOverlaps:
    def test_no_overlaps_all_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_path = os.path.join(tmpdir, "primary.fa")
            control_path = os.path.join(tmpdir, "control.fa")
            output_path = os.path.join(tmpdir, "output.fa")

            _write_fasta_file({"gene_A": "AAAA"}, primary_path)
            _write_fasta_file({"gene_B": "CCCC", "gene_C": "GGGG"}, control_path)

            result_path = remove_fasta_overlaps(primary_path, control_path, output_path)
            loaded = load_fasta_as_dict(result_path)
            assert len(loaded) == 2

    def test_overlapping_header_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_path = os.path.join(tmpdir, "primary.fa")
            control_path = os.path.join(tmpdir, "control.fa")
            output_path = os.path.join(tmpdir, "output.fa")

            _write_fasta_file({"shared_gene": "AAAA"}, primary_path)
            _write_fasta_file(
                {"shared_gene": "CCCC", "unique_gene": "GGGG"}, control_path
            )

            result_path = remove_fasta_overlaps(primary_path, control_path, output_path)
            loaded = load_fasta_as_dict(result_path)
            assert "unique_gene" in loaded
            assert "shared_gene" not in loaded


class TestDedupeFastaHeaders:
    def test_no_dupes_all_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.fa")
            output_path = os.path.join(tmpdir, "output.fa")

            _write_fasta_file({"a": "AAAA", "b": "CCCC"}, input_path)
            dropped = dedupe_fasta_headers(input_path, output_path)
            assert dropped == 0
            loaded = load_fasta_as_dict(output_path)
            assert set(loaded.keys()) == {"a", "b"}

    def test_duplicate_header_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.fa")
            output_path = os.path.join(tmpdir, "output.fa")

            # Manually write duplicate headers (can't use dict for dupes)
            with open(input_path, "w") as f:
                f.write(">seq\nAAAA\n>seq\nCCCC\n>other\nGGGG\n")

            dropped = dedupe_fasta_headers(input_path, output_path)
            assert dropped == 1
            loaded = load_fasta_as_dict(output_path)
            assert "seq" in loaded
            assert "other" in loaded
