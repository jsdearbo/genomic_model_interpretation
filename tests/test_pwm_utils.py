"""Tests for motif_tools.pwm_utils module."""

import os
import tempfile

import numpy as np
import pytest

from motif_tools.pwm_utils import (
    reverse_complement_ppm,
    reverse_complement_motifs_dict,
    write_motifs_to_meme,
    load_meme,
)


class TestReverseComplementPPM:
    def test_identity_after_double_rc(self):
        """Two reverse complements should return the original."""
        ppm = np.random.dirichlet([1, 1, 1, 1], size=10).T  # (4, 10)
        rc = reverse_complement_ppm(ppm)
        rc_rc = reverse_complement_ppm(rc)
        np.testing.assert_allclose(ppm, rc_rc, atol=1e-7)

    def test_swaps_and_reverses(self):
        """A at pos 0 should map to T at last position after RC."""
        ppm = np.zeros((4, 5), dtype=np.float32)
        ppm[0, 0] = 1.0  # A at position 0
        rc = reverse_complement_ppm(ppm)
        # Should be T (index 3) at position 4 (last)
        assert rc[3, 4] == 1.0

    def test_shape_preserved(self):
        ppm = np.random.rand(4, 12).astype(np.float32)
        rc = reverse_complement_ppm(ppm)
        assert rc.shape == (4, 12)


class TestReverseComplementDict:
    def test_key_suffix(self):
        motifs = {"motif_0": np.random.rand(4, 8)}
        rc_dict = reverse_complement_motifs_dict(motifs)
        assert "motif_0_rc" in rc_dict
        assert len(rc_dict) == 1


class TestMEMERoundTrip:
    def test_write_then_read(self):
        """Motifs should survive a write→read round trip."""
        motifs = {
            "test_motif_A": np.random.dirichlet([1, 1, 1, 1], size=8).T.astype(np.float32),
            "test_motif_B": np.random.dirichlet([1, 1, 1, 1], size=12).T.astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".meme", delete=False) as f:
            path = f.name

        try:
            write_motifs_to_meme(motifs, path)
            loaded = load_meme(path)

            assert set(loaded.keys()) == set(motifs.keys())
            for name in motifs:
                assert loaded[name].shape == motifs[name].shape
                # Values should be close (clipping + normalization add small changes)
                np.testing.assert_allclose(
                    loaded[name].sum(axis=0),
                    np.ones(motifs[name].shape[1]),
                    atol=1e-4,
                )
        finally:
            os.unlink(path)

    def test_empty_motifs(self):
        """Writing empty dict should produce a valid but motif-less file."""
        with tempfile.NamedTemporaryFile(suffix=".meme", delete=False) as f:
            path = f.name

        try:
            write_motifs_to_meme({}, path)
            loaded = load_meme(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_transposed_input(self):
        """(L, 4) input should be auto-transposed."""
        motif = np.random.dirichlet([1, 1, 1, 1], size=6)  # (6, 4) — NOT (4, 6)

        with tempfile.NamedTemporaryFile(suffix=".meme", delete=False) as f:
            path = f.name

        try:
            write_motifs_to_meme({"motif": motif}, path)
            loaded = load_meme(path)
            assert loaded["motif"].shape == (4, 6)
        finally:
            os.unlink(path)
