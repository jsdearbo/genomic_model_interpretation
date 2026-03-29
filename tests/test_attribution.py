"""Tests for interpret.attribution module."""

import numpy as np
import pandas as pd
import pytest

from interpret.attribution import (
    attribution_native_only,
    mask_attributions,
    _parse_intron_name,
)


# ---------------------------------------------------------------------------
# attribution_native_only
# ---------------------------------------------------------------------------

class TestAttributionNativeOnly:
    """Tests for the native-base filtering function."""

    def test_basic_filtering_channel_first(self):
        """Keep only native base channel, zero others."""
        seq = "ACGT"
        attrs = np.ones((4, 4), dtype=np.float32)
        result = attribution_native_only(attrs, seq)

        # Position 0: base A → channel 0
        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0
        assert result[2, 0] == 0.0
        assert result[3, 0] == 0.0

        # Position 1: base C → channel 1
        assert result[1, 1] == 1.0
        assert result[0, 1] == 0.0

        # Position 2: base G → channel 2
        assert result[2, 2] == 1.0

        # Position 3: base T → channel 3
        assert result[3, 3] == 1.0

    def test_channel_last_input(self):
        """(L, 4) input should return (L, 4)."""
        seq = "ACG"
        attrs = np.ones((3, 4), dtype=np.float32)
        result = attribution_native_only(attrs, seq)
        assert result.shape == (3, 4)
        # Position 0, channel A
        assert result[0, 0] == 1.0
        assert result[0, 1] == 0.0

    def test_unknown_base_zero(self):
        """N bases should zero the column by default."""
        seq = "AN"
        attrs = np.ones((4, 2), dtype=np.float32)
        result = attribution_native_only(attrs, seq, on_unknown="zero")
        assert result[:, 1].sum() == 0.0

    def test_unknown_base_ignore(self):
        """on_unknown='ignore' keeps all channels for unknown bases."""
        seq = "AN"
        attrs = np.ones((4, 2), dtype=np.float32)
        result = attribution_native_only(attrs, seq, on_unknown="ignore")
        assert result[:, 1].sum() == 4.0

    def test_unknown_base_error(self):
        """on_unknown='error' raises on N."""
        seq = "AN"
        attrs = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown base"):
            attribution_native_only(attrs, seq, on_unknown="error")

    def test_shape_mismatch_raises(self):
        """Mismatched seq length and array dimension should raise."""
        seq = "ACGT"
        attrs = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            attribution_native_only(attrs, seq)


# ---------------------------------------------------------------------------
# mask_attributions
# ---------------------------------------------------------------------------

class TestMaskAttributions:
    """Tests for the region-based masking function."""

    def _make_attrs(self, n=1, seq_len=1000, channels=4):
        return np.ones((n, channels, seq_len), dtype=np.float32)

    def test_element_only_masks_outside(self):
        """element_only should zero everything outside element ± flank."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        result = mask_attributions(attrs, coords, mode="element_only", seq_len=1000)

        # Element is 200bp long → centered at [400, 600] in a 1000bp window
        elem_len = 200
        attr_start = (1000 - elem_len) // 2  # 400
        attr_end = attr_start + elem_len      # 600

        # Inside element should be 1.0
        assert result[0, 0, attr_start] == 1.0
        assert result[0, 0, attr_end - 1] == 1.0

        # Outside should be masked (~0)
        assert result[0, 0, 0] < 1e-10
        assert result[0, 0, 999] < 1e-10

    def test_element_only_with_flank(self):
        """element_only with flank should extend the kept region."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        result = mask_attributions(
            attrs, coords, mode="element_only", flank=50, seq_len=1000,
        )

        elem_len = 200
        attr_start = (1000 - elem_len) // 2
        # With flank=50, kept region extends 50bp more each side
        assert result[0, 0, attr_start - 50] == 1.0
        assert result[0, 0, attr_start - 51] < 1e-10

    def test_context_only_masks_inside(self):
        """context_only should zero the element region, keep the rest."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        result = mask_attributions(attrs, coords, mode="context_only", seq_len=1000)

        elem_len = 200
        attr_start = (1000 - elem_len) // 2
        attr_end = attr_start + elem_len

        # Inside element should be masked
        assert result[0, 0, attr_start + 1] < 1e-10
        # Outside should be kept
        assert result[0, 0, 0] == 1.0
        assert result[0, 0, 999] == 1.0

    def test_in_place_modifies_original(self):
        """in_place=True should modify the original array."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        mask_attributions(attrs, coords, mode="element_only", seq_len=1000, in_place=True)
        assert attrs[0, 0, 0] < 1e-10  # original was modified

    def test_not_in_place_preserves_original(self):
        """Default (in_place=False) should not modify the original."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        mask_attributions(attrs, coords, mode="element_only", seq_len=1000)
        assert attrs[0, 0, 0] == 1.0  # original unchanged

    def test_unknown_mode_raises(self):
        """Unknown mode should raise ValueError."""
        attrs = self._make_attrs(1, 1000)
        coords = pd.DataFrame({"start": [400], "end": [600]})
        with pytest.raises(ValueError, match="Unknown masking mode"):
            mask_attributions(attrs, coords, mode="invalid_mode", seq_len=1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestParseIntronName:
    def test_simple(self):
        assert _parse_intron_name("Myl4-201_3") == ("Myl4-201", 3)

    def test_complex_name(self):
        assert _parse_intron_name("Srsf3-ENST_12") == ("Srsf3-ENST", 12)
