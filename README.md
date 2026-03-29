# genomic-model-interpretation

Interpretation and visualization toolkit for genomic sequence-to-function models (Borzoi, Enformer, etc.). Provides attribution analysis, in-silico mutagenesis, motif discovery with TF-MoDISco, enrichment testing, and publication-quality visualization — all with optional [grelu](https://github.com/Genentech/gReLU) integration.

## Modules

### `interpret/` — Model interpretation

| Function | Description |
|---|---|
| `get_attributions_for_element` | Compute per-base attribution scores for a genomic element using gradient-based methods |
| `attribution_native_only` | Zero out non-native base channels to isolate the reference-sequence signal |
| `mask_attributions` | Region-aware masking (exon-only, intron-only, upstream/downstream context, peak-only, etc.) |
| `evaluate_ism` | In-silico mutagenesis across a specified window with configurable bin aggregation |
| `run_prediction` | Forward-pass wrapper with automatic coordinate/transform handling |
| `run_modisco` | Run TF-MoDISco on attribution arrays, returning an H5 report |
| `extract_seqlets_from_h5` | Parse modiscolite H5 files into a seqlet DataFrame with genomic coordinates |
| `shift_seqlet_coordinates` | Convert attribution-tensor indices back to genome coordinates |

### `motif_tools/` — Motif I/O and enrichment

| Function | Description |
|---|---|
| `reverse_complement_ppm` | Reverse-complement a (4, L) position probability matrix |
| `write_motifs_to_meme` | Export a dict of PPMs to MEME minimal format |
| `load_meme` | Parse a MEME file into a dict of (4, L) arrays |
| `convert_modisco_h5_to_meme` | One-step conversion from modiscolite H5 → MEME file |
| `run_sea` | Simple Enrichment Analysis via the MEME Suite CLI |
| `run_fimo_enrichment` | Motif scanning with tangermeme's FIMO implementation |
| `run_enrichment_analysis` | Orchestrator: SEA + FIMO + caching + comparison |
| `validate_dna_sequences` | Check that a dict of sequences contains only valid DNA characters |
| `write_fasta` / `load_fasta_as_dict` | FASTA round-trip I/O |
| `remove_fasta_overlaps` | Remove overlapping regions on the same chromosome |
| `dedupe_fasta_headers` | Resolve duplicate FASTA headers by appending suffixes |

### `visualization/` — Publication-quality plots

| Function | Description |
|---|---|
| `plot_gene_map` | Render a gene-structure diagram (exons, introns, UTRs) from a GTF DataFrame |
| `plot_predictions` | Multi-track prediction overlay for model outputs |
| `plot_ism_heatmap` | 4×L heatmap of ISM effect sizes |
| `plot_read_density_tracks` | BigWig signal tracks aligned to a genomic region |
| `plot_attribution_logo` | Sequence logo from attribution scores (tangermeme backend) |
| `plot_logo_with_gene_map` | Combined logo + gene structure panel |
| `plot_motif_scatter` | Enrichment comparison scatter (log-fold-change vs. significance) |
| `plot_cosi_boxplot` | CoSI (Contribution per Sequence Index) box plots |

## Installation

```bash
pip install -e .                    # core only (numpy, pandas, h5py)
pip install -e ".[viz]"             # + matplotlib, seaborn, pyBigWig, adjustText
pip install -e ".[motif]"           # + tangermeme
pip install -e ".[grelu]"           # + grelu (Genentech genomic DL framework)
pip install -e ".[all]"             # everything
```

## Quick Start

```python
from interpret.attribution import get_attributions_for_element, mask_attributions
from interpret.modisco import run_modisco, extract_seqlets_from_h5
from motif_tools.pwm_utils import convert_modisco_h5_to_meme
from motif_tools.enrichment import run_enrichment_analysis
from visualization.logos import plot_attribution_logo

# 1. Compute attributions
attrs = get_attributions_for_element(model, seq_one_hot, genome_start, element_bed_row)

# 2. Mask to region of interest
masked = mask_attributions(attrs, mode="element_only", element_row=element_bed_row,
                           genome_bed_start=genome_start)

# 3. Run TF-MoDISco
run_modisco(masked, seq_one_hot, output_h5="results/modisco_report.h5")

# 4. Extract discovered motifs and convert to MEME format
convert_modisco_h5_to_meme("results/modisco_report.h5", "results/motifs.meme")

# 5. Enrichment analysis
results = run_enrichment_analysis(
    target_fasta="target_seqs.fa",
    background_fasta="control_seqs.fa",
    meme_motif_file="results/motifs.meme",
)

# 6. Visualize
plot_attribution_logo(masked)
```

## Architecture

The toolkit is designed around the Borzoi model's coordinate system (524,288 bp input → 196,608 bp output at 32 bp resolution) but adapts to any sequence-to-function architecture. All grelu and tangermeme dependencies are optional — the core interpretation logic depends only on NumPy, pandas, and h5py.

```
interpret/         Pure-Python interpretation logic (attribution, ISM, MoDISco)
motif_tools/       Motif format I/O, enrichment testing, FASTA utilities
visualization/     Matplotlib/seaborn plotting (tracks, logos, gene maps, scatter)
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
