"""
motif_tools — PWM utilities, enrichment analysis, and FASTA handling for motif discovery.
"""

from motif_tools.pwm_utils import (
    reverse_complement_ppm,
    write_motifs_to_meme,
    load_meme,
    convert_modisco_h5_to_meme,
)
from motif_tools.enrichment import (
    run_sea,
    run_fimo_enrichment,
    run_enrichment_analysis,
)
from motif_tools.fasta_utils import (
    write_fasta,
    load_fasta_as_dict,
    remove_fasta_overlaps,
    dedupe_fasta_headers,
)
