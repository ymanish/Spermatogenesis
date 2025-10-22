#!/usr/bin/env bash
# subset_by_pos.sh — make a subset of id_lookup.tsv by dropping rows with pos in [2010,2030]
# Usage: ./subset_by_pos.sh [in.tsv] [out.tsv]
# Defaults: in=id_lookup.tsv, out=id_lookup_subset.tsv

set -euo pipefail

# Input and output file paths
in="${1:-/group/pol_schiessel/Manish/HAMNucRetSeq_pipeline/output/minpoint_unboundpromoter_regions_breath/crystal_freedna_md_merged/reduced/id_lookup.tsv}"
out="${2:-/group/pol_schiessel/Manish/Spermatogensis/output/exactpoint_unboundpromoter_regions_breath/id_lookup_subset.tsv}"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$out")"

# Process the file
awk -F '\t' '
BEGIN {OFS = FS}
NR == 1 {
    print
    next
}
{
    # Split the first field by "|" to get individual components
    n = split($1, parts, "|")
    if (n >= 5) {
        # Split the 5th part by ":" to separate "pos" from the number
        split(parts[5], pos_parts, ":")
        if (pos_parts[1] == "pos" && pos_parts[2] ~ /^[0-9]+$/) {
            pos = pos_parts[2] + 0  # Convert to number
            # Keep only rows where position is between 2010 and 2030 inclusive
            if (pos >= 2010 && pos <= 2030) {
                print
            }
            next
        }
    }
}' "$in" > "$out"
