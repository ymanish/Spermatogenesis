"""Dinucleotide and DNA-shape diagnostics for 147 bp nucleosome FASTA sets."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


DINUCLEOTIDES = [
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
]

CONTACT_FIRST_PHOSPHATE = [2, 14, 24, 34, 45, 55, 65, 76, 86, 96, 107, 116, 128, 139]

MGW_DINUC_APPROX = {
    "AA": 3.38,
    "AT": 3.26,
    "AC": 4.30,
    "AG": 4.12,
    "TA": 4.54,
    "TT": 3.38,
    "TC": 4.60,
    "TG": 4.62,
    "CA": 4.62,
    "CT": 4.12,
    "CC": 4.82,
    "CG": 4.94,
    "GA": 4.60,
    "GT": 4.30,
    "GC": 5.02,
    "GG": 4.82,
}


def read_fasta(path: str | Path, keep_acgt_only: bool = True) -> dict[str, str]:
    """Read a FASTA file into ``{record_id: sequence}``."""
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = _clean_sequence("".join(chunks), keep_acgt_only)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        records[current_id] = _clean_sequence("".join(chunks), keep_acgt_only)
    return records


def _clean_sequence(seq: str, keep_acgt_only: bool) -> str:
    seq = seq.upper().replace("U", "T")
    if keep_acgt_only:
        seq = "".join(base for base in seq if base in "ACGT")
    return seq


def validate_sequences(records: dict[str, str], expected_len: int = 147) -> list[str]:
    """Return validation warnings for non-ACGT or non-147 bp records."""
    warnings = []
    bad_len = sum(1 for seq in records.values() if len(seq) != expected_len)
    bad_alpha = sum(1 for seq in records.values() if set(seq) - set("ACGT"))
    if bad_len:
        warnings.append(f"{bad_len} records are not {expected_len} bp")
    if bad_alpha:
        warnings.append(f"{bad_alpha} records contain non-ACGT characters")
    return warnings


def load_id_lookup(path: str | Path) -> dict[int, str]:
    """Read ``SPRM_data/<dataset>/id_lookup.tsv``."""
    lookup: dict[int, str] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            lookup[int(row["global_id"])] = row["seq_id"]
    return lookup


def load_energy_state(
    path: str | Path,
    left_open: int = 0,
    right_open: int = 0,
    value_col: str = "dF_total",
) -> dict[int, float]:
    """Load one SPRM energy state keyed by ``global_id``."""
    values: dict[int, float] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row["left_open"]) == left_open and int(row["right_open"]) == right_open:
                values[int(row["global_id"])] = float(row[value_col])
    return values


def load_dataset(
    dataset: str,
    fasta_path: str | Path,
    sprm_root: str | Path,
    left_open: int = 0,
    right_open: int = 0,
) -> list[dict[str, object]]:
    """Join FASTA records to the selected SPRM energy state by sequence id."""
    fasta = read_fasta(fasta_path)
    dataset_dir = Path(sprm_root) / dataset
    id_lookup = load_id_lookup(dataset_dir / "id_lookup.tsv")
    energy_by_gid = load_energy_state(dataset_dir / "energies.tsv", left_open, right_open)

    rows: list[dict[str, object]] = []
    missing_fasta = []
    for global_id, seq_id in id_lookup.items():
        seq = fasta.get(seq_id)
        if seq is None:
            missing_fasta.append(seq_id)
            continue
        rows.append(
            {
                "dataset": dataset,
                "global_id": global_id,
                "seq_id": seq_id,
                "sequence": seq,
                "dF_total": energy_by_gid.get(global_id, np.nan),
            }
        )

    if missing_fasta:
        preview = ", ".join(missing_fasta[:5])
        raise ValueError(
            f"{dataset}: {len(missing_fasta)} ids from id_lookup.tsv were absent from FASTA. "
            f"First missing ids: {preview}"
        )
    return rows


def dinucleotide_frequencies(seq: str) -> dict[str, float]:
    """Compute normalized dinucleotide frequencies for one sequence."""
    seq = seq.upper()
    counts = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
    total = sum(counts.values())
    return {dn: counts.get(dn, 0) / total for dn in DINUCLEOTIDES}


def add_dinucleotide_features(rows: list[dict[str, object]]) -> None:
    """Mutate row dictionaries by adding ``dn_<step>`` frequency columns."""
    for row in rows:
        freqs = dinucleotide_frequencies(str(row["sequence"]))
        for dn, value in freqs.items():
            row[f"dn_{dn}"] = value
        row["gc_fraction"] = gc_fraction(str(row["sequence"]))


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def compare_feature(
    ret_values: Iterable[float],
    ctrl_values: Iterable[float],
) -> dict[str, float]:
    """Compare two feature distributions with Mann-Whitney U and Cohen-like d."""
    ret = np.asarray(list(ret_values), dtype=float)
    ctrl = np.asarray(list(ctrl_values), dtype=float)
    ret = ret[np.isfinite(ret)]
    ctrl = ctrl[np.isfinite(ctrl)]
    p_value = mann_whitney_pvalue(ret, ctrl)
    diff = float(np.mean(ret) - np.mean(ctrl))
    pooled = math.sqrt((float(np.var(ret)) + float(np.var(ctrl))) / 2)
    effect = diff / pooled if pooled else np.nan
    return {
        "ret_mean": float(np.mean(ret)),
        "ret_std": float(np.std(ret)),
        "ctrl_mean": float(np.mean(ctrl)),
        "ctrl_std": float(np.std(ctrl)),
        "diff": diff,
        "p_value": float(p_value),
        "effect_size": float(effect),
        "n_ret": int(ret.size),
        "n_ctrl": int(ctrl.size),
    }


def mann_whitney_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Return a two-sided Mann-Whitney U p-value, preferring SciPy if present."""
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return _mann_whitney_normal_approx(x, y)


def _mann_whitney_normal_approx(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-corrected normal approximation fallback for Mann-Whitney U."""
    values = np.concatenate([x, y])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    n1 = len(x)
    n2 = len(y)
    rank_x = np.sum(ranks[:n1])
    u1 = rank_x - n1 * (n1 + 1) / 2
    mean_u = n1 * n2 / 2
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts)
    sd_u = math.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie_term / ((n1 + n2) * (n1 + n2 - 1))))
    if sd_u == 0:
        return 1.0
    z = (u1 - mean_u) / sd_u
    return math.erfc(abs(z) / math.sqrt(2))


def compare_dinucleotides(
    ret_rows: list[dict[str, object]],
    ctrl_rows: list[dict[str, object]],
    ctrl_label: str,
) -> list[dict[str, object]]:
    """Return one comparison row per dinucleotide for RET vs a control."""
    results = []
    for dn in DINUCLEOTIDES:
        stats = compare_feature(
            [float(row[f"dn_{dn}"]) for row in ret_rows],
            [float(row[f"dn_{dn}"]) for row in ctrl_rows],
        )
        results.append({"comparison": f"RET_vs_{ctrl_label}", "feature": dn, **stats})
    return sorted(results, key=lambda row: abs(float(row["diff"])), reverse=True)


def get_site_ranges(n_sites: int = 14) -> list[tuple[int, int]]:
    ranges = []
    for j in range(n_sites):
        start = 0 if j == 0 else CONTACT_FIRST_PHOSPHATE[j]
        end = 147 if j == n_sites - 1 else CONTACT_FIRST_PHOSPHATE[j + 1]
        ranges.append((start, end))
    return ranges


def positional_dinucleotide_density(seq: str, target_dn: str, n_sites: int = 14) -> list[int]:
    """Count a target dinucleotide per phosphate-defined nucleosome site."""
    counts = []
    for start, end in get_site_ranges(n_sites):
        count = sum(1 for i in range(start, min(end, len(seq) - 1)) if seq[i : i + 2] == target_dn)
        counts.append(count)
    return counts


def positional_profile(rows: list[dict[str, object]], target_dn: str) -> tuple[np.ndarray, np.ndarray]:
    profiles = np.asarray(
        [positional_dinucleotide_density(str(row["sequence"]), target_dn) for row in rows],
        dtype=float,
    )
    return np.mean(profiles, axis=0), sem(profiles, axis=0)


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(values), axis=axis))


def load_shape_table(filepath: str | Path) -> dict[str, float]:
    """Load a two-column pentamer-to-shape lookup table."""
    table: dict[str, float] = {}
    with Path(filepath).open() as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) != 2 or parts[0].lower() in {"pentamer", "kmer", "seq"}:
                continue
            try:
                table[parts[0].upper()] = float(parts[1])
            except ValueError:
                continue
    return table


def predict_shape(seq: str, shape_table: dict[str, float], feature_type: str = "nucleotide") -> list[float]:
    """Predict a pentamer-table DNA shape feature along a sequence."""
    seq = seq.upper()
    values: list[float] = []
    if feature_type == "nucleotide":
        for i in range(len(seq)):
            if i < 2 or i >= len(seq) - 2:
                values.append(np.nan)
            else:
                values.append(shape_table.get(seq[i - 2 : i + 3], np.nan))
    elif feature_type == "step":
        for i in range(len(seq) - 1):
            if i < 2 or i >= len(seq) - 3:
                values.append(np.nan)
            else:
                values.append(shape_table.get(seq[i - 1 : i + 4], np.nan))
    else:
        raise ValueError("feature_type must be 'nucleotide' or 'step'")
    return values


def predict_mgw_simple(seq: str) -> list[float]:
    """Approximate MGW at dinucleotide steps from dinucleotide-level averages."""
    seq = seq.upper()
    return [MGW_DINUC_APPROX.get(seq[i : i + 2], np.nan) for i in range(len(seq) - 1)]


def shape_per_site(shape_values: Iterable[float], n_sites: int = 14) -> list[float]:
    values = list(shape_values)
    means = []
    for start, end in get_site_ranges(n_sites):
        site_values = np.asarray(values[start:min(end, len(values))], dtype=float)
        means.append(float(np.nanmean(site_values)) if np.isfinite(site_values).any() else np.nan)
    return means


def add_shape_features(
    rows: list[dict[str, object]],
    shape_tables: dict[str, tuple[dict[str, float], str]] | None = None,
    include_simple_mgw: bool = True,
) -> list[str]:
    """Add per-sequence mean and site shape features. Returns added feature names."""
    added: list[str] = []
    shape_tables = shape_tables or {}
    if include_simple_mgw:
        for row in rows:
            values = predict_mgw_simple(str(row["sequence"]))
            row["shape_MGW_simple_mean"] = float(np.nanmean(values))
            for idx, value in enumerate(shape_per_site(values)):
                row[f"shape_MGW_simple_site_{idx:02d}"] = value
        added.append("MGW_simple")

    for name, (table, feature_type) in shape_tables.items():
        feature = f"shape_{name}"
        for row in rows:
            values = predict_shape(str(row["sequence"]), table, feature_type)
            row[f"{feature}_mean"] = float(np.nanmean(values))
            for idx, value in enumerate(shape_per_site(values)):
                row[f"{feature}_site_{idx:02d}"] = value
        added.append(name)
    return added


def compare_shape_feature(
    ret_rows: list[dict[str, object]],
    ctrl_rows: list[dict[str, object]],
    ctrl_label: str,
    feature_name: str,
) -> dict[str, object]:
    column = f"shape_{feature_name}_mean"
    stats = compare_feature([float(row[column]) for row in ret_rows], [float(row[column]) for row in ctrl_rows])
    return {"comparison": f"RET_vs_{ctrl_label}", "feature": feature_name, **stats}


def write_table(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write a list of dictionaries to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def significant_features(rows: list[dict[str, object]], alpha: float = 0.003) -> list[str]:
    return sorted({str(row["feature"]) for row in rows if float(row["p_value"]) < alpha})


def plot_dinucleotide_barplot(comparison_rows: list[dict[str, object]], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    features = DINUCLEOTIDES
    comparisons = sorted({str(row["comparison"]) for row in comparison_rows})
    by_key = {(row["comparison"], row["feature"]): row for row in comparison_rows}
    score = {
        dn: max(abs(float(by_key[(comp, dn)]["diff"])) for comp in comparisons if (comp, dn) in by_key)
        for dn in features
    }
    features = sorted(features, key=lambda dn: score[dn], reverse=True)

    x = np.arange(len(features))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5))
    ret_means = [float(by_key[(comparisons[0], dn)]["ret_mean"]) for dn in features]
    ret_stds = [float(by_key[(comparisons[0], dn)]["ret_std"]) for dn in features]
    ax.bar(x - width, ret_means, width, yerr=ret_stds, capsize=2, label="RET", color="#b33a3a")

    colors = ["#3d8b55", "#3b6fb6", "#8b6bb1"]
    for idx, comp in enumerate(comparisons):
        means = [float(by_key[(comp, dn)]["ctrl_mean"]) for dn in features]
        stds = [float(by_key[(comp, dn)]["ctrl_std"]) for dn in features]
        label = comp.replace("RET_vs_", "")
        ax.bar(x + idx * width, means, width, yerr=stds, capsize=2, label=label, color=colors[idx % len(colors)])
        for xpos, dn in zip(x + idx * width, features):
            row = by_key[(comp, dn)]
            if float(row["p_value"]) < 0.003:
                ax.text(xpos, float(row["ctrl_mean"]) + float(row["ctrl_std"]) + 0.004, "*", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel("Frequency per sequence")
    ax.set_title("Dinucleotide frequencies")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_positional_dinucleotides(
    datasets: dict[str, list[dict[str, object]]],
    features: list[str],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if not features:
        return
    ncols = 2
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    x = np.arange(14)
    colors = {"RET": "#b33a3a", "ctrl02": "#3d8b55", "ctrl03": "#3b6fb6"}
    for ax, dn in zip(axes.ravel(), features):
        for label, rows in datasets.items():
            mean, err = positional_profile(rows, dn)
            ax.errorbar(x, mean, yerr=err, marker="o", linewidth=1.5, capsize=2, label=label, color=colors.get(label))
        ax.set_title(dn)
        ax.set_xlabel("Site")
        ax.set_ylabel("Mean count")
        ax.set_xticks(x)
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_shape_distributions(
    datasets: dict[str, list[dict[str, object]]],
    features: list[str],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if not features:
        return
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), squeeze=False)
    colors = {"RET": "#b33a3a", "ctrl02": "#3d8b55", "ctrl03": "#3b6fb6"}
    for ax, feature in zip(axes.ravel(), features):
        data = []
        labels = []
        for label, rows in datasets.items():
            data.append([float(row[f"shape_{feature}_mean"]) for row in rows])
            labels.append(label)
        box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(colors.get(label, "#777777"))
            patch.set_alpha(0.6)
        ax.set_title(feature)
        ax.set_ylabel("Per-sequence mean")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_shape_positional_profiles(
    datasets: dict[str, list[dict[str, object]]],
    features: list[str],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if not features:
        return
    fig, axes = plt.subplots(1, len(features), figsize=(5.5 * len(features), 4), squeeze=False)
    x = np.arange(14)
    colors = {"RET": "#b33a3a", "ctrl02": "#3d8b55", "ctrl03": "#3b6fb6"}
    for ax, feature in zip(axes.ravel(), features):
        for label, rows in datasets.items():
            matrix = np.asarray(
                [[float(row[f"shape_{feature}_site_{idx:02d}"]) for idx in range(14)] for row in rows],
                dtype=float,
            )
            ax.errorbar(
                x,
                np.nanmean(matrix, axis=0),
                yerr=sem(matrix, axis=0),
                marker="o",
                linewidth=1.5,
                capsize=2,
                label=label,
                color=colors.get(label),
            )
        ax.set_title(feature)
        ax.set_xlabel("Site")
        ax.set_ylabel("Mean shape value")
        ax.set_xticks(x)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_energy_feature_scatter(
    datasets: dict[str, list[dict[str, object]]],
    feature_column: str,
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"RET": "#b33a3a", "ctrl02": "#3d8b55", "ctrl03": "#3b6fb6"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, rows in datasets.items():
        x = [float(row["dF_total"]) for row in rows]
        y = [float(row[feature_column]) for row in rows]
        ax.scatter(x, y, s=10, alpha=0.35, label=label, color=colors.get(label))
    ax.set_xlabel("SPRM dF_total, left_open=0/right_open=0")
    ax.set_ylabel(feature_column)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
