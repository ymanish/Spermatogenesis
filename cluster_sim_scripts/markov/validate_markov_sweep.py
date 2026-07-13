#!/usr/bin/env python3
"""Validation report + figures for the Markov MFPT/survival sweep.

Physics sanity checks and diagnostic figures for a completed Markov sweep
(cluster_sim_scripts/markov/). Loads every cell robustly via parameters.json
(so it is immune to the p{conc:.1f} folder-name collision that maps 0.0 and 0.01
to the same directory prefix), then:

  * prints a per-cell table (flag counts, median MFPT, reachable counts),
  * checks that median MFPT decreases with protamine concentration,
  * writes figures to <root>/_validation/.

Usage:
    python cluster_sim_scripts/markov/validate_markov_sweep.py \
        --root output/markov_sweep_stable147_v1 \
        [--cap 5000] [--survival-dataset ret_all_stable147_refined]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ret_all first (the retained set), then controls; stable colors.
DATASET_ORDER = [
    "ret_all_stable147_refined",
    "ctrl01_random_genome_safe_stable147_refined",
    "ctrl02_random_genome_gcmatched_stable147_refined",
    "ctrl03_som_gcmatched_stable147_refined",
    "ctrl04_bound_prom_evicted_stable147_refined",
    "ctrl05_unbound_prom_yazdi_stable147_refined",
]
def short(ds): return ds.replace("_stable147_refined", "")


def index_cells(root, dataset):
    """(conc, coop) -> (summary_tsv, cell_dir) via parameters.json."""
    cells = {}
    for pj in glob.glob(f"{root}/{dataset}/*/parameters.json"):
        p = json.load(open(pj)); pp = p["prot_params"]
        cdir = os.path.dirname(pj)
        summ = os.path.join(cdir, "summaries", f"{dataset}.tsv")
        if os.path.exists(summ):
            cells[(round(pp["p_conc"], 6), round(pp["cooperativity"], 6))] = (summ, cdir)
    return cells


def build_table(root, datasets, cap):
    rows = []
    for ds in datasets:
        for (conc, coop), (summ, _) in sorted(index_cells(root, ds).items()):
            df = pd.read_csv(summ, sep="\t")
            m = df["mfpt"]
            finite = (df["mfpt_flag"] == "ok") & np.isfinite(m)
            rows.append(dict(
                dataset=ds, conc=conc, coop=coop, n=len(df),
                n_censored=int((~finite).sum()),
                cens_frac=float((~finite).mean()),
                median=float(m[finite].median()),
                reach=int((m < cap).sum()),
                reach_frac=float((m < cap).mean()),
            ))
    return pd.DataFrame(rows)


def fig_mfpt_vs_conc(R, datasets, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, coop in zip(axes, [0.0, 4.5]):
        for ds in datasets:
            sub = R[(R.dataset == ds) & (R.coop == coop)].sort_values("conc")
            if sub.empty: continue
            # conc=0 -> plot at a small positive x for the log axis
            x = sub.conc.replace(0.0, 3e-3).values
            ax.loglog(x, sub["median"], "o-", label=short(ds), ms=4)
        ax.set_title(f"cooperativity = {coop}")
        ax.set_xlabel("protamine concentration (µM)  [0 shown at 3e-3]")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("median MFPT  (dimensionless τ, finite set)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Median first-passage time vs protamine concentration")
    fig.text(0.5, -0.02, "Note: at conc≲0.1 the finite-set median is biased upward by "
             "censoring (fewer underflowed nucleosomes at higher conc). "
             "See reachable-fraction figure for a censoring-free view.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_mfpt_vs_conc.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig_reachable_vs_conc(R, datasets, cap, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, coop in zip(axes, [0.0, 4.5]):
        for ds in datasets:
            sub = R[(R.dataset == ds) & (R.coop == coop)].sort_values("conc")
            if sub.empty: continue
            x = sub.conc.replace(0.0, 3e-3).values
            ax.semilogx(x, sub["reach_frac"] * 100, "o-", label=short(ds), ms=4)
        ax.set_title(f"cooperativity = {coop}")
        ax.set_xlabel("protamine concentration (µM)  [0 shown at 3e-3]")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(f"% nucleosomes reachable (MFPT < {cap:g} τ)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Reachable fraction vs concentration  (sampling cap = {cap:g} τ)")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_reachable_vs_conc.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig_survival(root, dataset, outdir):
    cells = index_cells(root, dataset)
    concs = [0.0, 1.0, 10.0, 100.0, 1000.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(concs)))
    for c, col in zip(concs, cmap):
        key = (round(c, 6), 4.5) if c > 0 else (0.0, 0.0)
        if key not in cells:
            continue
        _, cdir = cells[key]
        pq = os.path.join(cdir, "survivals", f"{dataset}.parquet")
        df = pd.read_parquet(pq, columns=["tau_grid", "survival"])
        tau = np.asarray(df["tau_grid"].iloc[0])
        S = np.vstack(df["survival"].values).mean(axis=0)
        ax.semilogx(tau[1:], S[1:], color=col, label=f"conc={c:g}")
    ax.set_xlabel("τ  (dimensionless time)")
    ax.set_ylabel("ensemble survival  ⟨S(τ)⟩")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Ensemble survival — {short(dataset)}  (cooperativity 4.5)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_survival_ensemble.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig_censored(R, datasets, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in datasets:
        sub = R[(R.dataset == ds) & (R.coop == 0.0)].sort_values("conc")
        if sub.empty: continue
        x = sub.conc.replace(0.0, 3e-3).values
        ax.semilogx(x, sub["cens_frac"] * 100, "o-", label=short(ds), ms=4)
    ax.set_xlabel("protamine concentration (µM)  [0 shown at 3e-3]")
    ax.set_ylabel("% censored (underflowed / inf MFPT)")
    ax.set_title("Censored fraction vs concentration  (cooperativity 0)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_censored_fraction.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cap", type=float, default=5000.0)
    ap.add_argument("--survival-dataset", default="ret_all_stable147_refined")
    args = ap.parse_args()

    present = [d for d in DATASET_ORDER if os.path.isdir(os.path.join(args.root, d))]
    outdir = os.path.join(args.root, "_validation")
    os.makedirs(outdir, exist_ok=True)

    R = build_table(args.root, present, args.cap)
    R.to_csv(os.path.join(outdir, "cell_summary.tsv"), sep="\t", index=False)

    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\nMonotonicity (median MFPT should fall as conc rises; conc=0→0.01 "
          "wobble is a censoring artifact — see report):")
    for ds in present:
        for coop in [0.0, 4.5]:
            sub = R[(R.dataset == ds) & (R.coop == coop)].sort_values("conc")
            med = sub["median"].values
            if len(med) < 2: continue
            viol = int(np.sum(np.diff(med) > 0))
            print(f"  {short(ds):28s} coop={coop:<4g} "
                  f"{'OK' if viol == 0 else f'{viol} up-step(s)'}  "
                  f"({med[0]:.2g} → {med[-1]:.2g})")

    figs = [
        fig_mfpt_vs_conc(R, present, outdir),
        fig_reachable_vs_conc(R, present, args.cap, outdir),
        fig_survival(args.root, args.survival_dataset, outdir),
        fig_censored(R, present, outdir),
    ]
    print("\nWrote:")
    for f in figs + [os.path.join(outdir, "cell_summary.tsv")]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
