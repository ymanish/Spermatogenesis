#!/usr/bin/env python3
"""
ADHOC SCRIPT. Not part of any pipeline, not imported by anything, and not
maintained. It exists to re-derive one column of an existing dataset in place of
re-running the free-energy calculation, and it should be deleted once the
two-energy adsorption model is folded into
``HAMNucRetSeq_pipeline/src/workflows/fepostprocess/som_fe_postprocess.py``.

What it does
------------
``SPRM_data/*_stable147_refined/energies.tsv`` carries one column of energy,
``dF_total``, built by ``som_fe_postprocess.py`` as

    dF_total = dF - E_homo * n_bound

with a single homogeneous adsorption energy ``E_homo = 16.32`` kT per binding
site on every one of the ``n_bound = 14 - left_open - right_open`` sites a state
still holds. This script removes that term and puts a two-energy model in its
place:

    dF          = dF_total + E_homo * n_bound            (recovered intrinsic)
    dF_total'   = dF - E_out_site * n_out - E_in_site * n_in

where ``n_out`` counts the bound sites in the two outermost positions on each
end and ``n_in`` counts the rest.

Units, and why the energies are doubled
---------------------------------------
The SAXS fit that produced ``E_out`` and ``E_in`` is quoted per phosphate
contact, and a binding site covers two contacts, so a per-contact energy becomes
a per-site energy by doubling. The fit's ``K = 4`` outer contacts per side
likewise become ``2`` outer binding sites per side.

    crystal/md fit, per contact : E_out = 6.70, E_in >= 14.00 (soft lower bound)
    used here,      per site    : E_out = 13.40, E_in = 2 * (value passed in)

The homogeneous 16.32 kT per site the data already carries is the same fit's
one-energy solution, 8.16 kT per contact, doubled. Passing
``--e-out 8.16 --e-in 8.16`` therefore has to reproduce ``dF_total`` exactly,
and ``--verify`` checks that on real rows before writing anything.

Output
------
One directory per (input directory, E_in) pair:

    SPRM_data/<name>_Eout_<e_out>_Ein<e_in>/
        energies.tsv     same four columns, dF_total replaced
        id_lookup.tsv    copied unchanged from the source
        provenance.txt   parameters and formula used

Usage
-----
    python adhoc_two_energy_adsorption.py --e-in 10.5 11.0
    python adhoc_two_energy_adsorption.py --e-in 10.5 11.0 --dry-run
    python adhoc_two_energy_adsorption.py --e-in 10.5 --only ctrl02 ret_all
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "SPRM_data"
PATTERN = "*_stable147_refined"

N_SITES = 14        # binding sites per nucleosome, so left_open + right_open <= 13
K_SITES = 2         # outermost binding sites per side (= K = 4 contacts per side)
E_HOMO_SITE = 16.32  # kT per site, the term already in the data
COLS = ["global_id", "left_open", "right_open", "dF_total"]


# ---------------------------------------------------------------- reading
def scan_energies(path: Path) -> pl.LazyFrame:
    """
    Lazily read one energies.tsv, tolerating whitespace-padded fields.

    Five of the six files are plain polars CSV output. The sixth (ctrl02) is
    padded, header included, so every field is read as a string and stripped
    before casting rather than trusting type inference.
    """
    lf = pl.scan_csv(path, separator="\t", infer_schema_length=0)
    raw = lf.collect_schema().names()
    stripped = [c.strip() for c in raw]
    if stripped != COLS:
        raise ValueError("%s has columns %r, expected %r" % (path, stripped, COLS))
    return (
        lf.rename(dict(zip(raw, stripped)))
        .with_columns([pl.col(c).str.strip_chars() for c in COLS])
        .with_columns([
            pl.col("global_id").cast(pl.Int64),
            pl.col("left_open").cast(pl.Int8),
            pl.col("right_open").cast(pl.Int8),
            pl.col("dF_total").cast(pl.Float64),
        ])
    )


# ---------------------------------------------------------------- the model
def outer_bound_expr(n: int = N_SITES, k: int = K_SITES) -> pl.Expr:
    """
    Bound sites lying in the two outer groups, as a polars expression.

    A state holds sites ``left_open .. n - right_open - 1``. The outer groups
    are ``[0, k)`` and ``[n - k, n)``. Same counting as the SAXS fit's
    ``outer_bound_counts``, written over binding sites rather than contacts.
    """
    stop = pl.lit(n, dtype=pl.Int32) - pl.col("right_open").cast(pl.Int32)
    left = pl.col("left_open").cast(pl.Int32)
    lo = (pl.min_horizontal(stop, pl.lit(k, dtype=pl.Int32)) - left).clip(lower_bound=0)
    hi = (stop - pl.max_horizontal(left, pl.lit(n - k, dtype=pl.Int32))).clip(lower_bound=0)
    return (lo + hi).alias("n_out")


def retotal(lf: pl.LazyFrame, e_out_site: float, e_in_site: float,
            e_homo_site: float = E_HOMO_SITE, n: int = N_SITES) -> pl.LazyFrame:
    """Replace dF_total with the two-energy version, keeping the column order."""
    n_bound = (pl.lit(n, dtype=pl.Int32)
               - pl.col("left_open").cast(pl.Int32)
               - pl.col("right_open").cast(pl.Int32))
    return (
        lf
        .with_columns([n_bound.alias("n_bound"), outer_bound_expr(n)])
        .with_columns(
            (pl.col("dF_total") + pl.lit(e_homo_site) * pl.col("n_bound")).alias("dF")
        )
        .with_columns(
            (pl.col("dF")
             - pl.lit(e_out_site) * pl.col("n_out")
             - pl.lit(e_in_site) * (pl.col("n_bound") - pl.col("n_out"))
             ).cast(pl.Float32).alias("dF_total")
        )
        .select(COLS)
    )


# ---------------------------------------------------------------- checks
def verify(path: Path, e_homo: float = E_HOMO_SITE, n_rows: int = 200_000) -> None:
    """
    Round-trip the homogeneous model through the two-energy code path.

    Setting both energies to the value already in the data must give back
    dF_total unchanged. A failure here means the removal step, the site
    bookkeeping or the file format is wrong, and nothing should be written.
    """
    lf = scan_energies(path).head(n_rows)
    got = retotal(lf, e_homo, e_homo, e_homo).collect()
    want = lf.collect()

    d = (got["dF_total"].cast(pl.Float64) - want["dF_total"]).abs().max()
    # The stored column is Float32, so a value near 235 kT already carries
    # ~1.4e-5 of representation error; the round trip must not add more.
    assert d < 1e-3, "round trip changed dF_total by up to %.3g kT" % d

    # And the site bookkeeping: every state must have 1..14 bound sites, of
    # which at most 2*K sit in the outer groups, and a fully wrapped state must
    # have exactly 2*K.
    aux = (scan_energies(path).head(n_rows)
           .with_columns([(pl.lit(N_SITES) - pl.col("left_open").cast(pl.Int32)
                           - pl.col("right_open").cast(pl.Int32)).alias("n_bound"),
                          outer_bound_expr()])
           .collect())
    assert aux["n_bound"].min() >= 1 and aux["n_bound"].max() <= N_SITES
    assert aux["n_out"].min() >= 0 and aux["n_out"].max() <= 2 * K_SITES
    full = aux.filter((pl.col("left_open") == 0) & (pl.col("right_open") == 0))
    assert len(full) and (full["n_out"] == 2 * K_SITES).all()
    assert (aux["n_out"] <= aux["n_bound"]).all()
    print("  verify   : %s rows round-trip to within %.2e kT" % (f"{len(want):,}", d))


def report(path: Path, e_out_site: float, e_in_site: float,
           e_homo: float = E_HOMO_SITE) -> None:
    """Print what the swap did to the fully wrapped state, as a sanity read."""
    lf = scan_energies(path).filter(
        (pl.col("left_open") == 0) & (pl.col("right_open") == 0))
    before = lf.select(pl.col("dF_total")).collect()["dF_total"]
    after = retotal(lf, e_out_site, e_in_site, e_homo).collect()["dF_total"]
    old_ads = e_homo * N_SITES
    new_ads = e_out_site * 2 * K_SITES + e_in_site * (N_SITES - 2 * K_SITES)
    print("  full wrap: adsorption %.2f -> %.2f kT,  dF_total mean %.2f -> %.2f kT"
          % (old_ads, new_ads, before.mean(), after.mean()))


# ---------------------------------------------------------------- driver
def convert(src: Path, dst: Path, e_out_site: float, e_in_site: float,
            e_out_contact: float, e_in_contact: float, e_homo: float,
            overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError("%s exists; pass --overwrite to replace it" % dst)
    dst.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    out = dst / "energies.tsv"
    retotal(scan_energies(src / "energies.tsv"), e_out_site, e_in_site,
            e_homo).sink_csv(out, separator="\t")

    lookup = src / "id_lookup.tsv"
    if lookup.exists():
        shutil.copy2(lookup, dst / "id_lookup.tsv")

    (dst / "provenance.txt").write_text(
        "written by adhoc_two_energy_adsorption.py\n"
        "source            : %s\n"
        "binding sites      : %d, outer %d per side\n"
        "removed            : homogeneous %.4f kT per site on every bound site\n"
        "added, per site    : E_out %.4f, E_in %.4f\n"
        "added, per contact : E_out %.4f, E_in %.4f  (a site covers two contacts)\n"
        "fully wrapped ads  : %.4f kT  (was %.4f)\n"
        "formula            : dF_total' = (dF_total + %.4f * n_bound)\n"
        "                                 - %.4f * n_out - %.4f * n_in\n"
        % (src, N_SITES, K_SITES, e_homo, e_out_site, e_in_site,
           e_out_contact, e_in_contact,
           e_out_site * 2 * K_SITES + e_in_site * (N_SITES - 2 * K_SITES),
           e_homo * N_SITES, e_homo, e_out_site, e_in_site))

    print("  written  : %s  (%.1f MB, %.1f s)"
          % (out, out.stat().st_size / 1e6, time.perf_counter() - t0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("What it does")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="directory holding the *_stable147_refined subdirectories")
    p.add_argument("--pattern", default=PATTERN,
                   help="glob for the source directories (default %(default)s)")
    p.add_argument("--only", nargs="+", metavar="SUBSTR",
                   help="restrict to source directories containing any of these")
    p.add_argument("--e-out", type=float, default=6.70, metavar="kT",
                   help="outer adsorption energy PER CONTACT (default %(default)s)")
    p.add_argument("--e-in", type=float, nargs="+", required=True, metavar="kT",
                   help="inner adsorption energies PER CONTACT, one output set each")
    p.add_argument("--homo", type=float, default=E_HOMO_SITE, metavar="kT",
                   help="homogeneous energy PER SITE to remove (default %(default)s)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="verify and report, write nothing")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the round-trip check (not recommended)")
    a = p.parse_args(argv)

    srcs = sorted(d for d in a.data_dir.glob(a.pattern) if (d / "energies.tsv").is_file())
    if a.only:
        srcs = [d for d in srcs if any(s in d.name for s in a.only)]
    if not srcs:
        print("no source directory matched %s/%s" % (a.data_dir, a.pattern),
              file=sys.stderr)
        return 1

    e_out_site = 2.0 * a.e_out
    print("source dirs   : %d under %s" % (len(srcs), a.data_dir))
    print("removing      : %.4f kT per site, all %d sites when fully wrapped"
          % (a.homo, N_SITES))
    print("E_out         : %.4f kT per contact = %.4f per site, %d outer sites per side"
          % (a.e_out, e_out_site, K_SITES))
    print("E_in          : %s kT per contact"
          % ", ".join("%.4f" % e for e in a.e_in))
    print("mode          : %s" % ("dry run, nothing written" if a.dry_run else "writing"))

    if not a.skip_verify:
        print("\nround-trip check on %s" % srcs[0].name)
        verify(srcs[0] / "energies.tsv", a.homo)

    t0 = time.perf_counter()
    for e_in_contact in a.e_in:
        e_in_site = 2.0 * e_in_contact
        suffix = "_Eout_%.2f_Ein%.2f" % (a.e_out, e_in_contact)
        print("\n=== E_in %.2f kT per contact (%.2f per site) -> *%s"
              % (e_in_contact, e_in_site, suffix))
        for src in srcs:
            dst = a.data_dir / (src.name + suffix)
            print(" %s" % src.name)
            report(src / "energies.tsv", e_out_site, e_in_site, a.homo)
            if a.dry_run:
                print("  would write: %s" % dst)
                continue
            convert(src, dst, e_out_site, e_in_site, a.e_out, e_in_contact,
                    a.homo, a.overwrite)

    print("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
