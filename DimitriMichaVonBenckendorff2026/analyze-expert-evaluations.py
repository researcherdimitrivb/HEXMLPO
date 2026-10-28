#!/usr/bin/env python3
"""
analyse_evaluation.py
---------------------
Parse the anonymised expert-evaluation PDF, build tidy data frames, compute
summary statistics and produce the thesis figures.

Usage:  python analyse_evaluation.py <evaluation.pdf> [output_dir]
"""
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------
EXPERTS = ["XAI1", "XAI2", "ONT1", "ONT2", "ONT3", "HC1"]   # order in the PDF
GROUP = {"XAI": "XAI expert", "ONT": "Ontology engineer", "HC": "Healthcare expert"}
EXPECTED_SPECIFIC = {"XAI": 12, "ONT": 13, "HC": 11}         # role-specific criteria
EXPECTED_OVERALL = 6                                         # common to all experts

FIG_WIDTH = 6.3   # inches = text width of the thesis (set to e.g. 4.8 for an LNCS single column)

# Colour-blind-safe (Okabe-Ito) palette; XAI = blues, ontology = warm, healthcare = green.
# Marker shapes differ per expert so that the figures stay readable in greyscale.
STYLE = {
    "XAI1": ("o", "#0072B2"), "XAI2": ("s", "#56B4E9"),
    "ONT1": ("^", "#D55E00"), "ONT2": ("P", "#CC79A7"), "ONT3": ("v", "#E69F00"),
    "HC1": ("D", "#009E73"),
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


# --------------------------------------------------------------------------
# 1. Load the PDF and transform the tables into tidy data frames
# --------------------------------------------------------------------------
def _clean(cell):
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def _to_float(txt):
    try:
        return float(txt)
    except ValueError:        # '-', '/', '' -> not rated
        return np.nan


def parse_pdf(pdf_path):
    """Return (ratings, issues) as tidy DataFrames.

    The PDF holds, per expert, (i) a role-specific rating table, (ii) a common
    'overall assessment' table and (iii) an issue list. Tables that continue
    across a page break have no header row, so the parser keeps state
    (current expert / section / mode) while iterating over all tables in order.
    """
    ratings, issues = [], []
    block, mode, expert, section = 0, None, None, None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    cells = [_clean(c) for c in row]
                    head = cells[0]
                    if head == "Criterion":            # new rating table
                        block += 1
                        expert = EXPERTS[(block - 1) // 2]
                        section = "specific" if block % 2 == 1 else "overall"
                        mode = "ratings"
                    elif head == "ID":                 # issue list header
                        mode = "issues"
                    elif mode == "ratings":
                        crit, stmt, val = cells[:3]
                        ratings.append(dict(expert=expert, section=section,
                                            criterion=crit, statement=stmt,
                                            rating=_to_float(val)))
                    elif mode == "issues":
                        if head == "":                 # wrapped cell of previous row
                            for i, c in enumerate(cells):
                                if i and c:
                                    issues[-1]["cells"][i] += " " + c
                        else:
                            issues.append(dict(expert=expert, cells=cells))

    ratings = pd.DataFrame(ratings)
    ratings["group"] = ratings.expert.str.extract(r"^([A-Z]+)")[0].map(GROUP)

    fix = lambda t: re.sub(r"(?<=[a-z])- (?=[a-z])", "", t)      # un-hyphenate line wraps
    issues = pd.DataFrame(
        [dict(expert=it["expert"], id=it["cells"][0], description=fix(it["cells"][1]),
              modification=fix(it["cells"][2])) for it in issues])
    return ratings, issues


EXPECTED_ISSUES = {"XAI1": 3, "XAI2": 5, "ONT1": 6, "ONT2": 23, "ONT3": 6, "HC1": 3}


def sanity_check(ratings, issues):
    for e in EXPERTS:
        fam = re.match(r"[A-Z]+", e).group()
        n_spec = ((ratings.expert == e) & (ratings.section == "specific")).sum()
        n_over = ((ratings.expert == e) & (ratings.section == "overall")).sum()
        assert n_spec == EXPECTED_SPECIFIC[fam], (e, n_spec)
        assert n_over == EXPECTED_OVERALL, (e, n_over)
        assert (issues.expert == e).sum() == EXPECTED_ISSUES[e], (e, (issues.expert == e).sum())


# --------------------------------------------------------------------------
# 2. Summary statistics
# --------------------------------------------------------------------------
def criterion_stats(df):
    g = df.groupby(["section", "criterion"], sort=False).rating
    out = g.agg(n="count", mean="mean", sd="std", min="min", max="max").reset_index()
    out["range"] = out["max"] - out["min"]
    return out


def agreement(df, a, b):
    piv = df.pivot_table(index=["section", "criterion"], columns="expert", values="rating")
    pair = piv[[a, b]].dropna()
    d = (pair[a] - pair[b]).abs()
    return dict(pair=f"{a} vs {b}", n_items=len(pair), mean_abs_diff=d.mean(),
                exact=(d == 0).mean(), within_1=(d <= 1).mean(), max_diff=d.max(),
                largest=[f"{i[1]} ({pair.loc[i, a]:g} vs {pair.loc[i, b]:g})"
                         for i in d[d == d.max()].index])


def print_summary(ratings, issues):
    pd.set_option("display.width", 160, "display.max_rows", 200)
    r = ratings
    print("\n=== Response completeness (non-missing ratings per expert) ===")
    comp = r.groupby("expert").rating.agg(rated="count", total="size")
    comp["missing"] = comp.total - comp.rated
    print(comp.loc[EXPERTS])

    print("\n=== Mean rating per expert (all rated items / overall items only) ===")
    m_all = r.groupby("expert").rating.mean()
    m_ov = r[r.section == "overall"].groupby("expert").rating.mean()
    fav = r.groupby("expert").rating.apply(lambda s: (s.dropna() >= 4).mean())
    print(pd.DataFrame({"mean_all": m_all, "mean_overall": m_ov, "share>=4": fav}).loc[EXPERTS].round(2))

    st = criterion_stats(r)
    print("\n=== Overall assessment: per criterion ===")
    print(st[st.section == "overall"].round(2).to_string(index=False))

    ov = r[r.section == "overall"].rating.dropna()
    print(f"\nOverall-assessment pooled: n={len(ov)}, mean={ov.mean():.2f}, sd={ov.std():.2f}, "
          f"share>=4={np.mean(ov >= 4):.0%}, share==5={np.mean(ov == 5):.0%}, min={ov.min()}")

    for fam in ["XAI", "ONT", "HC"]:
        sub = st[(st.section == "specific") & st.criterion.isin(
            r[(r.section == "specific") & r.expert.str.startswith(fam)].criterion)]
        # Restrict to the family's own criteria (names such as 'Completeness' recur across families)
        fam_rows = r[(r.section == "specific") & r.expert.str.startswith(fam)]
        s = fam_rows.groupby("criterion", sort=False).rating.agg(n="count", mean="mean",
                                                                  min="min", max="max").dropna(subset=["mean"])
        print(f"\n=== Role-specific criteria: {fam} (sorted by mean) ===")
        print(s.sort_values("mean").round(2))
        print(f"{fam} pooled mean = {fam_rows.rating.mean():.2f} over {fam_rows.rating.count()} ratings")

    print("\n=== Inter-rater agreement (same instrument) ===")
    for a, b in [("XAI1", "XAI2"), ("ONT1", "ONT2"), ("ONT1", "ONT3"), ("ONT2", "ONT3")]:
        ag = agreement(r, a, b)
        print({k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in ag.items()})
    piv = r.pivot_table(index=["section", "criterion"], columns="expert", values="rating")
    ont = piv[["ONT1", "ONT2", "ONT3"]].dropna(thresh=2)
    spread = ont.max(axis=1) - ont.min(axis=1)
    print("ONT items with >=2 raters:", len(ont), "| spread>=2:", int((spread >= 2).sum()))
    print(spread[spread >= 2].to_string())
    xai = piv[["XAI1", "XAI2"]].dropna()
    print("XAI spread>=2:", (xai.max(axis=1) - xai.min(axis=1) >= 2).sum(), "of", len(xai))
    ov = r[r.section == "overall"].pivot_table(index="criterion", columns="expert", values="rating")
    print("\nOverall items, rater tendencies by group (mean of rated):")
    print(r[r.section == "overall"].groupby("group").rating.agg(["count", "mean"]).round(2))

    print("\n=== Issues ===")
    cnt = issues.expert.value_counts().reindex(EXPERTS).astype(int)
    print("per expert:", cnt.to_dict(), "| total:", len(issues),
          "| ONT2 share: %.0f%%" % (100 * cnt["ONT2"] / len(issues)))


# --------------------------------------------------------------------------
# 3. Figures
# --------------------------------------------------------------------------
def _save(fig, out):
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_overall(ratings, out):
    """Dot plot: each expert's rating per overall criterion on the full 1-5 scale;
    grey line = min-max range, black bar = mean."""
    d = ratings[ratings.section == "overall"]
    crit = list(dict.fromkeys(d.criterion))
    experts = [e for e in EXPERTS if d[d.expert == e].rating.notna().any()]
    offs = np.linspace(-0.33, 0.33, len(experts))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 0.52 * len(crit) + 0.95))
    seen = set()                                   # label each expert once in the legend
    for i, c in enumerate(crit):
        vals = d[d.criterion == c].rating.dropna()
        ax.plot([vals.min(), vals.max()], [i, i], color="0.7", lw=0.9, zorder=1)
        for off, e in zip(offs, experts):
            v = d[(d.expert == e) & (d.criterion == c)].rating.iloc[0]
            if np.isnan(v):
                continue
            mk, col = STYLE[e]
            ax.scatter(v, i + off, marker=mk, s=24, color=col, edgecolor="k", linewidth=0.4,
                       zorder=3, label=e if e not in seen else None)
            seen.add(e)
        ax.plot([vals.mean()] * 2, [i - 0.40, i + 0.40], color="k", lw=2.0, zorder=4,
                solid_capstyle="butt", label="Mean" if i == 0 else None)
        ax.text(1.02, i, f"{vals.mean():.2f} ({len(vals)})", transform=ax.get_yaxis_transform(),
                va="center", ha="left")
        if i:
            ax.axhline(i - 0.5, color="0.9", lw=0.5, zorder=0)
    ax.text(1.02, -0.78, r"Mean ($n$)", transform=ax.get_yaxis_transform(), va="center",
            ha="left", weight="bold")
    ax.set_yticks(range(len(crit)), crit)
    ax.set_ylim(len(crit) - 0.5, -0.5)
    ax.set_xlim(1, 5.1)                            # full Likert range
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Rating (Likert scale, 1 to 5)")
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.86, top=0.93, bottom=0.21)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in experts + ["Mean"] if x in l]       # legend in expert order
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="lower center",
               ncol=len(order), frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.2)
    _save(fig, out)


def plot_specific(ratings, out):
    """Heat maps of the role-specific criteria (criteria x experts, plus mean column)."""
    panels = [("XAI", "XAI experts"), ("ONT", "Ontology engineers"), ("HC", "Healthcare expert")]
    cell_w, cell_h, label_w, gap, top, bottom = 0.30, 0.36, 1.05, 0.14, 0.62, 0.55
    ncols = {fam: len([e for e in EXPERTS if e.startswith(fam)]) + 1 for fam, _ in panels}
    nrows = EXPECTED_SPECIFIC
    W = sum(label_w + ncols[f] * cell_w for f, _ in panels) + gap * 2 + 0.1
    H = top + max(nrows.values()) * cell_h + bottom
    fig = plt.figure(figsize=(W, H))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f4f4f4")
    norm = mpl.colors.Normalize(1, 5)

    x = 0.05
    for fam, title in panels:
        exps = [e for e in EXPERTS if e.startswith(fam)]
        d = ratings[(ratings.section == "specific") & ratings.expert.isin(exps)]
        crit = list(dict.fromkeys(d.criterion))
        M = d.pivot_table(index="criterion", columns="expert", values="rating",
                          aggfunc="first", dropna=False).reindex(index=crit, columns=exps)
        spread = (M.max(axis=1) - M.min(axis=1)).to_numpy(float)      # rater spread per criterion
        M["Mean"] = M.mean(axis=1, skipna=True)
        arr = M.to_numpy(float)
        h = len(crit) * cell_h
        ax = fig.add_axes([(x + label_w) / W, (H - top - h) / H,
                           ncols[fam] * cell_w / W, h / H])
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, aspect="auto")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="/////",
                                           edgecolor="#cfcfcf", lw=0))
                    continue
                rgba = cmap(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt = f"{v:.1f}" if j == arr.shape[1] - 1 else f"{v:g}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="w" if lum < 0.5 else "k",
                        weight="bold" if j == arr.shape[1] - 1 else None)
        for i, sp in enumerate(spread):
            if sp >= 2:                                               # strong disagreement
                ax.add_patch(Rectangle((arr.shape[1] - 1.5, i - .5), 1, 1, fill=False,
                                       edgecolor="#c0392b", lw=1.6, zorder=5))
        ax.set_xticks(range(arr.shape[1]), list(M.columns), fontsize=7.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(crit)), [textwrap.fill(c, 19) for c in crit], fontsize=7.5)
        ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", lw=1.2)
        ax.tick_params(which="both", length=0)
        ax.axvline(arr.shape[1] - 1.5, color="w", lw=3.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, weight="bold", loc="left", pad=15, x=-label_w / (ncols[fam] * cell_w))
        x += label_w + ncols[fam] * cell_w + gap

    cax = fig.add_axes([0.36, 0.05 / H * 2.2, 0.28, 0.12 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1, 2, 3, 4, 5])
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.set_label("Rating (hatched = not rated; red outline = raters differ by $\\geq$2 points)", fontsize=8)
    _save(fig, out)


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main(pdf_path, outdir="."):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ratings, issues = parse_pdf(pdf_path)
    sanity_check(ratings, issues)
    ratings.to_csv(outdir / "ratings_long.csv", index=False)
    issues.to_csv(outdir / "issues.csv", index=False)
    print_summary(ratings, issues)
    plot_overall(ratings, outdir / "fig_overall_assessment")
    plot_specific(ratings, outdir / "fig_specific_criteria")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
