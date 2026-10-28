#!/usr/bin/env python3
"""
RQ1 clustering pipeline: stakeholder types (k=10) and explanation goals (k=20),
run separately on the SLR, HMI and Both tabs of RQ1-Extraction.xlsx.

Reconstruction of the pipeline used in the original session
(load_clean.py + cluster_stakeholders.py + cluster_goals.py + outlier dump).

Method
  1. Load each tab, align the HMI column names to the SLR schema.
  2. Clean the free text (strip footnote digits and "(Author et al. - YYYY...)"
     artefacts, normalise whitespace/dashes).
  3. Stakeholders: one mention per row.  Goals: multi-value cells split on ';'.
  4. TF-IDF (1-2-grams, English stop words) + k-means (k=10 stakeholders, k=20 goals).
  5. Per-cluster size, mean silhouette, top centroid terms, member phrases.
  6. Outliers = lowest per-item silhouette scores + clusters with <= 2 members.

Usage
  pip install pandas openpyxl scikit-learn
  python cluster_analysis.py --path RQ1-Extraction.xlsx --outdir analysis_out
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples

SHEETS = ["SLR", "HMI", "Both"]
SEED = 42

# HMI (and possibly other tabs) use different capitalisation / a renamed column.
COLMAP = {
    "Stakeholder Type": "Stakeholder type",
    "Background Knowledge Level": "Background knowledge level",
    "Preferred Explanation Modalities": "Preferred explanation modalities",
    "Explicit Explanation Goals": "Explicit explanation goals",
    "Explanation Type": "Explanation type",
    "Explanation ML Technique (duplicate of Explanation Type column)": "Explanation technique",
    "Explanation Quality Criteria (Co-12 Mapping)": "Explanation quality criteria (Co-12 mapping)",
}


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def load_tab(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.rename(columns=COLMAP)
    df["__source_tab"] = sheet
    return df


def clean_text(s):
    """Remove annotation artefacts but keep the semantic content of the label."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")  # non-breaking hyphen, en/em dash
    s = re.sub(r"\([^()]*et al\.[^()]*\)", " ", s)     # "(Eiband et al. - 2018...)" source tags
    s = re.sub(r"\s+\d{1,2}\s*$", "", s)               # trailing footnote/count digits (1, 2, 3)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_multi(s):
    """Split a multi-valued goals cell on semicolons; drop empties / placeholders."""
    parts = [clean_text(p) for p in str(s).split(";")]
    return [p for p in parts if p and p not in {"-", "\u2014", "nan", "Not stated"}]


# --------------------------------------------------------------------------
# Generic clustering
# --------------------------------------------------------------------------
def cluster_texts(texts, k, seed=SEED, n_top_terms=8):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    sil = silhouette_samples(X, labels)  # default Euclidean metric (matches original run)

    terms = np.array(vec.get_feature_names_out())
    results = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        top = terms[np.argsort(km.cluster_centers_[c])[::-1][:n_top_terms]].tolist()
        results.append({
            "cluster": int(c),
            "n": int(len(idx)),
            "mean_sil": float(sil[idx].mean()),
            "top_terms": top,
            "members": [texts[i] for i in idx],
        })
    results.sort(key=lambda r: r["n"], reverse=True)
    return labels, sil, results


def cluster_stakeholders(path, sheet, k=10):
    df = load_tab(path, sheet)
    df["clean_stakeholder"] = df["Stakeholder type"].map(clean_text)
    df = df[df["clean_stakeholder"] != ""].reset_index(drop=True)
    labels, sil, results = cluster_texts(df["clean_stakeholder"].tolist(), k)
    df["cluster"], df["sil"] = labels, sil
    return df, results


def cluster_goals(path, sheet, k=20):
    df = load_tab(path, sheet)
    rows = []
    for _, r in df.iterrows():
        for g in split_multi(r["Explicit explanation goals"]):
            rows.append({"goal": g,
                         "author": r["Author surname (year)"],
                         "stakeholder": clean_text(r["Stakeholder type"])})
    gdf = pd.DataFrame(rows)
    labels, sil, results = cluster_texts(gdf["goal"].tolist(), k)
    gdf["cluster"], gdf["sil"] = labels, sil
    return gdf, results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_clusters(title, results, n_show=6):
    lines = [f"===== {title} ====="]
    for r in results:
        lines.append(f"\n--- Cluster {r['cluster']} (n={r['n']}, mean_sil={r['mean_sil']:.3f}) ---")
        lines.append(f"top terms: {r['top_terms']}")
        for m in r["members"][:n_show]:
            lines.append(f"   - {m}")
        if r["n"] > n_show:
            lines.append(f"   ... (+{r['n'] - n_show} more)")
    return "\n".join(lines) + "\n"


def format_outliers(df, text_col, n=8):
    worst = df.sort_values("sil").head(n)[[text_col, "cluster", "sil"]]
    sizes = df["cluster"].value_counts()
    small = {int(c): int(v) for c, v in sizes.items() if v <= 2}
    return worst.to_string(index=False) + f"\nSmall clusters (<=2 members): {small}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="RQ1-Extraction.xlsx")
    ap.add_argument("--outdir", default="analysis_out")
    ap.add_argument("--k-stakeholders", type=int, default=10)
    ap.add_argument("--k-goals", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 250)

    # Sanity check: is "Both" simply SLR + HMI?
    n = {s: len(load_tab(args.path, s)) for s in SHEETS}
    print("Rows per tab:", n, "| SLR+HMI =", n["SLR"] + n["HMI"])

    stake_txt, goal_txt, outlier_txt, dump = [], [], [], {}

    for sheet in SHEETS:
        sdf, sres = cluster_stakeholders(args.path, sheet, args.k_stakeholders)
        gdf, gres = cluster_goals(args.path, sheet, args.k_goals)

        stake_txt.append(format_clusters(f"{sheet} - stakeholder clusters (k={args.k_stakeholders})", sres))
        goal_txt.append(format_clusters(
            f"{sheet} - goal mentions total: {len(gdf)} - clusters (k={args.k_goals})", gres))

        outlier_txt.append(f"--- {sheet} stakeholder outliers ---\n" + format_outliers(sdf, "clean_stakeholder", 8))
        outlier_txt.append(f"--- {sheet} goal outliers ---\n" + format_outliers(gdf, "goal", 6))

        dump[sheet] = {"stakeholders": sres, "goals": gres}

    with open(os.path.join(args.outdir, "stakeholders_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stake_txt))
    with open(os.path.join(args.outdir, "goals_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(goal_txt))
    with open(os.path.join(args.outdir, "outliers_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(outlier_txt))
    with open(os.path.join(args.outdir, "all_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    print("\n".join(stake_txt))
    print("\n".join(goal_txt))
    print("\n".join(outlier_txt))
    print(f"\nSaved outputs to ./{args.outdir}/")


if __name__ == "__main__":
    main()
