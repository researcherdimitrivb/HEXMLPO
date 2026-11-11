#!/usr/bin/env python3
"""Rebuild the appendix frequency tables (goals, types, quality criteria) as LaTeX
from extraction-second-pass.xlsx.

Method: for every canonical label, count its case-insensitive occurrences in the
relevant free-text column of each sheet (SLR, HMI = the HCI corpus). Every
occurrence counts, so a label mentioned twice in one cell counts twice. The "Both"
column is SLR + HCI, so the "Both" sheet is not needed.

Usage:
    python analyze-extractions-second-pass.py extraction-second-pass.xlsx -o analysis_out\appendix_tables.tex
"""
import argparse
import re

import pandas as pd

# Canonical labels. Order matters only for breaking ties in the descending sort.
GOALS = ["Decision Support", "Calibrate Trust", "Improve", "Discover",
         "Audit", "Justify", "Contest", "Persuade"]
TYPES = ["Rationale", "Data", "Trace-Based", "Scientific", "Everyday",
         "Safety and Performance", "Counterfactual", "Impact", "Contextual",
         "Statistical", "Case-Based", "Contrastive", "Responsibility",
         "Fairness", "Simulation-Based"]
QUALITIES = ["Context", "Coherence", "Completeness", "Correctness", "Confidence",
             "Consistency", "Covariate Complexity", "Controllability",
             "Composition", "Contrastivity", "Compactness", "Continuity"]

# (labels, column-name keyword, first-column header, LaTeX label, caption)
TABLES = [
    (GOALS, "goals", "Explanation goal", "tab:goal_frequency_paperlevel",
     r"Frequency of explanation goal occurrences mapped directly onto "
     r"\citeauthor{Adhikari2022}'s taxonomy~\cite{Adhikari2022}, per-paper unit of "
     r"analysis, in descending order of explanation goal count."),
    (TYPES, "Chari", "Explanation type", "tab:explanation_type_distribution",
     r"Frequency of explanation types~\cite{chari_explanation_2024}, per-paper unit "
     r"of analysis, in descending order of explanation type count."),
    (QUALITIES, "quality criteria", "Quality criterion", "tab:co12_distribution",
     r"Frequency of explanation quality criteria~\cite{Nauta2023}, per-paper unit of "
     r"analysis, in descending order of explanation quality criterion count."),
]


def normalise(text) -> str:
    """Map Unicode dashes (e.g. the non-breaking hyphen in 'Trace-Based') to '-'."""
    return re.sub(r"[\u2010-\u2015\u2212]", "-", str(text))


def find_column(df: pd.DataFrame, keyword: str) -> str:
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    raise KeyError(f"No column containing {keyword!r} in {list(df.columns)}")


def count_labels(df: pd.DataFrame, keyword: str, labels: list[str]) -> dict[str, int]:
    cells = df[find_column(df, keyword)].fillna("").map(normalise)
    return {lab: sum(len(re.findall(re.escape(lab), c, flags=re.I)) for c in cells)
            for lab in labels}


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}"


def latex_table(labels, first_col, label, caption, slr, hci) -> str:
    both = {lab: slr[lab] + hci[lab] for lab in labels}
    order = sorted(labels, key=lambda lab: -both[lab])  # stable: ties keep list order
    ts, th = sum(slr.values()), sum(hci.values())
    tb = ts + th
    rows = "\n".join(
        f"        {lab} & {slr[lab]} & {pct(slr[lab], ts)} & {hci[lab]} & "
        f"{pct(hci[lab], th)} & {both[lab]} & {pct(both[lab], tb)} \\\\"
        for lab in order)
    return rf"""\begin{{table}}[htbp]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{l rr rr rr}}
        \toprule
        & \multicolumn{{2}}{{c}}{{SLR}} & \multicolumn{{2}}{{c}}{{HCI}} & \multicolumn{{2}}{{c}}{{Both}} \\
        \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
        {first_col} & Count & \% & Count & \% & Count & \% \\
        \midrule
{rows}
        \midrule
        \textbf{{Total}} & \textbf{{{ts}}} & 100 & \textbf{{{th}}} & 100 & \textbf{{{tb}}} & 100 \\
        \bottomrule
    \end{{tabular}}
\end{{table}}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", help="path to extraction-second-pass.xlsx")
    ap.add_argument("-o", "--output", help="write LaTeX here instead of stdout")
    ap.add_argument("--slr-sheet", default="SLR")
    ap.add_argument("--hci-sheet", default="HMI", help="sheet holding the HCI corpus")
    args = ap.parse_args()

    sheets = pd.read_excel(args.xlsx, sheet_name=[args.slr_sheet, args.hci_sheet])
    slr_df = sheets[args.slr_sheet].dropna(how="all")
    hci_df = sheets[args.hci_sheet].dropna(how="all")

    blocks = []
    for labels, keyword, first_col, label, caption in TABLES:
        slr = count_labels(slr_df, keyword, labels)
        hci = count_labels(hci_df, keyword, labels)
        blocks.append(latex_table(labels, first_col, label, caption, slr, hci))
    out = "\n\n".join(blocks) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
