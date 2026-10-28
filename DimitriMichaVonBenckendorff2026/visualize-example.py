"""
Worked example figure: prediction + SHAP + LIME + DiCE for ONE patient, written for a lay reader.

Inputs  (all produced by explaining-CAN-screening.ipynb, EL-LR model):
    GE-71_Data_Summary_Table_final.csv
    shap_local_values_EL-LR.json
    lime_local_explanations_EL-LR.json
    dice_counterfactuals_EL-LR.json
    model_info.json   <- exported with the snippet below (coefficients, scaler, probabilities, unscaled values)

Usage:
    python make_worked_example.py --patient 29 --out figures

--------------------------------------------------------------------------
Add this to the notebook right after the final model is fitted (inside the
`for model_name, config in models.items():` loop, after `base_estimator = ...`)
to export model_info.json:

    from sklearn.pipeline import Pipeline as _P
    sc = pre_pipe.named_steps['scaler']
    X_unscaled = _P(pre_pipe.steps[:-1]).transform(X)          # engineered, imputed, NOT scaled
    logit = X_transformed_df.values @ base_estimator.coef_[0] + base_estimator.intercept_[0]
    json.dump({
        'columns': list(X_transformed_df.columns),
        'coef': [float(c) for c in base_estimator.coef_[0]],
        'intercept': float(base_estimator.intercept_[0]),
        'threshold': float(best_model.named_steps['clf'].threshold_),
        'center': [float(c) for c in sc.center_], 'scale': [float(c) for c in sc.scale_],
        'prob_CAN': [float(p) for p in 1 / (1 + np.exp(-logit))],
        'target': [int(v) for v in y],
        'unscaled': {c: [float(v) for v in X_unscaled[c]] for c in X_transformed_df.columns},
    }, open('model_info.json', 'w'))
--------------------------------------------------------------------------
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOWARD, AWAY, GREY = "#D55E00", "#0072B2", "#8c8c8c"   # colour-blind-safe pair; direction is also written out
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False})

# plain-language names for the variables the model actually uses (others fall back to a cleaned name)
PLAIN = {
    "HF_Power": "HF power of the heartbeat rhythm",
    "DB_RMSSD_ms": "Heart-rate variability (RMSSD)",
    "Duration": "Years with high blood pressure",
    "INT_Hypertension_Duration": "Years with high blood pressure",
    "INT_Vascular_Toxicity": "Years with diabetes (with high blood pressure)",
}


def plain(f):
    return PLAIN.get(f, f.replace("_", " ").replace("INT ", ""))


def fmt(v):
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}" if abs(v) >= 1 else f"{v:.2g}"


class Data:
    def __init__(self, a, i):
        r = lambda f: json.load(open(os.path.join(a.data, f)))
        self.m = r("model_info.json")
        self.shap, self.lime, self.dice = (r("shap_local_values_EL-LR.json"), r("lime_local_explanations_EL-LR.json"),
                                           r("dice_counterfactuals_EL-LR.json"))
        raw = os.path.join(a.data, "dice_counterfactuals_EL-LR_raw.json")
        self.dice_raw = os.path.exists(raw)                       # raw-unit output of dice_raw_counterfactuals.py
        if self.dice_raw:
            self.dice = json.load(open(raw))
        self.csv = pd.read_csv(os.path.join(a.data, "GE-71_Data_Summary_Table_final.csv"))
        self.i, self.cols = i, self.m["columns"]
        self.k = {c: n for n, c in enumerate(self.cols)}
        self.coef = np.array(self.m["coef"])
        self.prob = self.m["prob_CAN"][i]
        self.target = self.m["target"][i]
        self.subject = self.csv["SUBJECT NUMBER"].iloc[i]   # row i of the CSV = key i in the JSON files

    def raw(self, f, z=None):          # scaled value -> clinical units (inverse RobustScaler)
        z = (self.m["unscaled"][f][self.i] - self.m["center"][self.k[f]]) / self.m["scale"][self.k[f]] if z is None else z
        return self.m["center"][self.k[f]] + self.m["scale"][self.k[f]] * z

    def patient_value(self, f):
        return self.m["unscaled"][f][self.i]

    def prob_from_scaled(self, z):     # model probability for a (counterfactual) scaled feature vector
        return 1 / (1 + np.exp(-(np.array([z[c] for c in self.cols]) @ self.coef + self.m["intercept"])))

    def reference_max(self, f):        # highest value seen in another patient, ignoring ONE extreme case
        v = np.sort(np.array(self.m["unscaled"][f]))
        return v[-2] if v[-1] > 20 * max(v[-2], 1e-9) else v[-1]


def head(ax, title, sub):
    ax.text(0, 1.16, title, transform=ax.transAxes, fontweight="bold", fontsize=12.5, va="bottom")
    ax.text(0, 1.05, sub, transform=ax.transAxes, fontsize=9.3, color="#444", va="bottom", style="italic")


def barlist(ax, rows, xlim, colour=True):
    """Bars with the label written ABOVE each bar, so nothing spills into the neighbouring panel."""
    n = len(rows)
    for r, (label, v) in enumerate(rows):
        y = n - 1 - r
        ax.barh(y, v, height=0.34, color=TOWARD if v >= 0 else AWAY)
        ax.text(v + (0.02 if v >= 0 else -0.02) * (xlim[1] - xlim[0]) / 1.6, y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10.5, fontweight="bold")
        ax.text(xlim[0], y + 0.27, label, va="bottom", ha="left", fontsize=9.6)
    ax.set_xlim(*xlim); ax.set_ylim(-0.55, n - 0.2); ax.set_yticks([]); ax.set_xticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)


# ---------------------------------------------------------------- panel 1
def panel_prediction(ax, d):
    row = d.csv.iloc[d.i]
    who = {"F": "woman", "M": "man"}.get(str(row.get("Gender")), "patient")
    bits = [f"{who}, {int(row['Age'])}", "diabetes"]
    if str(row.get("Hypertension")).lower() == "yes" and not np.isnan(row.get("Duration", np.nan)):
        bits.append(f"high blood pressure for {int(row['Duration'])} years")
    head(ax, "1  The prediction", f"Patient {d.subject}: " + ", ".join(bits))
    p, thr = d.prob * 100, d.m["threshold"] * 100
    ax.barh([0], [100], height=0.34, color="#e6e6e6")
    ax.barh([0], [p], height=0.34, color=TOWARD if p >= thr else AWAY)
    ax.plot([thr, thr], [-0.26, 0.26], color="k", lw=1.8, ls="--")
    ax.text(thr, 0.30, f"decision line ({thr:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.text(p / 2, 0, f"{p:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=17)
    pred = "CAN" if p >= thr else "no CAN"
    truth = "CAN" if d.target == 1 else "no CAN"
    ax.text(0, -0.42, "Estimated likelihood of cardiac autonomic neuropathy (CAN)", fontsize=9.5)
    ax.text(0, -0.78, f"Model\u2019s prediction:  {pred}", fontsize=12, fontweight="bold")
    ax.text(0, -1.10, f"Actual diagnosis (target):  {truth}   " + ("\u2713 correct" if pred == truth else "\u2717 wrong"),
            fontsize=12, fontweight="bold", color="#222")
    ax.set_xlim(0, 100); ax.set_ylim(-1.25, 0.6); ax.axis("off")


# ---------------------------------------------------------------- panel 2
def panel_shap(ax, d, top=4):
    s = dict(d.shap[str(d.i)])
    # two model variables hold the identical number for a patient with high blood pressure -> add them up
    if "Duration" in s and "INT_Hypertension_Duration" in s and \
            np.isclose(d.patient_value("Duration"), d.patient_value("INT_Hypertension_Duration")):
        s["Duration"] += s.pop("INT_Hypertension_Duration")
    items = sorted(s.items(), key=lambda kv: -abs(kv[1]))[:top]
    items = [(f, v) for f, v in items if d.coef[d.k[f]] != 0 or f == "Duration"]
    head(ax, "2  SHAP: what pushed the estimate?", "How much each fact moved the estimate (vs. the average patient)")
    rows = []
    for f, v in items:
        val = d.patient_value(f)
        unit = " ms" if f == "DB_RMSSD_ms" else (" years" if f in ("Duration", "INT_Vascular_Toxicity") else "")
        extra = ", estimated" if f == "INT_Vascular_Toxicity" else ""
        rows.append((f"{plain(f)}: {fmt(val)}{unit}{extra}", v))
    xm = max(abs(v) for _, v in items) * 1.5
    barlist(ax, rows, (-xm * 0.15, xm))
    for r in range(len(rows)):                      # zero line only beside the bars, never through the labels
        ax.plot([0, 0], [len(rows) - 1 - r - 0.25, len(rows) - 1 - r + 0.25], color="k", lw=0.9)
    ax.text(0, -0.5, "pushes toward CAN \u2192", fontsize=9.3, va="top")


# ---------------------------------------------------------------- panel 3
RULE = re.compile(r"^(?:(?P<lo>-?\d+\.\d+) < )?(?P<f>.+?) (?P<op><=|>) (?P<hi>-?\d+\.\d+)$")


def humanise(d, rule):
    m = RULE.match(rule)
    f, op = m["f"], m["op"]
    if f not in d.k:
        return f, None
    c, sc = d.m["center"][d.k[f]], d.m["scale"][d.k[f]]
    hi = c + sc * float(m["hi"])
    if m["lo"] is not None:
        txt = f"{fmt(c + sc * float(m['lo']))} < {plain(f)} \u2264 {fmt(hi)}"
    else:
        txt = f"{plain(f)} {'\u2264' if op == '<=' else '>'} {fmt(hi)}"
    return txt, f


def panel_lime(ax, d, top=3):
    rules = {}
    for rule, w in d.lime[str(d.i)]:
        txt, f = humanise(d, rule)
        if f is None or d.coef[d.k[f]] == 0 and f != "Duration":
            continue                                   # skip rules about variables the model does not use
        key = txt.replace(plain("INT_Hypertension_Duration"), plain("Duration")) if f in ("Duration", "INT_Hypertension_Duration") else txt
        rules[key] = rules.get(key, 0) + w             # identical variables -> merge
    items = sorted(rules.items(), key=lambda kv: -abs(kv[1]))[:top]
    n_other = len(d.lime[str(d.i)]) - len(items)
    head(ax, "3  LIME: simple rules near this patient", "If-then rules that mimic the model around this patient")
    rows = [("Rule: " + t, w) for t, w in items]
    xm = max(abs(w) for _, w in items) * 1.5
    barlist(ax, rows, (0, xm))
    ax.text(0, -0.5, f"pushes toward CAN \u2192   ({n_other} further rules with tiny weights not shown)", fontsize=9.0, va="top", color="#222")


# ---------------------------------------------------------------- panel 4
def panel_dice_raw(ax, d):
    cfs = d.dice[str(d.i)]
    head(ax, "4  DiCE: what would flip the prediction?", "\u201cWhat-if\u201d patients for whom the model predicts the opposite")
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if not cfs:
        ax.text(0.0, 0.62, "No realistic what-if found.", fontsize=12, fontweight="bold", va="center")
        ax.text(0.0, 0.40, "Within the range seen in real patients, no change in HF power or\nheart-rate variability flips the prediction. The estimate rests mainly\non years with high blood pressure, which cannot be undone.",
                fontsize=9.6, va="center", linespacing=1.4)
        return
    shown = [f for f in cfs[0] if f in d.k and any(abs(cf[f] - d.patient_value(f)) > 1e-6 for cf in cfs)]
    xs = [0.50, 0.65, 0.80, 0.95]
    y = 0.90
    for x, h in zip(xs, ["This\npatient", "What-if\n1", "What-if\n2", "What-if\n3"]):
        ax.text(x, y, h, ha="center", va="center", fontweight="bold", fontsize=9.3, linespacing=1.1)
    ax.plot([0, 1], [y - 0.09, y - 0.09], color="k", lw=0.8)
    y -= 0.21
    for f in shown:
        unit = ", ms" if f == "DB_RMSSD_ms" else ""
        ax.text(0.0, y, plain(f).replace("of the heartbeat rhythm", "").strip() + unit, va="center", fontsize=9.3)
        ax.text(xs[0], y, fmt(d.patient_value(f)), ha="center", va="center", fontsize=9.5)
        for x, cf in zip(xs[1:], cfs[:3]):
            same = abs(cf[f] - d.patient_value(f)) < 1e-6
            ax.text(x, y, "\u2013" if same else fmt(cf[f]), ha="center", va="center", fontsize=9.5)
        y -= 0.17
    ax.plot([0, 1], [y + 0.09, y + 0.09], color="#bbb", lw=0.8)
    ax.text(0.0, y, "Likelihood of CAN", va="center", fontsize=9.5, fontweight="bold")
    ax.text(xs[0], y, f"{d.prob * 100:.0f}%", ha="center", va="center", fontsize=9.5, fontweight="bold")
    for x, cf in zip(xs[1:], cfs[:3]):
        ax.text(x, y, f"{cf['_prob_CAN'] * 100:.0f}%", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.0, y - 0.16, "All values lie within the range seen in real patients (5th to 95th percentile).",
            fontsize=8.6, color="#444", va="center")


def panel_dice(ax, d):
    if d.dice_raw:
        return panel_dice_raw(ax, d)
    cfs = d.dice[str(d.i)]
    z0 = {c: (d.m["unscaled"][c][d.i] - d.m["center"][d.k[c]]) / d.m["scale"][d.k[c]] for c in d.cols}
    changed = [c for c in d.cols if any(abs(cf[c] - z0[c]) > 1e-6 for cf in cfs)]
    shown = [c for c in changed if d.coef[d.k[c]] != 0]            # only variables the model actually uses
    ignored = [plain(c) for c in changed if d.coef[d.k[c]] == 0]
    head(ax, "4  DiCE: what would flip the prediction?", "\u201cWhat-if\u201d patients for whom the model predicts the opposite")
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xs = [0.50, 0.65, 0.80, 0.95]
    y = 0.90
    for x, h in zip(xs, ["This\npatient", "What-if\n1", "What-if\n2", "What-if\n3"]):
        ax.text(x, y, h, ha="center", va="center", fontweight="bold", fontsize=9.3, linespacing=1.1)
    ax.plot([0, 1], [y - 0.09, y - 0.09], color="k", lw=0.8)
    y -= 0.21
    flagged = False
    for f in shown:
        unit = ", ms" if f == "DB_RMSSD_ms" else ""
        ax.text(0.0, y, plain(f).replace("of the heartbeat rhythm", "").strip() + unit, va="center", fontsize=9.3)
        ax.text(xs[0], y, fmt(d.patient_value(f)), ha="center", va="center", fontsize=9.5)
        for x, cf in zip(xs[1:], cfs):
            raw = d.raw(f, cf[f]); same = abs(cf[f] - z0[f]) < 1e-6
            far = raw > d.reference_max(f) * 1.5
            ax.text(x, y, "\u2013" if same else fmt(raw) + ("*" if far else ""), ha="center", va="center", fontsize=9.5,
                    color=TOWARD if far else "#222", fontweight="bold" if far else "normal")
            flagged |= far and not same
        y -= 0.17
    ax.plot([0, 1], [y + 0.09, y + 0.09], color="#bbb", lw=0.8)
    ax.text(0.0, y, "Likelihood of CAN", va="center", fontsize=9.5, fontweight="bold")
    ax.text(xs[0], y, f"{d.prob * 100:.0f}%", ha="center", va="center", fontsize=9.5, fontweight="bold")
    for x, cf in zip(xs[1:], cfs):
        p = d.prob_from_scaled(cf) * 100
        ax.text(x, y, "<1%" if p < 1 else f"{p:.0f}%", ha="center", va="center", fontsize=9.5, fontweight="bold")
    y -= 0.16
    if flagged and "HF_Power" in shown:
        ax.text(0.0, y, f"* far beyond any real patient (highest HF power in another patient: {fmt(d.reference_max('HF_Power'))})",
                fontsize=8.6, color=TOWARD, va="center")
        y -= 0.09
    if ignored:
        ax.text(0.0, y, "Also changed, but not used by the model: " + ", ".join(sorted(set(ignored))), fontsize=8.6,
                color="#444", va="center")


PANELS = [("prediction", panel_prediction), ("shap", panel_shap), ("lime", panel_lime), ("dice", panel_dice)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=int, default=29, help="row index (0-based) = key in the JSON files")
    ap.add_argument("--data", default=".", help="folder with the CSV, the three JSON files and model_info.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = Data(a, a.patient)
    # combined 2x2 figure
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.14, left=0.02, right=0.99, top=0.90, bottom=0.04)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax, (_, fn) in zip(axes, PANELS):
        fn(ax, d)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"worked_example_{d.subject}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # one file per panel (for LaTeX subfigures)
    for name, fn in PANELS:
        f, ax = plt.subplots(figsize=(6.4, 3.9))
        f.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.06)
        fn(ax, d)
        for ext in ("png", "pdf"):
            f.savefig(os.path.join(a.out, f"worked_example_{d.subject}_{name}.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(f)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
