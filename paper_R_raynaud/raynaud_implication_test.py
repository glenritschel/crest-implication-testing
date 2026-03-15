"""
raynaud_implication_test.py
----------------------------
Statistical implication testing framework applied to Raynaud phenomenon (RP)
in Systemic Sclerosis (SSc) — the R in CREST syndrome.

Paper 02 in the CREST Implication Testing Series.

Dataset: Synthetic cohort calibrated from published clinical statistics:
  - VEDOSS registry (Lancet Rheumatol 2021, n=764): ANA, capillaroscopy,
    autoantibodies, progression to SSc
  - Hughes & Herrick (Nat Rev Rheumatol 2020): RP prevalence, digital ulcers
  - EUSTAR database (Walker et al. 2007): organ involvement
  - Cutolo et al. (Arthritis Rheum 2016): capillaroscopy and digital ulcers

Key calibration targets:
  - RP prevalence in SSc: ~96% (near-universal; Clements & Furst 2003)
  - Abnormal capillaroscopy in RP+SSc: ~85%
  - ANA positive in RP: ~74% (VEDOSS)
  - SSc-specific autoantibodies in RP: ~40% (VEDOSS)
  - Digital ulcers in SSc: ~30%
  - Severe RP (digital ulcers/gangrene) in SSc: ~50%

Four clinically motivated implications tested:
  1. Abnormal nailfold capillaroscopy => Raynaud phenomenon present
  2. ANA positive => Raynaud phenomenon present
  3. SSc-specific autoantibody (ACA or Scl-70) => Raynaud phenomenon present
  4. Digital ulcers present => Severe Raynaud (recurrent/digital ischaemia)

Methods:
  - PAC violation bounds (Clopper-Pearson, 95% CI)
  - Chi-squared association test (Phi coefficient)
  - Causal discovery (PC algorithm, Fisher-Z)
  - Invariance testing across sex
  - Rosenbaum sensitivity analysis

Requirements:
    pip3 install pandas numpy scipy scikit-learn matplotlib seaborn causal-learn

Usage:
    python3 raynaud_implication_test.py

Outputs:
    results/raynaud_implication_results.json
    figures/pac_bounds_raynaud.png
    figures/contingency_raynaud.png
    figures/sex_stratified_raynaud.png
    figures/summary_table_raynaud.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET GENERATION
# Calibrated from VEDOSS (Lancet Rheumatol 2021, n=764),
# Hughes & Herrick (Nat Rev Rheumatol 2020),
# Cutolo et al. (Arthritis Rheum 2016)
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
N = 800

def generate_ssc_cohort(n, seed=42):
    """
    Generate a synthetic SSc cohort with Raynaud-focused variables.

    Calibration targets:
      - RP prevalence: ~96% (near-universal in SSc)
      - Severe RP prevalence: ~50% (recurrent attacks + digital ischaemia)
      - Female: ~87%
      - lcSSc: ~62%
      - ANA positive: ~95%
      - ACA positive: ~35% overall (higher in lcSSc)
      - Anti-Scl70 positive: ~25% overall (higher in dcSSc)
      - SSc-specific autoantibody (ACA or Scl70): ~55%
      - Abnormal capillaroscopy: ~85% of SSc patients
      - Digital ulcers: ~30%
      - Puffy fingers: ~20%
      - Disease duration: mean ~10yr
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    sex              = rng.binomial(1, 0.87, n)   # 1=female
    age              = rng.normal(52, 14, n).clip(18, 85)
    disease_duration = rng.exponential(9, n).clip(0.5, 40)

    # ── Disease subtype ───────────────────────────────────────────────────────
    p_lc  = np.where(sex == 1, 0.65, 0.52)
    lc_ssc = rng.binomial(1, p_lc)

    # ── Autoantibodies ────────────────────────────────────────────────────────
    # ACA: strongly associated with lcSSc
    p_aca  = np.where(lc_ssc == 1, 0.48, 0.08)
    aca    = rng.binomial(1, p_aca)

    # Anti-Scl70: strongly associated with dcSSc
    p_scl70 = np.where(lc_ssc == 1, 0.06, 0.42)
    scl70   = rng.binomial(1, p_scl70)

    # ANA: near-universal in SSc
    ana = rng.binomial(1, 0.95, n)

    # SSc-specific autoantibody (ACA or Scl70)
    ssc_ab = np.clip(aca + scl70, 0, 1)

    # ── Capillaroscopy ────────────────────────────────────────────────────────
    # Abnormal capillaroscopy (SSc pattern) in ~85% of SSc patients
    # More abnormal with longer disease duration and dcSSc
    p_cap = 0.70 + 0.10 * (lc_ssc == 0) + 0.005 * disease_duration.clip(0, 20)
    p_cap = p_cap.clip(0.50, 0.97)
    abnormal_cap = rng.binomial(1, p_cap)

    # Late capillaroscopy pattern (avascular areas): more severe
    p_late = 0.25 + 0.15 * (lc_ssc == 0) + 0.01 * disease_duration.clip(0, 20)
    p_late = p_late.clip(0.10, 0.65)
    late_cap = rng.binomial(1, p_late)

    # ── Vascular features ─────────────────────────────────────────────────────
    # Digital ulcers: associated with dcSSc, longer duration, late capillaroscopy
    p_du = 0.15 + 0.18 * (lc_ssc == 0) + 0.12 * late_cap + \
           0.005 * disease_duration.clip(0, 20)
    p_du = p_du.clip(0.05, 0.75)
    digital_ulcers = rng.binomial(1, p_du)

    # Puffy fingers: early feature
    p_puffy = 0.15 + 0.05 * (disease_duration < 3)
    puffy_fingers = rng.binomial(1, p_puffy, n)

    # Telangiectasias
    p_tel = np.where(lc_ssc == 1, 0.62, 0.44)
    telangiectasias = rng.binomial(1, p_tel)

    # ── Raynaud phenomenon (outcome A: presence) ──────────────────────────────
    # RP is near-universal in SSc (~96%), but not all patients present with
    # classic RP at every visit. Abnormal capillaroscopy and autoantibodies
    # are the strongest predictors.
    log_odds_rp = (
        2.50                        # intercept (~92% baseline)
        + 0.80 * abnormal_cap       # capillaroscopy SSc pattern
        + 0.50 * ssc_ab             # SSc-specific autoantibody
        + 0.40 * ana                # ANA
        - 0.20 * (lc_ssc == 0)     # slightly lower in dcSSc (earlier severe damage)
    )
    p_rp = 1 / (1 + np.exp(-log_odds_rp))
    raynaud = rng.binomial(1, p_rp)

    # ── Severe Raynaud (outcome B: recurrent + digital ischaemia) ────────────
    # Severe RP defined as: recurrent attacks causing digital ischaemia,
    # ulceration, or persistent colour change. ~50% of SSc-RP patients.
    log_odds_severe = (
        0.00                        # intercept (~50% baseline)
        + 0.90 * digital_ulcers     # strongest predictor
        + 0.60 * late_cap           # late capillaroscopy pattern
        + 0.30 * (lc_ssc == 0)     # dcSSc more severe vascular disease
        + 0.04 * disease_duration   # longer duration
        - 0.20 * (sex == 1)        # women slightly less severe vascular
    )
    p_severe = 1 / (1 + np.exp(-log_odds_severe))
    severe_rp = rng.binomial(1, p_severe)
    # Severe RP only meaningful if RP present
    severe_rp = severe_rp * raynaud

    df = pd.DataFrame({
        "sex":              sex,
        "age":              age.round(1),
        "disease_duration": disease_duration.round(1),
        "lc_ssc":           lc_ssc,
        "aca":              aca,
        "scl70":            scl70,
        "ana":              ana,
        "ssc_autoantibody": ssc_ab,
        "abnormal_cap":     abnormal_cap,
        "late_cap":         late_cap,
        "digital_ulcers":   digital_ulcers,
        "puffy_fingers":    puffy_fingers,
        "telangiectasias":  telangiectasias,
        "raynaud":          raynaud,
        "severe_rp":        severe_rp,
    })
    return df


df = generate_ssc_cohort(N)

print(f"Synthetic SSc cohort: n={len(df)}")
print(f"Raynaud prevalence:      {df['raynaud'].mean():.1%}")
print(f"Severe RP prevalence:    {df['severe_rp'].mean():.1%}")
print(f"Female:                  {df['sex'].mean():.1%}")
print(f"lcSSc:                   {df['lc_ssc'].mean():.1%}")
print(f"ANA positive:            {df['ana'].mean():.1%}")
print(f"SSc autoantibody:        {df['ssc_autoantibody'].mean():.1%}")
print(f"Abnormal capillaroscopy: {df['abnormal_cap'].mean():.1%}")
print(f"Digital ulcers:          {df['digital_ulcers'].mean():.1%}")
print()

df.to_csv("results/synthetic_ssc_raynaud_cohort.csv", index=False)
print("Saved results/synthetic_ssc_raynaud_cohort.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPLICATION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.05

IMPLICATIONS = [
    {
        "name":    "AbnormalCap => Raynaud",
        "a_col":   "abnormal_cap",
        "a_fn":    lambda x: x == 1,
        "b_col":   "raynaud",
        "claim":   "Abnormal nailfold capillaroscopy implies Raynaud phenomenon",
        "source":  "Cutolo et al. 2016; VEDOSS 2021; Smith et al. 2020",
        "rationale": (
            "Abnormal nailfold capillaroscopy (SSc pattern: giant capillaries, "
            "haemorrhages, avascular areas) is the definitive microvascular marker "
            "of SSc-related Raynaud phenomenon. The 2013 ACR/EULAR SSc "
            "classification criteria include abnormal capillaroscopy as a major "
            "criterion. The presence of SSc-pattern capillaroscopy in a patient "
            "with RP is considered diagnostic of secondary (SSc-associated) RP."
        ),
    },
    {
        "name":    "LateCapillaroscopy => SevereRaynaud",
        "a_col":   "late_cap",
        "a_fn":    lambda x: x == 1,
        "b_col":   "severe_rp",
        "claim":   "Late nailfold capillaroscopy pattern implies severe Raynaud phenomenon",
        "source":  "Smith et al. 2013 (J Rheumatol); Cutolo et al. 2016",
        "rationale": (
            "The late capillaroscopy pattern (extensive avascular areas, aberrant "
            "neoangiogenesis) is associated with the most severe vascular disease in "
            "SSc. Smith et al. (2013) demonstrated that late capillaroscopy pattern "
            "independently predicts new severe organ involvement. Late pattern implies "
            "severe Raynaud because the avascular dropout seen in capillaroscopy "
            "directly reflects the tissue ischaemia driving digital ulceration."
        ),
    },
    {
        "name":    "SSc_autoantibody => Raynaud",
        "a_col":   "ssc_autoantibody",
        "a_fn":    lambda x: x == 1,
        "b_col":   "raynaud",
        "claim":   "SSc-specific autoantibody (ACA or anti-Scl70) implies Raynaud phenomenon",
        "source":  "VEDOSS 2021 (ACA HR=3.94); Steen & Medsger 2007",
        "rationale": (
            "SSc-specific autoantibodies (ACA and anti-topoisomerase I/Scl-70) are "
            "highly specific for SSc and are present in ~55% of SSc patients. "
            "Their presence in a patient with RP strongly predicts SSc diagnosis "
            "and therefore SSc-associated RP. In the VEDOSS cohort, ACA increased "
            "the hazard of SSc progression 3.94-fold beyond ANA alone."
        ),
    },
    {
        "name":    "DigitalUlcers => SevereRaynaud",
        "a_col":   "digital_ulcers",
        "a_fn":    lambda x: x == 1,
        "b_col":   "severe_rp",
        "claim":   "Digital ulcers imply severe Raynaud phenomenon",
        "source":  "Hughes & Herrick (Nat Rev Rheumatol 2020); Matucci-Cerinic 2016",
        "rationale": (
            "Digital ulcers are the most common severe complication of SSc-related "
            "RP, representing the endpoint of repeated ischaemic attacks. The "
            "presence of digital ulcers implies that Raynaud attacks are severe "
            "enough to cause tissue necrosis. This is recognised in clinical "
            "guidelines as an indication for escalated vasodilator therapy "
            "(prostanoids, PDE5 inhibitors, endothelin receptor antagonists)."
        ),
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS (identical to calcinosis pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def pac_bound(n, k, alpha=ALPHA):
    if n == 0:
        return None
    if k == 0:
        return 1 - (alpha ** (1.0 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def chi_squared_test(a_vec, b_vec):
    ct = pd.crosstab(a_vec, b_vec)
    if ct.shape != (2, 2):
        return {"chi2": None, "p_value": None, "phi": None,
                "note": "Non-2x2 table"}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    phi = float(np.sqrt(chi2 / len(a_vec)))
    return {
        "chi2":    round(float(chi2), 3),
        "p_value": round(float(p), 6),
        "dof":     int(dof),
        "phi":     round(phi, 3),
    }


def causal_discovery(df, a_col, confounders, alpha=0.05):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cols  = confounders + [a_col, "raynaud"]
        # use severe_rp if implication targets it
        if a_col == "digital_ulcers":
            cols = confounders + [a_col, "severe_rp"]
        sub   = df[cols].dropna().values.astype(float)
        cg    = pc(sub, alpha=alpha, indep_test=fisherz, show_progress=False)
        g     = cg.G
        a_idx = cols.index(a_col)
        b_idx = len(cols) - 1
        has_edge    = bool(g.graph[a_idx, b_idx] != 0 or
                          g.graph[b_idx, a_idx] != 0)
        directed_ab = bool(g.graph[a_idx, b_idx] == -1 and
                          g.graph[b_idx, a_idx] ==  1)
        return {"has_edge": has_edge, "directed_A_to_B": directed_ab,
                "note": "PC algorithm, Fisher-Z"}
    except ImportError:
        return {"has_edge": None, "note": "causal-learn not installed"}
    except Exception as e:
        return {"has_edge": None, "error": str(e)}


def invariance_test(df, a_vec, b_col, group_col, min_size=10):
    results = {}
    for grp_val in sorted(df[group_col].unique()):
        mask  = df[group_col] == grp_val
        b1    = df.loc[mask & (a_vec == 1), b_col]
        n     = len(b1)
        if n < min_size:
            results[f"{group_col}={grp_val}"] = {"n": n, "skipped": True}
            continue
        k   = int((b1 == 0).sum())
        pac = pac_bound(n, k)
        results[f"{group_col}={grp_val}"] = {
            "n_a1":          n,
            "n_violations":  k,
            "violation_rate": round(k / n, 3),
            "pac_bound":     round(pac, 4) if pac else None,
        }
    any_viol = any(
        v.get("n_violations", 0) > 0
        for v in results.values() if not v.get("skipped")
    )
    return {"invariant": not any_viol, "groups": results}


def rosenbaum_gamma(n, k, alpha=ALPHA):
    pac = pac_bound(n, k, alpha)
    if pac is None or pac >= 1.0:
        return None
    return round(pac / (1.0 - pac), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

CONFOUNDERS = ["age", "sex", "disease_duration", "lc_ssc"]

all_results = {}

for imp in IMPLICATIONS:
    name  = imp["name"]
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    b_col = imp["b_col"]
    b_vec = df[b_col]

    print(f"\n{'='*65}")
    print(f"Testing: {name}")
    print(f"Claim:   {imp['claim']}")
    print(f"{'='*65}")

    ct = pd.crosstab(a_vec, b_vec,
                     rownames=["A (antecedent)"],
                     colnames=[f"B ({b_col})"])
    print(f"\nContingency table:\n{ct}\n")

    a1        = b_vec[a_vec == 1]
    n_a1      = len(a1)
    n_a0      = int((a_vec == 0).sum())
    n_viol    = int((a1 == 0).sum())
    viol_rate = n_viol / n_a1 if n_a1 > 0 else None

    pac   = pac_bound(n_a1, n_viol)
    chi   = chi_squared_test(a_vec, b_vec)
    print(f"N(A=1): {n_a1}  Violations: {n_viol}  Rate: {viol_rate:.3f}")
    print(f"PAC bound: P(violation) < {pac:.4f}  [95% CI]")
    print(f"Chi-squared: chi2={chi['chi2']}  p={chi['p_value']:.6f}  phi={chi['phi']}")

    print("Running causal discovery...")
    causal = causal_discovery(df, imp["a_col"], CONFOUNDERS)
    print(f"Causal edge: {causal}")

    inv    = invariance_test(df, a_vec, b_col, "sex")
    print(f"Invariant across sex: {inv['invariant']}")
    for grp, grp_res in inv["groups"].items():
        if not grp_res.get("skipped"):
            print(f"  {grp}: n={grp_res['n_a1']}, "
                  f"violations={grp_res['n_violations']}, "
                  f"rate={grp_res['violation_rate']}, "
                  f"PAC={grp_res['pac_bound']}")

    gamma = rosenbaum_gamma(n_a1, n_viol)
    print(f"Rosenbaum Gamma: {gamma}")

    chi_sig   = (chi["p_value"] or 1.0) < ALPHA
    low_viol  = (viol_rate or 1.0) < 0.30
    causal_ok = causal.get("has_edge", False)
    invariant = inv.get("invariant", False)

    verdict = (
        "STRONG"   if (low_viol and chi_sig and causal_ok and invariant) else
        "MODERATE" if (low_viol and chi_sig and (causal_ok or invariant))  else
        "WEAK"     if (low_viol and chi_sig)                               else
        "TRENDING" if chi_sig                                               else
        "REJECTED"
    )
    print(f"\n*** VERDICT: {verdict} ***")

    all_results[name] = {
        "claim":           imp["claim"],
        "source":          imp["source"],
        "rationale":       imp["rationale"],
        "outcome":         b_col,
        "n_a1":            n_a1,
        "n_a0":            n_a0,
        "n_violations":    n_viol,
        "violation_rate":  round(viol_rate, 3) if viol_rate is not None else None,
        "pac_bound":       round(pac, 4) if pac else None,
        "chi_squared":     chi,
        "causal":          causal,
        "invariance_sex":  inv,
        "rosenbaum_gamma": gamma,
        "verdict":         verdict,
    }

with open("results/raynaud_implication_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nSaved results/raynaud_implication_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

VERDICT_COLORS = {
    "STRONG":   "#2ecc71",
    "MODERATE": "#3498db",
    "WEAK":     "#f39c12",
    "TRENDING": "#e67e22",
    "REJECTED": "#e74c3c",
}

SHORT_LABELS = [
    "Abnormal Cap\n=> Raynaud",
    "Late Cap\n=> Severe RP",
    "SSc autoantibody\n=> Raynaud",
    "Digital Ulcers\n=> Severe RP",
]

# ── Figure 1: PAC bounds ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
names, rates, bounds, colors = [], [], [], []
for name, res in all_results.items():
    names.append(name)
    rates.append(res["violation_rate"] or 0)
    bounds.append(res["pac_bound"] or 0)
    colors.append(VERDICT_COLORS.get(res["verdict"], "#95a5a6"))

x = np.arange(len(names))
ax.bar(x - 0.2, rates,  width=0.35, color=colors, alpha=0.9,
       edgecolor="white", label="Observed violation rate")
ax.bar(x + 0.2, bounds, width=0.35, color=colors, alpha=0.35,
       edgecolor=colors, linewidth=1.5, hatch="//",
       label="PAC upper bound (95% CI)")
ax.axhline(0.30, color="gray", linestyle="--", linewidth=1.2,
           label="30% violation threshold")
ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, fontsize=9)
ax.set_ylabel("Violation probability", fontsize=10)
ax.set_ylim(0, 0.85)
ax.set_title("Raynaud Phenomenon Implication Violation Rates and PAC Bounds\n"
             "SSc Synthetic Cohort (n=800, calibrated to VEDOSS 2021 & EUSTAR)",
             fontsize=11)
ax.legend(fontsize=9)

for i, (v, b, name) in enumerate(zip(rates, bounds, names)):
    res = all_results[name]
    p   = res["chi_squared"]["p_value"] or 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(i, b + 0.025, f"χ² {sig}", ha="center", fontsize=8, color="dimgray")
    ax.text(x[i] - 0.2, v + 0.012, f"{v:.2f}", ha="center",
            fontsize=9, fontweight="bold", color="white" if v > 0.08 else "black")

plt.tight_layout()
plt.savefig("figures/pac_bounds_raynaud.png", dpi=150)
plt.close()
print("Saved figures/pac_bounds_raynaud.png")

# ── Figure 2: Contingency bar charts ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 4))
fig.suptitle("Raynaud Prevalence by Antecedent Status", fontsize=12)

outcome_labels = ["No RP", "RP present",
                  "No RP", "RP present",
                  "No RP", "RP present",
                  "Mild/no RP", "Severe RP"]

for ax, imp, short in zip(axes, IMPLICATIONS, SHORT_LABELS):
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    b_col = imp["b_col"]
    ct    = pd.crosstab(a_vec, df[b_col])
    ct.index = ["A=0\n(absent)", "A=1\n(present)"]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"],
                edgecolor="white", width=0.6)
    res = all_results[imp["name"]]
    ax.set_title(f"{short}\n[{res['verdict']}]", fontsize=8)
    ax.set_ylabel("% of patients" if ax == axes[0] else "")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("figures/contingency_raynaud.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/contingency_raynaud.png")

# ── Figure 3: Sex-stratified violation rates ──────────────────────────────────
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 5), sharey=True)
fig.suptitle("Raynaud Implication Violation Rates by Sex\n"
             "(Female = sex=1, Male = sex=0)", fontsize=11)

SEX_COLORS = {
    "Female\n(sex=1)": "#e91e8c",
    "Male\n(sex=0)":   "#1e88e5",
    "Overall":         "#607d8b",
}

for ax, imp, short in zip(axes, IMPLICATIONS, SHORT_LABELS):
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    b_col = imp["b_col"]
    res   = all_results[imp["name"]]

    group_labels, viol_rates, pac_ups, bar_colors = [], [], [], []

    for sex_val, sex_label in [(1, "Female\n(sex=1)"), (0, "Male\n(sex=0)")]:
        mask  = (df["sex"] == sex_val) & (a_vec == 1)
        sub_b = df.loc[mask, b_col]
        n     = len(sub_b)
        if n < 10:
            continue
        k   = int((sub_b == 0).sum())
        vr  = k / n
        pac = pac_bound(n, k)
        group_labels.append(sex_label)
        viol_rates.append(vr)
        pac_ups.append(pac or vr)
        bar_colors.append(SEX_COLORS[sex_label])

    group_labels.append("Overall")
    viol_rates.append(res["violation_rate"] or 0)
    pac_ups.append(res["pac_bound"] or 0)
    bar_colors.append(SEX_COLORS["Overall"])

    xp    = np.arange(len(group_labels))
    bars2 = ax.bar(xp, viol_rates, color=bar_colors,
                   edgecolor="white", width=0.5, alpha=0.88)
    yerr_upper = [max(0, p - v) for p, v in zip(pac_ups, viol_rates)]
    ax.errorbar(xp, viol_rates,
                yerr=[np.zeros(len(xp)), yerr_upper],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.axhline(0.30, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(xp)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_title(short.replace("\n", " "), fontsize=8)
    ax.set_ylim(0, 0.85)
    if ax == axes[0]:
        ax.set_ylabel("Violation rate  (↑ error bar = PAC bound)", fontsize=9)
    for bar, v in zip(bars2, viol_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=l) for l, c in SEX_COLORS.items()]
fig.legend(handles=legend_els, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("figures/sex_stratified_raynaud.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/sex_stratified_raynaud.png")

# ── Figure 4: Summary table ───────────────────────────────────────────────────
rows = []
for name, res in all_results.items():
    chi_p = res['chi_squared']['p_value']
    rows.append({
        "Implication":  name,
        "Outcome":      res["outcome"],
        "N(A=1)":       res["n_a1"],
        "Violations":   res["n_violations"],
        "Viol. rate":   f"{res['violation_rate']:.2f}",
        "PAC bound":    f"{res['pac_bound']:.3f}",
        "Chi-sq p":     f"{chi_p:.4f}",
        "Phi":          f"{res['chi_squared']['phi']:.3f}",
        "Causal edge":  str(res["causal"].get("has_edge", "?")),
        "Invariant":    str(res["invariance_sex"].get("invariant", "?")),
        "Gamma":        str(res["rosenbaum_gamma"]),
        "Verdict":      res["verdict"],
    })

df_summary = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(18, max(2, len(rows) * 0.9 + 2)))
ax.axis("off")
tbl = ax.table(cellText=df_summary.values,
               colLabels=df_summary.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
tbl.auto_set_column_width([0])

vc = df_summary.columns.get_loc("Verdict")
for i, row in enumerate(rows):
    tbl[(i + 1, vc)].set_facecolor(VERDICT_COLORS.get(row["Verdict"], "#fff"))
    tbl[(i + 1, vc)].set_text_props(color="white", fontweight="bold")
for j in range(len(df_summary.columns)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

ax.set_title("Raynaud Phenomenon Implication Testing Summary — CREST Paper 02",
             fontsize=13, pad=16, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/summary_table_raynaud.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/summary_table_raynaud.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RAYNAUD PHENOMENON IMPLICATION TESTING — FINAL SUMMARY")
print("="*65)
for name, res in all_results.items():
    chi_p = res['chi_squared']['p_value']
    print(f"\n{name}")
    print(f"  Outcome: {res['outcome']}")
    print(f"  N(A=1)={res['n_a1']}  violations={res['n_violations']}  "
          f"rate={res['violation_rate']:.3f}  PAC<{res['pac_bound']:.3f}")
    print(f"  chi2 p={chi_p:.4f}  phi={res['chi_squared']['phi']:.3f}  "
          f"causal={res['causal'].get('has_edge','?')}  "
          f"invariant={res['invariance_sex'].get('invariant','?')}  "
          f"gamma={res['rosenbaum_gamma']}")
    print(f"  VERDICT: {res['verdict']}")

print("\nAll outputs saved to results/ and figures/")
