"""
calcinosis_implication_test.py
-------------------------------
Statistical implication testing framework applied to calcinosis in
Systemic Sclerosis (SSc) — the C in CREST syndrome.

Paper 02 in the CREST Implication Testing Series.

Dataset: Synthetic cohort calibrated from published clinical statistics,
primarily Valenzuela et al. (2016) Semin Arthritis Rheum 46:344-349
(Scleroderma Clinical Trials Consortium, n=5218), supplemented by
Herrick & Gallas (2016) and Mahmood et al. (2016).

The synthetic data reproduces:
  - Overall calcinosis prevalence: ~25%
  - Sex distribution: ~85% female (typical SSc cohort)
  - Disease subtype distribution: ~60% lcSSc, ~40% dcSSc
  - Published association strengths between calcinosis and
    digital ulcers, anticentromere antibody, osteoporosis,
    telangiectasias, and disease subtype

Four clinically motivated implications tested:
  1. Anticentromere Ab positive => Calcinosis present
  2. Digital ulcers present    => Calcinosis present
  3. Limited cutaneous SSc     => Calcinosis present
  4. Osteoporosis present      => Calcinosis present

Methods:
  - PAC violation bounds (Clopper-Pearson, 95% CI)
  - Chi-squared association test (Phi coefficient)
  - Causal discovery (PC algorithm, Fisher-Z)
  - Invariance testing across sex
  - Rosenbaum sensitivity analysis

Requirements:
    pip3 install pandas numpy scipy scikit-learn matplotlib seaborn causal-learn

Usage:
    python3 calcinosis_implication_test.py

Outputs:
    results/calcinosis_implication_results.json
    figures/pac_bounds_calcinosis.png
    figures/contingency_calcinosis.png
    figures/sex_stratified_calcinosis.png
    figures/summary_table_calcinosis.png
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
# Calibrated from Valenzuela et al. 2016 (SCTC, n=5218)
# and Herrick & Gallas 2016, Mahmood et al. 2016
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
N = 800   # sample size — large enough for stable estimates, small enough for CPU

def generate_ssc_cohort(n, seed=42):
    """
    Generate a synthetic SSc cohort calibrated to published statistics.

    Key calibration targets (Valenzuela et al. 2016):
      - Calcinosis prevalence: ~25%
      - Female prevalence: ~85%
      - lcSSc prevalence: ~60%
      - ACA positive: ~35% overall (higher in lcSSc)
      - Digital ulcers: ~30% overall
      - Osteoporosis: ~5% overall
      - Telangiectasias: ~55% overall
      - Disease duration: mean ~10yr, sd ~8yr

    Conditional probabilities match published odds ratios:
      - OR(calcinosis|digital_ulcers) ≈ 3.9 (SCTC 2016)
      - OR(calcinosis|ACA)            ≈ 2.1 (SCTC 2016)
      - OR(calcinosis|osteoporosis)   ≈ 4.2 (SCTC 2016)
      - OR(calcinosis|lcSSc)          ≈ 1.5 (SCTC 2016, Herrick 2016)
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    sex = rng.binomial(1, 0.85, n)           # 1=female, 0=male (85% female)
    age = rng.normal(50, 14, n).clip(18, 85)
    disease_duration = rng.exponential(8, n).clip(0.5, 40)  # years from first non-RP symptom

    # ── Disease subtype ───────────────────────────────────────────────────────
    # lcSSc more common in women; dcSSc slightly more in men
    p_lc = np.where(sex == 1, 0.63, 0.52)
    lc_ssc = rng.binomial(1, p_lc)           # 1=lcSSc, 0=dcSSc

    # ── Autoantibodies ────────────────────────────────────────────────────────
    # ACA strongly associated with lcSSc (Steen 1984, SCTC 2016)
    p_aca = np.where(lc_ssc == 1, 0.48, 0.08)
    aca_pos = rng.binomial(1, p_aca)

    # Anti-Scl70 associated with dcSSc
    p_scl70 = np.where(lc_ssc == 1, 0.06, 0.38)
    scl70_pos = rng.binomial(1, p_scl70)

    # ── Clinical features ─────────────────────────────────────────────────────
    # Digital ulcers: more common in dcSSc and with longer disease duration
    p_du = 0.18 + 0.20 * (lc_ssc == 0) + 0.008 * disease_duration.clip(0, 15)
    p_du = p_du.clip(0.05, 0.70)
    digital_ulcers = rng.binomial(1, p_du)

    # Telangiectasias: more common in lcSSc
    p_tel = np.where(lc_ssc == 1, 0.62, 0.45)
    telangiectasias = rng.binomial(1, p_tel)

    # Raynaud phenomenon: near-universal
    raynaud = rng.binomial(1, 0.97, n)

    # Pulmonary involvement: more common in dcSSc
    p_lung = np.where(lc_ssc == 1, 0.30, 0.55)
    lung_involvement = rng.binomial(1, p_lung)

    # Osteoporosis: older patients, female, longer disease
    p_osteo = 0.01 + 0.025 * (age > 55) + 0.015 * (sex == 1) + 0.004 * disease_duration.clip(0,20)
    p_osteo = p_osteo.clip(0.005, 0.25)
    osteoporosis = rng.binomial(1, p_osteo)

    # Modified Rodnan Skin Score: higher in dcSSc
    mrss = np.where(lc_ssc == 1,
                    rng.normal(6, 5, n).clip(0, 20),
                    rng.normal(18, 9, n).clip(0, 51))

    # ── Calcinosis (outcome) ──────────────────────────────────────────────────
    # Base log-odds calibrated so overall prevalence ≈ 25%
    # Each predictor contributes log-OR from published literature
    log_odds = (
        -1.80                           # intercept (baseline ~14%)
        + 1.36 * digital_ulcers         # OR ≈ 3.9 (SCTC 2016)
        + 0.74 * aca_pos                # OR ≈ 2.1 (SCTC 2016)
        + 1.44 * osteoporosis           # OR ≈ 4.2 (SCTC 2016)
        + 0.40 * lc_ssc                 # OR ≈ 1.5 (Herrick 2016)
        + 0.35 * telangiectasias        # OR ≈ 1.4 (SCTC 2016)
        + 0.04 * disease_duration       # longer duration = more calcinosis
        - 0.35 * (lc_ssc == 0)         # dcSSc protective relative to lc
    )
    p_calc = 1 / (1 + np.exp(-log_odds))
    calcinosis = rng.binomial(1, p_calc)

    df = pd.DataFrame({
        "sex":              sex,
        "age":              age.round(1),
        "disease_duration": disease_duration.round(1),
        "lc_ssc":           lc_ssc,
        "aca_positive":     aca_pos,
        "scl70_positive":   scl70_pos,
        "digital_ulcers":   digital_ulcers,
        "telangiectasias":  telangiectasias,
        "raynaud":          raynaud,
        "lung_involvement": lung_involvement,
        "osteoporosis":     osteoporosis,
        "mrss":             mrss.round(0).astype(int),
        "calcinosis":       calcinosis,
    })
    return df

df = generate_ssc_cohort(N)

print(f"Synthetic SSc cohort: n={len(df)}")
print(f"Calcinosis prevalence: {df['calcinosis'].mean():.1%}")
print(f"Female: {df['sex'].mean():.1%}")
print(f"lcSSc: {df['lc_ssc'].mean():.1%}")
print(f"ACA positive: {df['aca_positive'].mean():.1%}")
print(f"Digital ulcers: {df['digital_ulcers'].mean():.1%}")
print(f"Osteoporosis: {df['osteoporosis'].mean():.1%}")
print(f"Telangiectasias: {df['telangiectasias'].mean():.1%}")
print()

# Save synthetic dataset
df.to_csv("results/synthetic_ssc_cohort.csv", index=False)
print("Saved results/synthetic_ssc_cohort.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPLICATION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.05  # confidence level for all PAC bounds

IMPLICATIONS = [
    {
        "name":    "ACA_positive => Calcinosis",
        "a_col":   "aca_positive",
        "a_fn":    lambda x: x == 1,
        "b_col":   "calcinosis",
        "claim":   "Anticentromere antibody positivity implies calcinosis",
        "source":  "Steen et al. 1984; Valenzuela et al. 2016 (SCTC); Mahmood et al. 2016",
        "rationale": (
            "ACA is the serological hallmark of limited cutaneous SSc and is the "
            "autoantibody most consistently associated with calcinosis across "
            "multiple cohorts. Mahmood et al. (2016) identified ACA as an "
            "independent predictor in a multivariate model (OR=2.1, p<0.001)."
        ),
    },
    {
        "name":    "DigitalUlcers => Calcinosis",
        "a_col":   "digital_ulcers",
        "a_fn":    lambda x: x == 1,
        "b_col":   "calcinosis",
        "claim":   "Digital ulcers imply calcinosis",
        "source":  "Valenzuela et al. 2016 (SCTC, OR=3.9); Chung et al. 2015",
        "rationale": (
            "Digital ulcers and calcinosis share a common pathophysiology of "
            "chronic ischaemia and tissue hypoxia. The SCTC study (n=5218) "
            "identified digital ulcers as the strongest clinical predictor of "
            "calcinosis (OR=3.9, 95% CI: 2.7-5.5, p<0.0001) in multivariate "
            "analysis."
        ),
    },
    {
        "name":    "lcSSc => Calcinosis",
        "a_col":   "lc_ssc",
        "a_fn":    lambda x: x == 1,
        "b_col":   "calcinosis",
        "claim":   "Limited cutaneous SSc subtype implies calcinosis",
        "source":  "Herrick & Gallas 2016; SCTC 2016; Akesson & Wollheim 1989",
        "rationale": (
            "Calcinosis is predominantly a feature of lcSSc (formerly CREST "
            "syndrome) rather than dcSSc. The CREST acronym itself was developed "
            "around lcSSc patients, and calcinosis prevalence in lcSSc (~30%) "
            "substantially exceeds that in dcSSc (~18%)."
        ),
    },
    {
        "name":    "Osteoporosis => Calcinosis",
        "a_col":   "osteoporosis",
        "a_fn":    lambda x: x == 1,
        "b_col":   "calcinosis",
        "claim":   "Osteoporosis implies calcinosis",
        "source":  "Valenzuela et al. 2016 (SCTC, OR=4.2); novel finding",
        "rationale": (
            "The association between osteoporosis and calcinosis was a novel "
            "finding of the SCTC study (OR=4.2, 95% CI: 2.3-7.9, p<0.0001). "
            "This may reflect shared pathophysiology involving dysregulated bone "
            "and calcium metabolism, with calcium mobilised from bone being "
            "deposited in soft tissues."
        ),
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def pac_bound(n, k, alpha=ALPHA):
    """One-sided Clopper-Pearson upper confidence bound on violation probability."""
    if n == 0:
        return None
    if k == 0:
        return 1 - (alpha ** (1.0 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def chi_squared_test(a_vec, b_vec):
    """Pearson chi-squared test and Phi coefficient for 2x2 table."""
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
    """
    Run PC algorithm on antecedent + outcome + confounders.
    Returns edge presence and direction between a_col and calcinosis.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cols = confounders + [a_col, "calcinosis"]
        sub  = df[cols].dropna().values.astype(float)
        cg   = pc(sub, alpha=alpha, indep_test=fisherz, show_progress=False)
        g    = cg.G
        a_idx = cols.index(a_col)
        b_idx = cols.index("calcinosis")
        has_edge    = bool(g.graph[a_idx, b_idx] != 0 or
                          g.graph[b_idx, a_idx] != 0)
        directed_ab = bool(g.graph[a_idx, b_idx] == -1 and
                          g.graph[b_idx, a_idx] ==  1)
        return {
            "has_edge":        has_edge,
            "directed_A_to_B": directed_ab,
            "note":            "PC algorithm, Fisher-Z independence test",
        }
    except ImportError:
        return {"has_edge": None, "note": "causal-learn not installed"}
    except Exception as e:
        return {"has_edge": None, "error": str(e)}


def invariance_test(df, a_vec, b_col, group_col, min_size=10):
    """
    Test whether the implication holds across subgroups of group_col.
    Returns per-group violation rates and an overall invariance verdict.
    """
    results = {}
    for grp_val in sorted(df[group_col].unique()):
        mask  = df[group_col] == grp_val
        a1    = b_vec_from_mask(df, a_vec, mask)
        b1    = df.loc[mask & (a_vec == 1), b_col]
        n     = len(b1)
        if n < min_size:
            results[f"{group_col}={grp_val}"] = {"n": n, "skipped": True}
            continue
        k   = int((b1 == 0).sum())
        pac = pac_bound(n, k)
        results[f"{group_col}={grp_val}"] = {
            "n_a1":         n,
            "n_violations": k,
            "violation_rate": round(k / n, 3),
            "pac_bound":    round(pac, 4) if pac else None,
        }
    any_viol = any(
        v.get("n_violations", 0) > 0
        for v in results.values() if not v.get("skipped")
    )
    return {"invariant": not any_viol, "groups": results}


def b_vec_from_mask(df, a_vec, mask):
    """Helper: subset a_vec by mask."""
    return a_vec[mask]


def rosenbaum_gamma(n, k, alpha=ALPHA):
    """Minimum confounding odds ratio needed to explain away violations."""
    pac = pac_bound(n, k, alpha)
    if pac is None or pac >= 1.0:
        return None
    return round(pac / (1.0 - pac), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

CONFOUNDERS = ["age", "sex", "disease_duration", "mrss"]

all_results = {}

for imp in IMPLICATIONS:
    name  = imp["name"]
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    b_vec = df[imp["b_col"]]

    print(f"\n{'='*65}")
    print(f"Testing: {name}")
    print(f"Claim:   {imp['claim']}")
    print(f"{'='*65}")

    # Contingency table
    ct = pd.crosstab(a_vec, b_vec,
                     rownames=["A (antecedent)"],
                     colnames=["B (calcinosis)"])
    print(f"\nContingency table:\n{ct}\n")

    a1       = b_vec[a_vec == 1]
    n_a1     = len(a1)
    n_a0     = int((a_vec == 0).sum())
    n_viol   = int((a1 == 0).sum())
    viol_rate = n_viol / n_a1 if n_a1 > 0 else None

    # 1. PAC bound
    pac = pac_bound(n_a1, n_viol)
    print(f"N(A=1): {n_a1}  Violations: {n_viol}  Rate: {viol_rate:.3f}")
    print(f"PAC bound on P(violation) < {pac:.4f}  [95% CI]")

    # 2. Chi-squared
    chi = chi_squared_test(a_vec, b_vec)
    print(f"Chi-squared: chi2={chi['chi2']}  p={chi['p_value']:.6f}  phi={chi['phi']}")

    # 3. Causal discovery
    print("Running causal discovery (PC algorithm)...")
    causal = causal_discovery(df, imp["a_col"], CONFOUNDERS)
    print(f"Causal edge: {causal}")

    # 4. Invariance test across sex
    inv = invariance_test(df, a_vec, imp["b_col"], "sex")
    print(f"Invariant across sex: {inv['invariant']}")
    for grp, grp_res in inv["groups"].items():
        if not grp_res.get("skipped"):
            print(f"  {grp}: n={grp_res['n_a1']}, "
                  f"violations={grp_res['n_violations']}, "
                  f"rate={grp_res['violation_rate']}, "
                  f"PAC={grp_res['pac_bound']}")

    # 5. Rosenbaum sensitivity
    gamma = rosenbaum_gamma(n_a1, n_viol)
    print(f"Rosenbaum Gamma: {gamma}")

    # Verdict
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

# Save JSON
with open("results/calcinosis_implication_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nSaved results/calcinosis_implication_results.json")

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
    "ACA positive\n=> Calcinosis",
    "Digital Ulcers\n=> Calcinosis",
    "lcSSc subtype\n=> Calcinosis",
    "Osteoporosis\n=> Calcinosis",
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
bars = ax.bar(x - 0.2, rates,  width=0.35, color=colors, alpha=0.9,
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
ax.set_title("Calcinosis Implication Violation Rates and PAC Bounds\n"
             "Systemic Sclerosis Synthetic Cohort (n=800, calibrated to SCTC 2016)",
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
plt.savefig("figures/pac_bounds_calcinosis.png", dpi=150)
plt.close()
print("Saved figures/pac_bounds_calcinosis.png")

# ── Figure 2: Contingency bar charts ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 4))
fig.suptitle("Calcinosis Prevalence by Antecedent Status", fontsize=12)

for ax, imp, short in zip(axes, IMPLICATIONS, SHORT_LABELS):
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    ct    = pd.crosstab(a_vec, df["calcinosis"])
    ct.index = ["A=0\n(absent)", "A=1\n(present)"]
    ct.columns = ["No calcinosis", "Calcinosis"]
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
plt.savefig("figures/contingency_calcinosis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/contingency_calcinosis.png")

# ── Figure 3: Sex-stratified violation rates ──────────────────────────────────
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 5), sharey=True)
fig.suptitle("Calcinosis Implication Violation Rates by Sex\n"
             "(Female = sex=1, Male = sex=0)", fontsize=11)

SEX_COLORS = {
    "Female\n(sex=1)": "#e91e8c",
    "Male\n(sex=0)":   "#1e88e5",
    "Overall":         "#607d8b",
}

for ax, imp, short in zip(axes, IMPLICATIONS, SHORT_LABELS):
    a_vec = imp["a_fn"](df[imp["a_col"]]).astype(int)
    res   = all_results[imp["name"]]

    group_labels, viol_rates, pac_ups, bar_colors = [], [], [], []

    for sex_val, sex_label in [(1, "Female\n(sex=1)"), (0, "Male\n(sex=0)")]:
        mask = (df["sex"] == sex_val) & (a_vec == 1)
        sub_b = df.loc[mask, "calcinosis"]
        n = len(sub_b)
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

    xp = np.arange(len(group_labels))
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
plt.savefig("figures/sex_stratified_calcinosis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/sex_stratified_calcinosis.png")

# ── Figure 4: Summary table ───────────────────────────────────────────────────
rows = []
for name, res in all_results.items():
    rows.append({
        "Implication":  name,
        "N(A=1)":       res["n_a1"],
        "Violations":   res["n_violations"],
        "Viol. rate":   f"{res['violation_rate']:.2f}",
        "PAC bound":    f"{res['pac_bound']:.3f}",
        "Chi-sq p":     f"{res['chi_squared']['p_value']:.4f}",
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

ax.set_title("Calcinosis Implication Testing Summary — CREST Paper 01",
             fontsize=13, pad=16, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/summary_table_calcinosis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/summary_table_calcinosis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("CALCINOSIS IMPLICATION TESTING — FINAL SUMMARY")
print("="*65)
for name, res in all_results.items():
    chi_p = res['chi_squared']['p_value']
    print(f"\n{name}")
    print(f"  N(A=1)={res['n_a1']}  violations={res['n_violations']}  "
          f"rate={res['violation_rate']:.3f}  PAC<{res['pac_bound']:.3f}")
    print(f"  chi2 p={chi_p:.4f}  phi={res['chi_squared']['phi']:.3f}  "
          f"causal={res['causal'].get('has_edge','?')}  "
          f"invariant={res['invariance_sex'].get('invariant','?')}  "
          f"gamma={res['rosenbaum_gamma']}")
    print(f"  VERDICT: {res['verdict']}")

print("\nAll outputs saved to results/ and figures/")
print("GitHub repo: https://github.com/glenritschel/crest-implication-testing")
