# CREST Implication Testing Series

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Statistical implication testing framework applied to the five clinical manifestations of **CREST syndrome** (Calcinosis, Raynaud phenomenon, Esophageal dysmotility, Sclerodactyly, Telangiectasia) in Systemic Sclerosis (SSc).

This series extends the PAC-bound + causal discovery + invariance testing framework introduced in:

> Ritschel, G. & Claude (Anthropic). (2026). *Beyond Significance: A Framework for Statistically Evaluating Logical Implications from Observational Data.* Zenodo. https://doi.org/10.5281/zenodo.19014745

---

## The CREST Series

| Paper | Letter | Manifestation | Script | Status |
|-------|--------|--------------|--------|--------|
| 01 | **C** | Calcinosis | `paper_C_calcinosis/` | Complete |
| 02 | **R** | Raynaud phenomenon | `paper_R_raynaud/` | Complete |
| 03 | **E** | Esophageal dysmotility | `paper_E_esophageal/` | Complete |
| 04 | **S** | Sclerodactyly | `paper_S_sclerodactyly/` | Complete |
| 05 | **T** | Telangiectasia | `paper_T_telangiectasia/` | Complete |

---

## Approach

Each paper uses a **synthetic cohort** calibrated to published clinical statistics from large SSc registries (EUSTAR, SCTC, VEDOSS, Canadian Scleroderma Research Group). This approach ensures full reproducibility without requiring institutional data access.

Each pipeline tests four clinically motivated implications of the form **A ⇒ B** using five complementary methods:

| Method | Purpose |
|--------|---------|
| PAC violation bounds | Clopper-Pearson 95% upper bound on P(violation) |
| Chi-squared + Phi | Statistical association strength |
| PC algorithm (Fisher-Z) | Causal graph discovery |
| Invariance testing | Does the implication hold equally across sexes? |
| Rosenbaum sensitivity | Minimum confounding OR to explain away violations |

**Verdict criteria:**

| Verdict | Criteria |
|---------|---------|
| STRONG | Violation rate <30% AND chi-sq significant AND causal edge AND sex-invariant |
| MODERATE | Violation rate <30% AND chi-sq significant AND (causal OR invariant) |
| WEAK | Violation rate <30% AND chi-sq significant |
| TRENDING | Chi-sq significant only |
| REJECTED | Not chi-sq significant |

---

## Repository Structure

```
crest-implication-testing/
├── README.md
├── requirements.txt
├── paper_C_calcinosis/
│   ├── calcinosis_implication_test.py
│   ├── results/          (generated)
│   └── figures/          (generated)
├── paper_R_raynaud/
│   ├── raynaud_implication_test.py
│   ├── results/
│   └── figures/
├── paper_E_esophageal/
│   ├── esophageal_implication_test.py
│   ├── results/
│   └── figures/
├── paper_S_sclerodactyly/
│   ├── sclerodactyly_implication_test.py
│   ├── results/
│   └── figures/
└── paper_T_telangiectasia/
    ├── telangiectasia_implication_test.py
    ├── results/
    └── figures/
```

---

## Usage

### Install dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn causal-learn
```

### Run a single paper

```bash
cd paper_C_calcinosis
python3 calcinosis_implication_test.py
```

Each script runs in under 60 seconds on CPU. Outputs are written to `results/` and `figures/` subdirectories.

### Run all papers

```bash
for dir in paper_C_calcinosis paper_R_raynaud paper_E_esophageal paper_S_sclerodactyly paper_T_telangiectasia; do
    echo "Running $dir..."
    cd $dir && python3 *.py && cd ..
done
```

---

## Key Findings Summary

### C — Calcinosis
Calibrated to Valenzuela et al. 2016 (SCTC, n=5218).

| Implication | Verdict |
|------------|---------|
| Anticentromere Ab ⇒ Calcinosis | MODERATE |
| Digital ulcers ⇒ Calcinosis | MODERATE |
| lcSSc ⇒ Calcinosis | MODERATE |
| Osteoporosis ⇒ Calcinosis | MODERATE |

### R — Raynaud Phenomenon
Calibrated to VEDOSS 2021 (n=764) and Hughes & Herrick 2020.

| Implication | Verdict |
|------------|---------|
| Abnormal capillaroscopy ⇒ Raynaud | MODERATE |
| Late capillaroscopy ⇒ Severe RP | WEAK |
| SSc autoantibody ⇒ Raynaud | MODERATE |
| Digital ulcers ⇒ Severe RP | REJECTED |

### E — Esophageal Dysmotility
Calibrated to Roman et al. 2011, Ebert et al. 2012.

| Implication | Verdict |
|------------|---------|
| dcSSc ⇒ Esophageal dysmotility | REJECTED* |
| Anti-Scl70 ⇒ Esophageal dysmotility | MODERATE |
| Dysphagia ⇒ Esophageal dysmotility | REJECTED* |
| Esophageal dysmotility ⇒ PPI-refractory GERD | REJECTED |

*Low violation rate (<20%) but fails causal + invariance jointly.

### S — Sclerodactyly
Calibrated to Khanna et al. 2017 (mRSS), EUSTAR database.

| Implication | Verdict |
|------------|---------|
| dcSSc ⇒ Sclerodactyly | REJECTED* |
| Anti-Scl70 ⇒ Sclerodactyly | REJECTED* |
| High mRSS (>14) ⇒ Sclerodactyly | REJECTED* |
| Sclerodactyly ⇒ Digital pitting scars | REJECTED |

*Very low violation rates (0–4%) but causal structure not recoverable — Scl70 and dcSSc are co-linear predictors of sclerodactyly, making causal direction unidentifiable.

### T — Telangiectasia
Calibrated to Zhang et al. 2015 (EUSTAR China), DETECT algorithm.

| Implication | Verdict |
|------------|---------|
| lcSSc ⇒ Telangiectasia | TRENDING |
| ACA ⇒ Telangiectasia | TRENDING |
| Disease duration >10yr ⇒ Telangiectasia | TRENDING |
| Telangiectasia ⇒ PAH | REJECTED |

---

## Interpretation

A key finding across the CREST series is that **statistical significance and low violation rates do not guarantee a strong implication**. Several implications (e.g., dcSSc ⇒ Sclerodactyly, Scl70 ⇒ Sclerodactyly) have near-zero violation rates but fail the causal and invariance criteria because the predictors are collinear. This illustrates the value of the multi-component framework over simple association testing.

The telangiectasia paper illustrates a different problem: telangiectasia is an **ensemble marker** — no single clinical feature strongly implies its presence, yet it emerges from the combination of lcSSc, ACA, and long disease duration. This has implications for clinical decision rules.

---

## Calibration Sources

| Paper | Primary source |
|-------|---------------|
| Calcinosis | Valenzuela et al. 2016, Semin Arthritis Rheum 46:344–349 |
| Raynaud | VEDOSS registry, Lancet Rheumatol 2021; Hughes & Herrick, Nat Rev Rheumatol 2020 |
| Esophageal | Roman et al. 2011, Neurogastroenterol Motil; Ebert et al. 2012, J Rheumatol |
| Sclerodactyly | Khanna et al. 2017, J Scleroderma Relat Disord; EUSTAR database |
| Telangiectasia | Zhang et al. 2015, Clin Exp Rheumatol; Coghlan et al. 2014 (DETECT) |

---

## Citation

```bibtex
@misc{ritschel_claude_2026_crest,
  author    = {Ritschel, Glen and Claude (Anthropic)},
  title     = {CREST Implication Testing Series: Statistical Evaluation
               of Clinical Implications in Systemic Sclerosis},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## Related

- **Heart disease implication testing** (framework paper):
  https://github.com/glenritschel/vae-implication-testing
  DOI: https://doi.org/10.5281/zenodo.19014745

- **Calcinosis scRNA-seq pipeline** (drug repurposing):
  https://github.com/glenritschel/calcinosis-crest

---

## Authors

Glen Ritschel (research direction) and Claude / claude-sonnet-4-6 (Anthropic)
(methodology, code, analysis, writing)
