# CREST Implication Testing Series

**Statistical evaluation of clinical implications in systemic sclerosis using the PAC-bound framework**

[![Paper 01 Calcinosis](https://zenodo.org/badge/DOI/10.5281/zenodo.19038416.svg)](https://doi.org/10.5281/zenodo.19038416)
[![Paper 02 Raynaud](https://zenodo.org/badge/DOI/10.5281/zenodo.19038798.svg)](https://doi.org/10.5281/zenodo.19038798)
[![Paper 03 Esophageal](https://zenodo.org/badge/DOI/10.5281/zenodo.19038921.svg)](https://doi.org/10.5281/zenodo.19038921)
[![Paper 04 Sclerodactyly](https://zenodo.org/badge/DOI/10.5281/zenodo.19039019.svg)](https://doi.org/10.5281/zenodo.19039019)
[![Paper 05 Telangiectasia](https://zenodo.org/badge/DOI/10.5281/zenodo.19039138.svg)](https://doi.org/10.5281/zenodo.19039138)
[![Framework paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19014745.svg)](https://doi.org/10.5281/zenodo.19014745)

---

## Overview

This repository contains the code, synthetic cohort generation scripts, and results for a five-paper series systematically evaluating the implication strength of clinical features in systemic sclerosis (SSc) / CREST syndrome.

Clinical guidelines and reviews frequently state that certain features of SSc *imply* each other — that ACA positivity implies calcinosis, that dcSSc implies sclerodactyly, that telangiectasia implies PAH. These are typically justified by statistically significant associations (odds ratios, chi-squared tests). But **association is not implication**.

This series applies the **PAC-bound implication testing framework** ([Ritschel & Claude 2026](https://doi.org/10.5281/zenodo.19014745)) to each of the five CREST features, asking: is the clinical association strong enough to constitute a statistical implication — that is, does the antecedent predict the consequent with a violation rate below 30%?

---

## The Five Papers

| # | Feature | Title | Verdict summary | DOI |
|---|---------|-------|-----------------|-----|
| 01 | **C** — Calcinosis | *When Association Is Not Implication* | 3× TRENDING, 1× REJECTED | [10.5281/zenodo.19038416](https://doi.org/10.5281/zenodo.19038416) |
| 02 | **R** — Raynaud Phenomenon | *Near-Universal but Not Implied* | 2× MODERATE, 1× WEAK, 1× REJECTED | [10.5281/zenodo.19038798](https://doi.org/10.5281/zenodo.19038798) |
| 03 | **E** — Esophageal Dysmotility | *Symptomatic but Not Sufficient* | 1× MODERATE, 3× REJECTED | [10.5281/zenodo.19038921](https://doi.org/10.5281/zenodo.19038921) |
| 04 | **S** — Sclerodactyly | *The Collinearity Problem* | 4× REJECTED (3 near-zero viol.) | [10.5281/zenodo.19039019](https://doi.org/10.5281/zenodo.19039019) |
| 05 | **T** — Telangiectasia | *An Ensemble Marker* | 3× TRENDING, 1× REJECTED | [10.5281/zenodo.19039138](https://doi.org/10.5281/zenodo.19039138) |

---

## Full Results

| Implication | N(A=1) | Viol. rate | PAC bound | Chi-sq p | Phi | Causal | Invariant | Γ* | Verdict |
|-------------|--------|-----------|-----------|----------|-----|--------|-----------|-----|---------|
| ACA ⇒ Calcinosis | 258 | 0.643 | <0.693 | 0.0001 | 0.142 | Yes | No | 2.26 | TRENDING |
| Digital ulcers ⇒ Calcinosis | 255 | 0.549 | <0.602 | <0.0001 | 0.288 | Yes | No | 1.51 | REJECTED |
| lcSSc ⇒ Calcinosis | 482 | 0.691 | <0.726 | 0.0005 | 0.124 | Yes | No | 2.64 | TRENDING |
| Osteoporosis ⇒ Calcinosis | 42 | 0.405 | <0.543 | <0.0001 | 0.171 | Yes | No | 1.19 | TRENDING |
| Abnormal cap ⇒ Raynaud | 624 | 0.018 | <0.029 | 0.0250 | 0.079 | Yes | No | 0.030 | MODERATE |
| Late cap ⇒ Severe RP | 310 | 0.297 | <0.342 | <0.0001 | 0.149 | No | No | 0.521 | WEAK |
| SSc autoantibody ⇒ Raynaud | 382 | 0.005 | <0.016 | 0.0014 | 0.113 | Yes | No | 0.017 | MODERATE |
| Digital ulcers ⇒ Severe RP | 237 | 0.219 | <0.268 | <0.0001 | 0.224 | Yes | No | 0.367 | REJECTED |
| dcSSc ⇒ Esoph. dysmotility | 312 | 0.114 | <0.148 | <0.0001 | 0.225 | Yes | No | 0.172 | REJECTED |
| Scl70 ⇒ Esoph. dysmotility | 215 | 0.135 | <0.194 | <0.0001 | 0.207 | Yes | No | 0.222 | MODERATE |
| Dysphagia ⇒ Esoph. dysmotility | 321 | 0.167 | <0.209 | <0.0001 | 0.230 | Yes | No | 0.251 | REJECTED |
| Esoph. dysmotility ⇒ PPI-refractory | 568 | 0.370 | <0.404 | <0.0001 | 0.210 | No | No | 0.587 | REJECTED |
| dcSSc ⇒ Sclerodactyly | 312 | 0.029 | <0.051 | <0.0001 | 0.258 | No | No | 0.052 | REJECTED |
| Scl70 ⇒ Sclerodactyly | 193 | 0.000 | <0.022 | <0.0001 | 0.200 | No | No | n/a | REJECTED |
| High mRSS ⇒ Sclerodactyly | 278 | 0.036 | <0.064 | <0.0001 | 0.227 | No | No | 0.066 | REJECTED |
| Sclerodactyly ⇒ Digital pitting | 665 | 0.355 | <0.387 | <0.0001 | 0.245 | Yes | No | 0.550 | REJECTED |
| lcSSc ⇒ Telangiectasia | 482 | 0.375 | <0.413 | 0.0008 | 0.115 | No | No | 0.600 | TRENDING |
| ACA ⇒ Telangiectasia | 258 | 0.305 | <0.356 | <0.0001 | 0.160 | No | No | 0.521 | TRENDING |
| Long duration ⇒ Telangiectasia | 382 | 0.369 | <0.418 | 0.0231 | 0.080 | No | No | 0.586 | TRENDING |
| Telangiectasia ⇒ PAH | 463 | 0.861 | <0.887 | 0.3241 | 0.037 | No | No | n/a | REJECTED |

---

## Verdict Key

| Verdict | Criteria |
|---------|----------|
| **STRONG** | Violation rate <30%, chi-sq significant, causal edge confirmed, sex-invariant |
| **MODERATE** | Violation rate <30%, chi-sq significant, causal edge confirmed OR sex-invariant |
| **WEAK** | Violation rate <30%, chi-sq significant |
| **TRENDING** | Chi-sq significant, violation rate ≥30% |
| **REJECTED** | Not chi-sq significant, or violation rate above threshold with no causal support |

---

## Key Findings

- **No implication achieved STRONG** across all 20 tested in the series
- **Two reached MODERATE**: abnormal capillaroscopy ⇒ Raynaud (0.5% violation) and SSc autoantibody ⇒ Raynaud (1.8% violation) — the only near-sufficient clinical predictors in SSc
- **The collinearity problem** (Paper 04): dcSSc, Scl70, and high mRSS are so co-linear that constraint-based causal discovery cannot orient edges between them, producing REJECTED verdicts despite near-zero violation rates
- **Telangiectasia ⇒ PAH** has an 86% violation rate — PAH affects only ~14% of telangiectasia-positive patients, supporting multi-feature screening algorithms (DETECT) over single-marker triage
- **Sex invariance failed** for every implication — male SSc patients consistently show higher violation rates than female patients

---

## Repository Structure

```
crest-implication-testing/
├── paper_C_calcinosis/
│   └── calcinosis_implication_test.py
├── paper_R_raynaud/
│   └── raynaud_implication_test.py
├── paper_E_esophageal/
│   └── esophageal_implication_test.py
├── paper_S_sclerodactyly/
│   └── sclerodactyly_implication_test.py
├── paper_T_telangiectasia/
│   └── telangiectasia_implication_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproducing the Results

```bash
git clone https://github.com/glenritschel/crest-implication-testing
cd crest-implication-testing
pip install -r requirements.txt

python paper_C_calcinosis/calcinosis_implication_test.py
python paper_R_raynaud/raynaud_implication_test.py
python paper_E_esophageal/esophageal_implication_test.py
python paper_S_sclerodactyly/sclerodactyly_implication_test.py
python paper_T_telangiectasia/telangiectasia_implication_test.py
```

Each script generates results in `results/` and figures in `figures/` within its directory.

---

## Dependencies

See `requirements.txt`. Core dependencies:

- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `causal-learn` (PC algorithm)
- `matplotlib`, `seaborn`

---

## Citation

If you use this code or results, please cite the individual paper(s) and the framework:

**Framework:**
> Ritschel, G. & Claude (Anthropic). (2026). *Beyond Significance: A Framework for Statistically Evaluating Logical Implications from Observational Data.* Zenodo. https://doi.org/10.5281/zenodo.19014745

**Individual papers** (use the DOI for the relevant CREST feature from the table above).

**BibTeX (framework):**
```bibtex
@misc{ritschel2026beyond,
  author    = {Ritschel, Glen and {Claude (Anthropic)}},
  title     = {Beyond Significance: A Framework for Statistically Evaluating
               Logical Implications from Observational Data},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19014745},
  url       = {https://doi.org/10.5281/zenodo.19014745}
}
```

---

## Authors

- **Glen Ritschel** — Independent Researcher
- **Claude (Anthropic)** — Anthropic, San Francisco, CA

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

