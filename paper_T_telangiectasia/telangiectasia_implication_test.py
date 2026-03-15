"""
telangiectasia_implication_test.py
------------------------------------
Statistical implication testing — Telangiectasia in SSc (T in CREST).
Paper 05 in the CREST Implication Testing Series.

Calibration:
  - Zhang et al. (EUSTAR China 2015): 41.7% telangiectasia in SSc
  - Telangiectasia assoc with: DUs (40.6% vs 23.1%), RP (97.9% vs 90.3%),
    PAH (25.0% vs 14.2%), longer disease duration
  - ACA associated with lcSSc and telangiectasia (DETECT algorithm uses ACA+Tel)
  - PAH prevalence 6.4% overall; higher with telangiectasia
  - lcSSc prevalence of telangiectasia: ~50%; dcSSc: ~35%

Four implications:
  1. lcSSc subtype           => Telangiectasia
  2. ACA positive            => Telangiectasia
  3. Long disease duration (>10yr) => Telangiectasia
  4. Telangiectasia          => Pulmonary arterial hypertension (PAH)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

np.random.seed(42)
N = 800

def generate_cohort(n, seed=42):
    rng = np.random.default_rng(seed)
    sex              = rng.binomial(1, 0.86, n)
    age              = rng.normal(54, 13, n).clip(18, 85)
    disease_duration = rng.exponential(10, n).clip(0.5, 40)
    lc_ssc           = rng.binomial(1, np.where(sex==1, 0.63, 0.50))

    p_aca   = np.where(lc_ssc==1, 0.48, 0.08); aca = rng.binomial(1, p_aca)
    p_scl70 = np.where(lc_ssc==1, 0.06, 0.42); scl70 = rng.binomial(1, p_scl70)

    long_duration = (disease_duration > 10).astype(int)

    # Digital ulcers
    p_du = 0.18 + 0.18*(lc_ssc==0) + 0.005*disease_duration.clip(0,20)
    digital_ulcers = rng.binomial(1, p_du.clip(0.05, 0.65))

    # Telangiectasia: ~42% overall; more in lcSSc + longer duration + ACA
    # Zhang EUSTAR 2015: no difference by autoantibody profile (p>0.05)
    # but more with longer disease duration
    log_odds_tel = (
        -0.50
        + 0.35*(lc_ssc==1)        # slightly more in lcSSc
        + 0.50*aca                 # ACA associated (DETECT algorithm)
        + 0.04*disease_duration.clip(0,20)  # longer duration
        + 0.40*digital_ulcers      # vascular disease marker
    )
    p_tel = 1/(1+np.exp(-log_odds_tel))
    telangiectasia = rng.binomial(1, p_tel)

    # PAH: ~6-8% overall; higher with telangiectasia + lcSSc + ACA + older age
    # Australian SSc cohort: 11.9% PAH (Morrisroe 2017)
    log_odds_pah = (
        -5.50                       # baseline ~3%
        + 0.80*telangiectasia       # key predictor (DETECT algorithm)
        + 0.60*aca                  # OR ~2 for PAH
        + 0.50*(lc_ssc==1)         # PAH more common in lcSSc
        + 0.04*age.clip(40,80)     # older age
        + 0.03*disease_duration.clip(0,20)
    )
    p_pah = 1/(1+np.exp(-log_odds_pah))
    pah = rng.binomial(1, p_pah)

    return pd.DataFrame({
        "sex": sex, "age": age.round(1), "disease_duration": disease_duration.round(1),
        "lc_ssc": lc_ssc, "aca": aca, "scl70": scl70,
        "long_duration": long_duration, "digital_ulcers": digital_ulcers,
        "telangiectasia": telangiectasia, "pah": pah,
    })

df = generate_cohort(N)
print(f"n={N}  telangiectasia={df.telangiectasia.mean():.1%}  "
      f"lcSSc_tel={df.loc[df.lc_ssc==1,'telangiectasia'].mean():.1%}  "
      f"dcSSc_tel={df.loc[df.lc_ssc==0,'telangiectasia'].mean():.1%}  "
      f"PAH={df.pah.mean():.1%}")
df.to_csv("results/synthetic_ssc_telangiectasia_cohort.csv", index=False)

IMPLICATIONS = [
    {"name":"lcSSc => Telangiectasia", "a_col":"lc_ssc", "a_fn":lambda x: x==1, "b_col":"telangiectasia",
     "claim":"Limited cutaneous SSc implies telangiectasia",
     "source":"Zhang et al. 2015 (EUSTAR China); Clements & Furst 2003",
     "rationale":"Telangiectasias are classically associated with lcSSc (the former CREST syndrome). The Zhang EUSTAR study (2015) showed telangiectasia in Chinese SSc patients was predominantly a feature of longer disease duration and vascular disease. lcSSc patients have a higher prevalence (~50%) than dcSSc (~35%), consistent with the original CREST description."},
    {"name":"ACA => Telangiectasia", "a_col":"aca", "a_fn":lambda x: x==1, "b_col":"telangiectasia",
     "claim":"Anticentromere antibody implies telangiectasia",
     "source":"DETECT algorithm (Coghlan 2014); Hoeper et al. 2013",
     "rationale":"ACA is a component of the DETECT screening algorithm for SSc-PAH, alongside telangiectasia, reflecting their strong co-occurrence. ACA-positive patients have the classic lcSSc phenotype with prominent telangiectasias. The DETECT algorithm uses ACA and telangiectasia as complementary vascular markers."},
    {"name":"LongDuration => Telangiectasia", "a_col":"long_duration", "a_fn":lambda x: x==1, "b_col":"telangiectasia",
     "claim":"Disease duration >10 years implies telangiectasia",
     "source":"Zhang et al. 2015 (EUSTAR): p<0.05 for disease duration); Herrick 2012",
     "rationale":"The Zhang EUSTAR study specifically identified longer disease duration from both RP onset and first non-RP manifestation as significantly associated with telangiectasia (p<0.05). This reflects the progressive vascular remodelling that produces telangiectasias over time as a late manifestation of microvascular disease."},
    {"name":"Telangiectasia => PAH", "a_col":"telangiectasia", "a_fn":lambda x: x==1, "b_col":"pah",
     "claim":"Telangiectasia implies pulmonary arterial hypertension",
     "source":"DETECT algorithm (Coghlan 2014); Humbert et al. 2019; Morrisroe 2017",
     "rationale":"Telangiectasia is a formal component of the DETECT PAH screening algorithm for SSc, reflecting its well-validated association with SSc-PAH (OR ~2.0). Humbert et al. confirmed telangiectasia as a clinical predictor of SSc-PAH. The underlying mechanism is shared pathophysiology of aberrant vasodilation and vascular remodeling affecting both skin and pulmonary vasculature."},
]

ALPHA = 0.05
VERDICT_COLORS = {"STRONG":"#2ecc71","MODERATE":"#3498db","WEAK":"#f39c12","TRENDING":"#e67e22","REJECTED":"#e74c3c"}
SHORT_LABELS = ["lcSSc\n=> Telangiect","ACA\n=> Telangiect","Long duration\n=> Telangiect","Telangiect\n=> PAH"]

def pac_bound(n,k,alpha=ALPHA):
    if n==0: return None
    if k==0: return 1-(alpha**(1.0/n))
    return float(stats.beta.ppf(1-alpha,k+1,n-k))

def chi2_phi(a,b):
    ct=pd.crosstab(a,b)
    if ct.shape!=(2,2): return {"chi2":None,"p_value":1.0,"phi":0.0}
    c2,p,_,_=stats.chi2_contingency(ct)
    return {"chi2":round(float(c2),3),"p_value":round(float(p),6),"phi":round(float(np.sqrt(c2/len(a))),3)}

def causal_edge(df,a_col,confounders,b_col):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cols=confounders+[a_col,b_col]
        sub=df[cols].dropna().values.astype(float)
        cg=pc(sub,alpha=0.05,indep_test=fisherz,show_progress=False)
        ai,bi=cols.index(a_col),len(cols)-1
        return {"has_edge":bool(cg.G.graph[ai,bi]!=0 or cg.G.graph[bi,ai]!=0)}
    except: return {"has_edge":None}

def invar(df,a_vec,b_col):
    r={}
    for g in sorted(df["sex"].unique()):
        m=(df["sex"]==g)&(a_vec==1); s=df.loc[m,b_col]; n=len(s)
        if n<10: r[f"sex={g}"]={"skipped":True}; continue
        k=int((s==0).sum()); p=pac_bound(n,k)
        r[f"sex={g}"]={"n_a1":n,"n_violations":k,"violation_rate":round(k/n,3),"pac_bound":round(p,4) if p else None}
    return {"invariant":not any(v.get("n_violations",0)>0 for v in r.values() if not v.get("skipped")),"groups":r}

all_results={}; CONF=["age","sex","disease_duration"]

for imp in IMPLICATIONS:
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int); b_vec=df[imp["b_col"]]
    a1=b_vec[a_vec==1]; n_a1=len(a1); n_viol=int((a1==0).sum())
    vr=n_viol/n_a1 if n_a1>0 else None
    pac=pac_bound(n_a1,n_viol); chi=chi2_phi(a_vec,b_vec)
    ca=causal_edge(df,imp["a_col"],CONF,imp["b_col"]); inv=invar(df,a_vec,imp["b_col"])
    gamma=round(pac/(1-pac),3) if pac and pac<1 else None
    chi_sig=(chi["p_value"] or 1)<ALPHA; low_viol=(vr or 1)<0.30
    causal_ok=ca.get("has_edge",False); invariant=inv.get("invariant",False)
    verdict=("STRONG" if (low_viol and chi_sig and causal_ok and invariant) else
             "MODERATE" if (low_viol and chi_sig and (causal_ok or invariant)) else
             "WEAK" if (low_viol and chi_sig) else
             "TRENDING" if chi_sig else "REJECTED")
    print(f"{imp['name']}: viol={vr:.3f} PAC<{pac:.3f} p={chi['p_value']:.4f} -> {verdict}")
    all_results[imp["name"]]={**{k:imp[k] for k in ["claim","source","rationale"]},
        "outcome":imp["b_col"],"n_a1":n_a1,"n_violations":n_viol,
        "violation_rate":round(vr,3) if vr else None,"pac_bound":round(pac,4) if pac else None,
        "chi_squared":chi,"causal":ca,"invariance_sex":inv,"rosenbaum_gamma":gamma,"verdict":verdict}

with open("results/telangiectasia_implication_results.json","w") as f:
    json.dump(all_results,f,indent=2,default=str)

# Figures
fig,ax=plt.subplots(figsize=(11,5))
ns=list(all_results.keys()); rates=[all_results[n]["violation_rate"] or 0 for n in ns]
bounds=[all_results[n]["pac_bound"] or 0 for n in ns]
colors=[VERDICT_COLORS.get(all_results[n]["verdict"],"#95a5a6") for n in ns]
x=np.arange(len(ns))
ax.bar(x-0.2,rates,width=0.35,color=colors,alpha=0.9,edgecolor="white",label="Observed violation rate")
ax.bar(x+0.2,bounds,width=0.35,color=colors,alpha=0.35,hatch="//",label="PAC upper bound (95% CI)")
ax.axhline(0.30,color="gray",linestyle="--",linewidth=1.2,label="30% threshold")
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS,fontsize=9)
ax.set_ylabel("Violation probability"); ax.set_ylim(0,0.85)
ax.set_title("Telangiectasia Implication Violation Rates\nSSc Synthetic Cohort (n=800, calibrated to Zhang 2015 & DETECT algorithm)",fontsize=11)
ax.legend(fontsize=9)
for i,(v,b,n) in enumerate(zip(rates,bounds,ns)):
    p=all_results[n]["chi_squared"]["p_value"] or 1
    sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    ax.text(i,b+0.025,f"χ² {sig}",ha="center",fontsize=8,color="dimgray")
    ax.text(x[i]-0.2,v+0.012,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold",color="white" if v>0.08 else "black")
plt.tight_layout(); plt.savefig("figures/pac_bounds_telangiectasia.png",dpi=150); plt.close()

fig,axes=plt.subplots(1,4,figsize=(14,4)); fig.suptitle("Telangiectasia Prevalence by Antecedent",fontsize=12)
for ax,imp,short in zip(axes,IMPLICATIONS,SHORT_LABELS):
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int)
    ct=pd.crosstab(a_vec,df[imp["b_col"]]); ct.index=["A=0","A=1"]
    ct_pct=ct.div(ct.sum(axis=1),axis=0)*100
    ct_pct.plot(kind="bar",ax=ax,color=["#3498db","#e74c3c"],edgecolor="white",width=0.6)
    ax.set_title(f"{short}\n[{all_results[imp['name']]['verdict']}]",fontsize=8)
    ax.set_ylabel("% patients" if ax==axes[0] else ""); ax.tick_params(axis="x",rotation=0)
    ax.legend(fontsize=7); ax.set_ylim(0,100)
plt.tight_layout(); plt.savefig("figures/contingency_telangiectasia.png",dpi=150,bbox_inches="tight"); plt.close()

SEX_COLORS={"Female\n(sex=1)":"#e91e8c","Male\n(sex=0)":"#1e88e5","Overall":"#607d8b"}
fig,axes=plt.subplots(1,4,figsize=(14,5),sharey=True); fig.suptitle("Telangiectasia Violation Rates by Sex",fontsize=11)
for ax,imp,short in zip(axes,IMPLICATIONS,SHORT_LABELS):
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int); b_col=imp["b_col"]; res=all_results[imp["name"]]
    gl,vr2,pu,bc=[],[],[],[]
    for sv,sl in [(1,"Female\n(sex=1)"),(0,"Male\n(sex=0)")]:
        m=(df["sex"]==sv)&(a_vec==1); s=df.loc[m,b_col]; nn=len(s)
        if nn<10: continue
        k=int((s==0).sum()); p=pac_bound(nn,k)
        gl.append(sl); vr2.append(k/nn); pu.append(p or k/nn); bc.append(SEX_COLORS[sl])
    gl.append("Overall"); vr2.append(res["violation_rate"] or 0); pu.append(res["pac_bound"] or 0); bc.append(SEX_COLORS["Overall"])
    xp=np.arange(len(gl)); bars=ax.bar(xp,vr2,color=bc,edgecolor="white",width=0.5,alpha=0.88)
    ax.errorbar(xp,vr2,yerr=[np.zeros(len(xp)),[max(0,p-v) for p,v in zip(pu,vr2)]],fmt="none",ecolor="black",capsize=5)
    ax.axhline(0.30,color="gray",linestyle="--",linewidth=1)
    ax.set_xticks(xp); ax.set_xticklabels(gl,fontsize=8); ax.set_title(short.replace("\n"," "),fontsize=8); ax.set_ylim(0,0.85)
    if ax==axes[0]: ax.set_ylabel("Violation rate")
    for bar,v in zip(bars,vr2): ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
fig.legend(handles=[Patch(facecolor=c,label=l) for l,c in SEX_COLORS.items()],
           loc="lower center",ncol=3,fontsize=9,bbox_to_anchor=(0.5,-0.05))
plt.tight_layout(); plt.savefig("figures/sex_stratified_telangiectasia.png",dpi=150,bbox_inches="tight"); plt.close()

rows=[{"Implication":n,"N(A=1)":r["n_a1"],"Violations":r["n_violations"],
       "Viol. rate":f"{r['violation_rate'] or 0:.2f}","PAC bound":f"{r['pac_bound'] or 0:.3f}",
       "Chi-sq p":f"{r['chi_squared']['p_value']:.4f}","Phi":f"{r['chi_squared']['phi']:.3f}",
       "Causal":str(r["causal"].get("has_edge","?")),"Invariant":str(r["invariance_sex"].get("invariant","?")),
       "Gamma":str(r["rosenbaum_gamma"]),"Verdict":r["verdict"]} for n,r in all_results.items()]
df_s=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(18,max(2,len(rows)*0.9+2)))
ax.axis("off"); tbl=ax.table(cellText=df_s.values,colLabels=df_s.columns,cellLoc="center",loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,2.0)
vc=df_s.columns.get_loc("Verdict")
for i,row in enumerate(rows):
    tbl[(i+1,vc)].set_facecolor(VERDICT_COLORS.get(row["Verdict"],"#fff"))
    tbl[(i+1,vc)].set_text_props(color="white",fontweight="bold")
for j in range(len(df_s.columns)): tbl[(0,j)].set_facecolor("#2c3e50"); tbl[(0,j)].set_text_props(color="white",fontweight="bold")
ax.set_title("Telangiectasia Implication Testing Summary — CREST Paper 05",fontsize=13,pad=16,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/summary_table_telangiectasia.png",dpi=150,bbox_inches="tight"); plt.close()
print("All telangiectasia figures saved.")
