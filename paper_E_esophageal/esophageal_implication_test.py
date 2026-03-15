"""
esophageal_implication_test.py
-------------------------------
Statistical implication testing — Esophageal dysmotility in SSc (E in CREST).
Paper 03 in the CREST Implication Testing Series.

Calibration sources:
  - Roman et al. 2011 (HRM study): 67% esophageal dysmotility overall;
    95% in dcSSc vs 59% in lcSSc
  - Ebert et al. 2012: dcSSc 95.4% abnormal ETS vs 58.5% lcSSc
  - Ling et al. PMC 2021: esophageal involvement ~90%
  - GI dysmotility associated with PPI-refractory esophagitis (Alcala 2024)

Four implications:
  1. dcSSc subtype         => Esophageal dysmotility
  2. Anti-Scl70 positive   => Esophageal dysmotility
  3. Dysphagia present     => Esophageal dysmotility
  4. Esophageal dysmotility => PPI-refractory symptoms (GERD)
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
    age              = rng.normal(52, 13, n).clip(18, 85)
    disease_duration = rng.exponential(9, n).clip(0.5, 40)
    lc_ssc           = rng.binomial(1, np.where(sex==1, 0.63, 0.50))

    # Autoantibodies
    p_aca   = np.where(lc_ssc==1, 0.48, 0.08)
    aca     = rng.binomial(1, p_aca)
    p_scl70 = np.where(lc_ssc==1, 0.06, 0.42)
    scl70   = rng.binomial(1, p_scl70)

    # Dysphagia: ~40% overall, higher in dcSSc
    p_dys = 0.28 + 0.20*(lc_ssc==0) + 0.005*disease_duration.clip(0,20)
    dysphagia = rng.binomial(1, p_dys.clip(0.15, 0.75))

    # Heartburn/reflux symptoms: ~70% overall
    p_hb = 0.60 + 0.12*(lc_ssc==0) + 0.004*disease_duration.clip(0,20)
    heartburn = rng.binomial(1, p_hb.clip(0.40, 0.90))

    # Esophageal dysmotility: 67% overall; 95% dcSSc, 59% lcSSc (Roman 2011)
    log_odds_ed = (
        -0.30
        + 1.50*(lc_ssc==0)    # dcSSc strong predictor
        + 0.70*scl70           # anti-Scl70 independent predictor
        + 0.80*dysphagia       # symptom predictor
        + 0.05*disease_duration.clip(0,20)
    )
    p_ed = 1/(1+np.exp(-log_odds_ed))
    esoph_dysmotility = rng.binomial(1, p_ed)

    # PPI-refractory GERD: ~33% of SSc GERD patients (Alcala 2024)
    # Requires esophageal dysmotility + GERD symptoms
    p_ppi = 0.15 + 0.40*esoph_dysmotility + 0.15*dysphagia + 0.08*(lc_ssc==0)
    ppi_refractory = rng.binomial(1, p_ppi.clip(0.05, 0.80))

    return pd.DataFrame({
        "sex": sex, "age": age.round(1), "disease_duration": disease_duration.round(1),
        "lc_ssc": lc_ssc, "aca": aca, "scl70": scl70,
        "dysphagia": dysphagia, "heartburn": heartburn,
        "esoph_dysmotility": esoph_dysmotility, "ppi_refractory": ppi_refractory,
    })

df = generate_cohort(N)
print(f"n={N}  esoph_dysmotility={df.esoph_dysmotility.mean():.1%}  "
      f"dcSSc_ed={df.loc[df.lc_ssc==0,'esoph_dysmotility'].mean():.1%}  "
      f"lcSSc_ed={df.loc[df.lc_ssc==1,'esoph_dysmotility'].mean():.1%}")
df.to_csv("results/synthetic_ssc_esophageal_cohort.csv", index=False)

IMPLICATIONS = [
    {"name":"dcSSc => EsophDysmotility", "a_col":"lc_ssc",
     "a_fn": lambda x: x==0, "b_col":"esoph_dysmotility",
     "claim":"Diffuse cutaneous SSc implies esophageal dysmotility",
     "source":"Ebert et al. 2012 (95.4% dcSSc vs 58.5% lcSSc); Roman et al. 2011",
     "rationale":"dcSSc is the dominant predictor of esophageal dysmotility (ED). Ebert et al. (2012) showed 95.4% of dcSSc patients had abnormal esophageal transit scintigraphy vs 58.5% of lcSSc patients. Roman et al. (2011) using high-resolution manometry found diffuse skin involvement, Scl70, and absence of ACA as independent predictors of ED."},
    {"name":"Scl70 => EsophDysmotility", "a_col":"scl70",
     "a_fn": lambda x: x==1, "b_col":"esoph_dysmotility",
     "claim":"Anti-Scl70 antibody implies esophageal dysmotility",
     "source":"Roman et al. 2011 (HRM); Medsger 2003",
     "rationale":"Anti-topoisomerase I (Scl70) antibody is the serological marker of dcSSc and independently predicts esophageal dysmotility in high-resolution manometry studies (Roman et al. 2011). Its association with dcSSc phenotype means Scl70-positive patients have the most severe esophageal smooth muscle fibrosis."},
    {"name":"Dysphagia => EsophDysmotility", "a_col":"dysphagia",
     "a_fn": lambda x: x==1, "b_col":"esoph_dysmotility",
     "claim":"Dysphagia implies esophageal dysmotility",
     "source":"Ling et al. PMC 2021; Alcala-Gonzalez et al. 2024",
     "rationale":"Dysphagia is the cardinal symptom of esophageal involvement in SSc, directly reflecting impaired peristalsis. The GERD study by Alcala-Gonzalez (2024) confirmed dysphagia as a predictor of PPI-refractory outcomes, and Ling et al. (2021) identified it as the main clinical manifestation of esophageal dysmotility in SSc."},
    {"name":"EsophDysmotility => PPIrefractory", "a_col":"esoph_dysmotility",
     "a_fn": lambda x: x==1, "b_col":"ppi_refractory",
     "claim":"Esophageal dysmotility implies PPI-refractory GERD symptoms",
     "source":"Alcala-Gonzalez et al. Rheumatology 2024; Ling et al. 2021",
     "rationale":"PPIs suppress acid but cannot restore esophageal motility. Patients with SSc-related dysmotility have impaired clearance of refluxate regardless of acid suppression, leading to PPI-refractory symptoms. Alcala-Gonzalez (2024) demonstrated GI dysmotility as the only independent predictor of PPI-refractory esophagitis in SSc."},
]

ALPHA = 0.05
VERDICT_COLORS = {"STRONG":"#2ecc71","MODERATE":"#3498db","WEAK":"#f39c12",
                  "TRENDING":"#e67e22","REJECTED":"#e74c3c"}
SHORT_LABELS = ["dcSSc\n=> Esoph DM","Scl70\n=> Esoph DM",
                "Dysphagia\n=> Esoph DM","Esoph DM\n=> PPI-refract"]

def pac_bound(n,k,alpha=ALPHA):
    if n==0: return None
    if k==0: return 1-(alpha**(1.0/n))
    return float(stats.beta.ppf(1-alpha,k+1,n-k))

def chi2_phi(a,b):
    ct=pd.crosstab(a,b)
    if ct.shape!=(2,2): return {"chi2":None,"p_value":1.0,"phi":0.0}
    c2,p,_,_=stats.chi2_contingency(ct)
    return {"chi2":round(float(c2),3),"p_value":round(float(p),6),"phi":round(float(np.sqrt(c2/len(a))),3)}

def causal_edge(df,a_col,confounders):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cols=confounders+[a_col,"esoph_dysmotility" if "ppi" not in a_col else "ppi_refractory"]
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

all_results={}
CONF=["age","sex","disease_duration"]

for imp in IMPLICATIONS:
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int); b_vec=df[imp["b_col"]]
    a1=b_vec[a_vec==1]; n_a1=len(a1); n_viol=int((a1==0).sum())
    vr=n_viol/n_a1 if n_a1>0 else None
    pac=pac_bound(n_a1,n_viol); chi=chi2_phi(a_vec,b_vec)
    ca=causal_edge(df,imp["a_col"],CONF); inv=invar(df,a_vec,imp["b_col"])
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

with open("results/esophageal_implication_results.json","w") as f:
    json.dump(all_results,f,indent=2,default=str)

# ── Figure 1: PAC bounds ──────────────────────────────────────────────────────
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
ax.set_title("Esophageal Dysmotility Implication Violation Rates\nSSc Synthetic Cohort (n=800, calibrated to Roman 2011, Ebert 2012)",fontsize=11)
ax.legend(fontsize=9)
for i,(v,b,n) in enumerate(zip(rates,bounds,ns)):
    p=all_results[n]["chi_squared"]["p_value"] or 1
    sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    ax.text(i,b+0.025,f"χ² {sig}",ha="center",fontsize=8,color="dimgray")
    ax.text(x[i]-0.2,v+0.012,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold",color="white" if v>0.08 else "black")
plt.tight_layout(); plt.savefig("figures/pac_bounds_esophageal.png",dpi=150); plt.close()

# ── Figure 2: Contingency ─────────────────────────────────────────────────────
fig,axes=plt.subplots(1,4,figsize=(14,4)); fig.suptitle("Esophageal Dysmotility Prevalence by Antecedent",fontsize=12)
for ax,imp,short in zip(axes,IMPLICATIONS,SHORT_LABELS):
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int)
    ct=pd.crosstab(a_vec,df[imp["b_col"]])
    ct.index=["A=0","A=1"]; ct_pct=ct.div(ct.sum(axis=1),axis=0)*100
    ct_pct.plot(kind="bar",ax=ax,color=["#3498db","#e74c3c"],edgecolor="white",width=0.6)
    ax.set_title(f"{short}\n[{all_results[imp['name']]['verdict']}]",fontsize=8)
    ax.set_ylabel("% patients" if ax==axes[0] else ""); ax.tick_params(axis="x",rotation=0)
    ax.legend(fontsize=7); ax.set_ylim(0,100)
plt.tight_layout(); plt.savefig("figures/contingency_esophageal.png",dpi=150,bbox_inches="tight"); plt.close()

# ── Figure 3: Sex-stratified ──────────────────────────────────────────────────
SEX_COLORS={"Female\n(sex=1)":"#e91e8c","Male\n(sex=0)":"#1e88e5","Overall":"#607d8b"}
fig,axes=plt.subplots(1,4,figsize=(14,5),sharey=True)
fig.suptitle("Esophageal Dysmotility Violation Rates by Sex",fontsize=11)
for ax,imp,short in zip(axes,IMPLICATIONS,SHORT_LABELS):
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int); b_col=imp["b_col"]; res=all_results[imp["name"]]
    gl,vr,pu,bc=[],[],[],[]
    for sv,sl in [(1,"Female\n(sex=1)"),(0,"Male\n(sex=0)")]:
        m=(df["sex"]==sv)&(a_vec==1); s=df.loc[m,b_col]; n=len(s)
        if n<10: continue
        k=int((s==0).sum()); p=pac_bound(n,k)
        gl.append(sl); vr.append(k/n); pu.append(p or k/n); bc.append(SEX_COLORS[sl])
    gl.append("Overall"); vr.append(res["violation_rate"] or 0); pu.append(res["pac_bound"] or 0); bc.append(SEX_COLORS["Overall"])
    xp=np.arange(len(gl)); bars=ax.bar(xp,vr,color=bc,edgecolor="white",width=0.5,alpha=0.88)
    ax.errorbar(xp,vr,yerr=[np.zeros(len(xp)),[max(0,p-v) for p,v in zip(pu,vr)]],fmt="none",ecolor="black",capsize=5)
    ax.axhline(0.30,color="gray",linestyle="--",linewidth=1)
    ax.set_xticks(xp); ax.set_xticklabels(gl,fontsize=8); ax.set_title(short.replace("\n"," "),fontsize=8); ax.set_ylim(0,0.85)
    if ax==axes[0]: ax.set_ylabel("Violation rate")
    for bar,v in zip(bars,vr): ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
fig.legend(handles=[Patch(facecolor=c,label=l) for l,c in SEX_COLORS.items()],
           loc="lower center",ncol=3,fontsize=9,bbox_to_anchor=(0.5,-0.05))
plt.tight_layout(); plt.savefig("figures/sex_stratified_esophageal.png",dpi=150,bbox_inches="tight"); plt.close()

# ── Figure 4: Summary table ───────────────────────────────────────────────────
rows=[{"Implication":n,"N(A=1)":r["n_a1"],"Violations":r["n_violations"],
       "Viol. rate":f"{r['violation_rate']:.2f}","PAC bound":f"{r['pac_bound']:.3f}",
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
ax.set_title("Esophageal Dysmotility Implication Testing Summary — CREST Paper 03",fontsize=13,pad=16,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/summary_table_esophageal.png",dpi=150,bbox_inches="tight"); plt.close()
print("All esophageal figures saved.")
