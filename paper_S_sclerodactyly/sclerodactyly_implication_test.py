"""
sclerodactyly_implication_test.py
----------------------------------
Statistical implication testing — Sclerodactyly in SSc (S in CREST).
Paper 04 in the CREST Implication Testing Series.

Calibration:
  - ACR/EULAR 2013 criteria: sclerodactyly scores 4 points (major criterion)
  - mRSS: dcSSc mean ~18, lcSSc mean ~6 (Khanna et al. 2017)
  - Digital pitting scars: ~40-53% SSc patients (EUSTAR)
  - dcSSc prevalence of sclerodactyly: ~95%; lcSSc: ~75%
  - Anti-Scl70 and high mRSS: independent predictors

Four implications:
  1. dcSSc                  => Sclerodactyly (skin thickening fingers)
  2. Anti-Scl70 positive    => Sclerodactyly
  3. High mRSS (>14)        => Sclerodactyly
  4. Sclerodactyly          => Digital pitting scars
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

    p_aca   = np.where(lc_ssc==1, 0.48, 0.08); aca = rng.binomial(1, p_aca)
    p_scl70 = np.where(lc_ssc==1, 0.06, 0.42); scl70 = rng.binomial(1, p_scl70)

    # mRSS: dcSSc mean ~18 (sd 9), lcSSc mean ~6 (sd 5) — Khanna 2017
    mrss_raw = np.where(lc_ssc==1,
                        rng.normal(6, 5, n).clip(0, 25),
                        rng.normal(18, 9, n).clip(0, 51))
    mrss = mrss_raw.astype(int)
    high_mrss = (mrss > 14).astype(int)

    # Sclerodactyly: near-universal in dcSSc (~95%), common in lcSSc (~75%)
    p_scl = 0.70 + 0.22*(lc_ssc==0) + 0.30*scl70 + 0.005*disease_duration.clip(0,20)
    p_scl = p_scl.clip(0.50, 0.99)
    sclerodactyly = rng.binomial(1, p_scl)

    # Digital pitting scars: ~40% overall; higher with sclerodactyly + dcSSc
    p_dps = 0.20 + 0.35*sclerodactyly + 0.12*(lc_ssc==0) + 0.004*disease_duration.clip(0,20)
    digital_pitting = rng.binomial(1, p_dps.clip(0.05, 0.85))

    # Puffy fingers (early sign)
    p_puffy = 0.20 + 0.10*(disease_duration < 3)
    puffy = rng.binomial(1, p_puffy, n)

    return pd.DataFrame({
        "sex": sex, "age": age.round(1), "disease_duration": disease_duration.round(1),
        "lc_ssc": lc_ssc, "aca": aca, "scl70": scl70,
        "mrss": mrss, "high_mrss": high_mrss,
        "sclerodactyly": sclerodactyly, "digital_pitting": digital_pitting, "puffy_fingers": puffy,
    })

df = generate_cohort(N)
print(f"n={N}  sclerodactyly={df.sclerodactyly.mean():.1%}  "
      f"dcSSc={df.loc[df.lc_ssc==0,'sclerodactyly'].mean():.1%}  "
      f"lcSSc={df.loc[df.lc_ssc==1,'sclerodactyly'].mean():.1%}  "
      f"pitting={df.digital_pitting.mean():.1%}")
df.to_csv("results/synthetic_ssc_sclerodactyly_cohort.csv", index=False)

IMPLICATIONS = [
    {"name":"dcSSc => Sclerodactyly", "a_col":"lc_ssc", "a_fn":lambda x: x==0, "b_col":"sclerodactyly",
     "claim":"Diffuse cutaneous SSc implies sclerodactyly",
     "source":"ACR/EULAR 2013 criteria; Khanna et al. 2017; EUSTAR database",
     "rationale":"Sclerodactyly (skin thickening of fingers) is a defining feature of dcSSc, scoring 4 points in the ACR/EULAR 2013 classification criteria. dcSSc by definition involves skin thickening proximal to elbows/knees, and sclerodactyly is near-universal (~95%) in this subtype. The mRSS trajectory shows rapid progression in dcSSc within 12-18 months of onset."},
    {"name":"Scl70 => Sclerodactyly", "a_col":"scl70", "a_fn":lambda x: x==1, "b_col":"sclerodactyly",
     "claim":"Anti-Scl70 antibody implies sclerodactyly",
     "source":"Medsger 2003; EUSTAR database; Denton & Khanna 2017",
     "rationale":"Anti-topoisomerase I (Scl70) is the autoantibody of dcSSc, strongly predicting progressive skin fibrosis. Scl70-positive patients have the highest rates of sclerodactyly and the highest mRSS scores. Antitopoisomerase was identified as an independent factor for onset of skin fibrosis (OR 3.08) in the EUSTAR database."},
    {"name":"HighmRSS => Sclerodactyly", "a_col":"high_mrss", "a_fn":lambda x: x==1, "b_col":"sclerodactyly",
     "claim":"High modified Rodnan Skin Score (>14) implies sclerodactyly",
     "source":"Khanna et al. J Scleroderma Relat Disord 2017; Clements et al. 1995",
     "rationale":"The modified Rodnan Skin Score (mRSS) includes 17 body sites including fingers. An mRSS >14 indicates severe widespread skin involvement and by definition requires significant finger/hand involvement (sclerodactyly). Severe skin involvement (mRSS >20) is an independent risk factor for poor outcomes."},
    {"name":"Sclerodactyly => DigitalPitting", "a_col":"sclerodactyly", "a_fn":lambda x: x==1, "b_col":"digital_pitting",
     "claim":"Sclerodactyly implies digital pitting scars",
     "source":"EUSTAR database; ACR/EULAR 2013 criteria (3 points for pitting scars)",
     "rationale":"Digital pitting scars are the direct consequence of chronic ischaemia in fingers with sclerodactyly. The ACR/EULAR 2013 classification criteria award 3 points for fingertip pitting scars (only slightly fewer than sclerodactyly's 4 points), indicating their strong co-occurrence. Sclerodactyly creates the vascular compromise that leads to pitting scars over time."},
]

ALPHA = 0.05
VERDICT_COLORS = {"STRONG":"#2ecc71","MODERATE":"#3498db","WEAK":"#f39c12","TRENDING":"#e67e22","REJECTED":"#e74c3c"}
SHORT_LABELS = ["dcSSc\n=> Sclerodact","Scl70\n=> Sclerodact","High mRSS\n=> Sclerodact","Sclerodact\n=> Dig Pitting"]

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

with open("results/sclerodactyly_implication_results.json","w") as f:
    json.dump(all_results,f,indent=2,default=str)

# Figures (same template)
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
ax.set_title("Sclerodactyly Implication Violation Rates\nSSc Synthetic Cohort (n=800, calibrated to EUSTAR & Khanna 2017)",fontsize=11)
ax.legend(fontsize=9)
for i,(v,b,n) in enumerate(zip(rates,bounds,ns)):
    p=all_results[n]["chi_squared"]["p_value"] or 1
    sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    ax.text(i,b+0.025,f"χ² {sig}",ha="center",fontsize=8,color="dimgray")
    ax.text(x[i]-0.2,v+0.012,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold",color="white" if v>0.08 else "black")
plt.tight_layout(); plt.savefig("figures/pac_bounds_sclerodactyly.png",dpi=150); plt.close()

fig,axes=plt.subplots(1,4,figsize=(14,4)); fig.suptitle("Sclerodactyly Prevalence by Antecedent",fontsize=12)
for ax,imp,short in zip(axes,IMPLICATIONS,SHORT_LABELS):
    a_vec=imp["a_fn"](df[imp["a_col"]]).astype(int)
    ct=pd.crosstab(a_vec,df[imp["b_col"]]); ct.index=["A=0","A=1"]
    ct_pct=ct.div(ct.sum(axis=1),axis=0)*100
    ct_pct.plot(kind="bar",ax=ax,color=["#3498db","#e74c3c"],edgecolor="white",width=0.6)
    ax.set_title(f"{short}\n[{all_results[imp['name']]['verdict']}]",fontsize=8)
    ax.set_ylabel("% patients" if ax==axes[0] else ""); ax.tick_params(axis="x",rotation=0)
    ax.legend(fontsize=7); ax.set_ylim(0,100)
plt.tight_layout(); plt.savefig("figures/contingency_sclerodactyly.png",dpi=150,bbox_inches="tight"); plt.close()

SEX_COLORS={"Female\n(sex=1)":"#e91e8c","Male\n(sex=0)":"#1e88e5","Overall":"#607d8b"}
fig,axes=plt.subplots(1,4,figsize=(14,5),sharey=True); fig.suptitle("Sclerodactyly Violation Rates by Sex",fontsize=11)
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
plt.tight_layout(); plt.savefig("figures/sex_stratified_sclerodactyly.png",dpi=150,bbox_inches="tight"); plt.close()

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
ax.set_title("Sclerodactyly Implication Testing Summary — CREST Paper 04",fontsize=13,pad=16,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/summary_table_sclerodactyly.png",dpi=150,bbox_inches="tight"); plt.close()
print("All sclerodactyly figures saved.")
