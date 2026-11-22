"""Paired statistical analysis; report dependent folds transparently."""
import numpy as np
from scipy.stats import wilcoxon


def holm_adjust(p_values):
    p=np.asarray(p_values,dtype=float)
    order=np.argsort(p);n=len(p); out=np.zeros(n)
    running=0.
    for j,idx in enumerate(order):
        running=max(running,min(1.,p[idx]*(n-j)))
        out[idx]=running
    return out.tolist()


def paired_comparison(proposed,baseline,metric="f1",bootstrap=2000,seed=42):
    a=np.asarray(proposed,float);b=np.asarray(baseline,float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2:
        raise ValueError("at least two paired scalar observations required")
    delta=a-b
    if np.allclose(delta,0): p=1.;effect=0.
    else:
        nz=delta[delta!=0]
        # Exact under the Wilcoxon zero/tie conditions; SciPy selects approximation otherwise.
        res=wilcoxon(delta,alternative="two-sided",method="auto",zero_method="wilcox")
        p=float(res.pvalue)
        from scipy.stats import rankdata
        ranks=rankdata(np.abs(nz))
        plus=float(ranks[nz>0].sum());minus=float(ranks[nz<0].sum())
        effect=(plus-minus)/max(plus+minus,1e-12)
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(delta,size=(bootstrap,len(delta)),replace=True),axis=1)
    return {"metric":metric,"n_pairs":len(a),"mean_proposed":float(a.mean()),
            "sd_proposed":float(a.std(ddof=1)),"mean_baseline":float(b.mean()),
            "sd_baseline":float(b.std(ddof=1)),"mean_paired_difference":float(delta.mean()),
            "CI95_low":float(np.percentile(means,2.5)),"CI95_high":float(np.percentile(means,97.5)),
            "wilcoxon_p_unadjusted":p,"rank_biserial":float(effect),
            "caveat":"overlapping rolling folds may make 25 pairs statistically dependent"}
