import numpy as np

LABELS=("objective","derivative","state_mode","mask_id","snapshot_id")


def validate_labels(rows):
    if not rows:raise ValueError("no candidates")
    for row in rows:
        if not set(LABELS).issubset(row):raise ValueError("unlabelled research result")
        if any(row[k]!=rows[0][k] for k in LABELS):raise ValueError("incompatible probe labels")


def _rank(values):
    result=np.empty(len(values),dtype=float)
    for value in np.unique(values):
        indices=np.flatnonzero(values==value)
        result[indices]=sum(values<value)+(len(indices)-1)/2
    return result


def ranking_metrics(predicted,actual):
    p=np.asarray(predicted,dtype=float);a=np.asarray(actual,dtype=float)
    if p.shape!=a.shape or p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("finite matching candidate score vectors required")
    rp,ra=_rank(p),_rank(a)
    constant=np.std(rp)==0 or np.std(ra)==0
    selected=int(np.argmin(p));best=int(np.argmin(a))
    return {"mae":float(np.abs(p-a).mean()),"rmse":float(np.sqrt(np.mean((p-a)**2))),
        "spearman":None if constant else float(np.corrcoef(rp,ra)[0,1]),
        "rank_status":"constant_ranking" if constant else "defined",
        "top1_agreement":selected==best,"regret":float(a[selected]-a[best]),"selected":selected}
