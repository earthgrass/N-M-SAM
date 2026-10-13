from time import perf_counter
from .gradient_path import gradient_path
from .hvp import hvp
from .curvature_decomp import decompose
from .candidate_ranking import ranking_metrics,validate_labels


def evaluate_direction_gap(loss,z,master,support,candidates,*,rho=.05,labels):
    path=gradient_path(loss,z,support,rho);g=path["g1"];base=float(loss(z));rows=[]
    for candidate in candidates:
        start=perf_counter();q=candidate["q"];new=candidate["support"]
        actual=float(loss(z+q))-base;first=float(g@q);hq=hvp(loss,z,q)
        second=first+.5*float(q@hq)
        row={**labels,"candidate_id":candidate["candidate_id"],"q_norm":float(q.norm()),
            "q_over_rho":float(q.norm())/rho if rho else None,"actual":actual,"first":first,"second":second,
            "magnitude":-float(master[new].abs().sum()),
            "taylor_g1":-float((master*path["g1"])[new].abs().sum()),
            "taylor_g2":-float((master*path["g2"])[new].abs().sum()),
            "direction_status":path["status"],"hvp_calls":1}
        if path["status"]=="defined":
            d=decompose(loss,z,q,path["u"],path["Hu"]);a,r=d["a"],d["r"]
            observed=a*a*(path["u"]@path["h_obs"])+2*a*(r@path["h_obs"])
            row.update(a=float(a),r_norm=float(r.norm()),parallel=float(d["parallel"]),
                cross=float(d["cross"]),residual=float(d["residual"]),
                observed_only=first+.5*float(observed),
                observed_plus_residual=first+.5*float(observed+d["residual"]),
                decomposition_error=float(abs(d["quadratic_sum"]-q@hq)),
                path_average_error=path["path_average_error"],hvp_calls=2)
        row["elapsed_seconds"]=perf_counter()-start;rows.append(row)
    validate_labels(rows)
    keys=["first","second","magnitude","taylor_g1","taylor_g2"]
    if path["status"]=="defined":keys += ["observed_only","observed_plus_residual"]
    rankings={k:ranking_metrics([r[k] for r in rows],[r["actual"] for r in rows]) for k in keys}
    for key in ("magnitude","taylor_g1","taylor_g2"):
        rankings[key].update(mae=None,rmse=None,units="ranking_score_only")
    return {"rows":rows,"rankings":rankings,"shared_hvp_calls":1 if path["status"]=="defined" else 0}
