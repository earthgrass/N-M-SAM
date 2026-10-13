from time import perf_counter
from .hvp import gradient,hvp
from .robust_probe import robust_proxy,quadratic_proxy
from .candidate_ranking import validate_labels,ranking_metrics


def evaluate_objective_gap(loss,z,support,candidates,*,rho=.05,steps=1,restarts=1,labels):
    base=float(loss(z));g=gradient(loss,z)[1].detach()
    old=robust_proxy(loss,z,support,rho,steps=steps,restarts=restarts)
    rows=[]
    for candidate in candidates:
        start=perf_counter();q,new=candidate["q"],candidate["support"]
        task_new=float(loss(z+q));delta_l=task_new-base
        result=robust_proxy(loss,z+q,new,rho,steps=steps,restarts=restarts)
        hq=hvp(loss,z,q);ordinary=float(g@q+.5*q@hq)
        local=quadratic_proxy(base+ordinary,g+hq,lambda v:hvp(loss,z,v),z,new,rho,steps=steps,restarts=restarts)
        delta_s=(result["value"]-task_new)-(old["value"]-base)
        rows.append({**labels,"candidate_id":candidate["candidate_id"],"L_old":base,"L_new":task_new,
            "delta_L":delta_l,"delta_S_proxy":delta_s,"delta_F_proxy":result["value"]-old["value"],
            "F_old_proxy":old["value"],"F_new_proxy":result["value"],
            "S_old_proxy":old["value"]-base,"S_new_proxy":result["value"]-task_new,
            "ordinary_second":ordinary,"Q_robust_proxy":local["value"],
            "new_support_delta_norm":float(result["delta"].norm()),
            "solver":result["solver"],"inner_steps":steps,"restarts":restarts,
            "backward_evaluations":result["backward_evaluations"],
            "global_max_proven":False,"elapsed_seconds":perf_counter()-start})
    validate_labels(rows)
    return {"rows":rows,"rankings":{
        "ordinary_vs_deltaF":ranking_metrics([r["ordinary_second"] for r in rows],[r["delta_F_proxy"] for r in rows]),
        "quadratic_vs_Fnew":ranking_metrics([r["Q_robust_proxy"] for r in rows],[r["F_new_proxy"] for r in rows])}}
