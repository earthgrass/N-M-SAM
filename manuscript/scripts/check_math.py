"""Independent small-matrix checks of manuscript identities, not research results."""
import json, itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261105); results=[]
def check(name,error,tol):
    error=float(error);assert error<tol,(name,error,tol)
    results.append({'check':name,'max_error':error,'tolerance':tol,'status':'PASS'})
Z=rng.normal(size=(7,7)); B=Z.T@Z+np.eye(7);Q=np.linalg.inv(B);w=rng.normal(size=7)
for indices in [[2],[1,4]]:
    E=np.eye(7)[:,indices]; KKT=np.block([[B,E],[E.T,np.zeros((len(indices),len(indices)))]])
    delta=np.linalg.solve(KKT,np.r_[np.zeros(7),-w[indices]])[:7]
    formula=-Q[:,indices]@np.linalg.solve(Q[np.ix_(indices,indices)],w[indices])
    check(f'KKT_vs_OBS_{len(indices)}deletions',np.max(np.abs(delta-formula)),1e-12)
    check(f'zero_constraint_{len(indices)}',np.max(np.abs((w+formula)[indices])),1e-12)
    cost=.5*w[indices]@np.linalg.solve(Q[np.ix_(indices,indices)],w[indices])
    check(f'quadratic_cost_{len(indices)}',abs(.5*delta@B@delta-cost),1e-12)
eta=.1;s=np.zeros(7);product=np.ones(7)
for e in range(5):
    q=np.exp(rng.normal(size=7));u=.5*np.log(q);u-=u.mean()
    s=(1-eta)*s+eta*u;product=product**(1-eta)*(q/np.exp(np.log(q).mean()))**eta
check('EMA_product',np.max(np.abs(np.exp(2*s)-product)),1e-12)
check('centering',abs(s.sum()),1e-12)
g=rng.normal(size=7);b=np.exp(2*s);eps=.05*b*g/np.linalg.norm(b*g)
check('global_radius',abs(np.linalg.norm(eps)-.05),1e-12)
p=(b*g)**2/np.sum((b*g)**2);i=2;h=1e-5
bp=b.copy();bm=b.copy();bp[i]*=np.exp(h);bm[i]*=np.exp(-h)
num=(((bp*g)**2/np.sum((bp*g)**2))-((bm*g)**2/np.sum((bm*g)**2)))/(2*h)
expected=-2*p[i]*p;expected[i]=2*p[i]*(1-p[i])
check('budget_derivative',np.max(np.abs(num-expected)),1e-8)
energy=rng.uniform(size=4)
random_mean=np.mean([energy[list(pair)].sum()/energy.sum() for pair in itertools.combinations(range(4),2)])
check('uniform_legal_mask_expectation',abs(random_mean-.5),1e-12)
(ROOT/'review/math_checks.json').write_text(json.dumps({'scope':'synthetic proof checks only; no model/data/Test accessed','checks':results},indent=2)+'\n')
print(f'{len(results)} independent mathematical checks PASS')
