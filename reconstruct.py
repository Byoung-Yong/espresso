"""Minimal public reference implementation of the espresso reconstruction.

Usage: python reconstruct.py shot.csv --tds 8.57 --out fit.json
The CSV needs elapsed, pressure, flow_out, current_total_shot_weight.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter

P = dict(base_porosity=.33, min_porosity=.14, max_comp=.08, comp_scale=150.,
         max_relief=.90, residual=.55, release=8., min_wet=1e-4,
         max_solubles=.30)
NAMES = ['R0','alpha_comp','alpha_ero','a_visc','b_visc','k_extr','t_off',
         'wetting_tau','hold_capacity_ratio','beta_swelling']
BOUNDS0 = [(0.30,100.),(0.,.5),(.1,10.),(0.,10.),(.1,3.),(.1,5.),
           (0.,20.),(.5,8.),(.05,.60),(0.,.18)]

def smooth(x):
    return x*x*(3-2*x) if 0 < x < 1 else float(x >= 1)

def load(path, dt=.1):
    d = pd.read_csv(path)
    if 'current_total_shot_weight' not in d and 'weight' in d:
        d['current_total_shot_weight'] = d['weight']
    cols = ['elapsed','pressure','flow_out','current_total_shot_weight']
    if any(c not in d for c in cols): raise ValueError('missing required CSV column')
    d = d[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(d) < 10: raise ValueError('fewer than 10 valid rows')
    t0 = d.elapsed.min(); t = np.arange(0., d.elapsed.max()-t0+dt/2, dt)
    p = np.interp(t, d.elapsed-t0, d.pressure); q = np.interp(t, d.elapsed-t0, d.flow_out)
    w = np.interp(t, d.elapsed-t0, d.current_total_shot_weight); w -= w[0]
    win = int(1.5/dt); win += 1-win%2
    if win >= 3 and win < len(t):
        p, q = savgol_filter(p,win,2), savgol_filter(q,win,2)
    return t, np.maximum(p,0), np.maximum(q,0), w

def simulate(t, pressure, x, dose=18., eta=.70):
    R0, acomp, aero, avis, bvis, kextr, toff, tau, hold, bswell = x
    n=len(t); dt=np.r_[0.,np.diff(t)]; q=np.zeros(n); mass=np.zeros(n); tds=np.zeros(n)
    resistance=np.zeros(n); retained=np.zeros(n); wet=np.zeros(n); swelling=np.zeros(n)
    extracted=stored=cump=0.; maxsol=P['max_solubles']*dose*eta
    for i,(ti,Pi) in enumerate(zip(t,pressure)):
        Pi=max(float(Pi),0.); cump += Pi*dt[i]
        wf = smooth((ti-toff)/tau) if ti>=toff and tau>.1 else float(ti>=toff)
        prev=mass[i-1] if i else 0.; frac=min(extracted/max(maxsol,1e-9),1.)
        proxy=extracted/max(prev,1e-9); cap=max(hold*dose,1e-6)
        h=min(max(stored/cap,0.),1.); sw=h**1.5
        comp=1-np.exp(-acomp*pump(cump)/P['comp_scale'])
        gross=P['max_comp']*comp+bswell*sw
        gate=wf*smooth((prev/dose-.15)/.25) if dose>1e-9 else 0.
        erosion=1-np.exp(-aero*frac*gate); drop=gross*(1-P['max_relief']*erosion)
        por=max(P['min_porosity'],P['base_porosity']-drop)
        perm=(P['base_porosity']**3/(1-P['base_porosity'])**2)/(por**3/(1-por)**2)
        vg=wf*smooth((prev/dose-.10)/.20) if dose>1e-9 else 0.
        visc=1+vg*avis*max(proxy,0.)**bvis
        r=max(1e-4,R0*perm*visc); qi=Pi/r; qo=wf*qi
        loading=max(qi-qo,0)*dt[i]; stored += loading
        target=cap*(P['residual']+(1-P['residual'])*(1-wf)); release=min(max(wf*(stored-target),0)*dt[i]/P['release'],stored); stored=max(stored-release,0.)
        dm=min(maxsol-extracted,kextr*qo*max(maxsol-extracted,0)/max(dose,1e-9)*dt[i])
        extracted += max(dm,0.); q[i]=qo+(dm/dt[i] if dt[i]>1e-12 else 0.); mass[i]=(mass[i-1] if i else 0)+q[i]*dt[i]
        tds[i]=100*extracted/max(mass[i],1e-6); resistance[i]=Pi/max(q[i],P['min_wet']); retained[i]=stored; wet[i]=wf; swelling[i]=sw
    return q,mass,tds,resistance,retained,wet,swelling

def pump(x): return max(float(x),0.)

def objective(x,t,p,qref,wref,dose,tds=None):
    q,w,tds_sim,r,*_=simulate(t,p,x,dose)
    mse_q=np.mean((q-qref)**2); mse_w=np.mean((w-wref)**2); final=(w[-1]-wref[-1])**2
    mask=qref>.1; mse_r=0.
    if mask.sum()>10:
        rr=np.maximum(p[mask]/qref[mask],1e-5); rs=np.maximum(r[mask],1e-5); mse_r=np.mean((np.log(rs)-np.log(rr))**2)
    onset=np.flatnonzero(qref>.1); tref=t[onset[0]] if len(onset) else 5.; onset2=np.flatnonzero(q>.1); tsim=t[onset2[0]] if len(onset2) else t[-1]
    loss=10*mse_q+15*mse_w+50*final+40*mse_r+300*(tsim-tref)**2
    if tds is not None: loss += 25*(tds_sim[-1]-tds)**2
    return float(loss)

def fit(csv, tds=None, dose=18.):
    t,p,q,w=load(csv); onset=np.flatnonzero(q>.1); onset=t[onset[0]] if len(onset) else 5.
    bounds=BOUNDS0.copy(); bounds[6]=(max(0,onset-1.5),onset+1.5)
    args=(t,p,q,w,dose,tds); de=differential_evolution(objective,bounds,args=args,seed=42,maxiter=60,popsize=15,tol=.02,polish=False)
    local=minimize(objective,de.x,args=args,method='L-BFGS-B',bounds=bounds,tol=1e-6)
    q2,w2,tds2,r,*_=simulate(t,p,local.x,dose); r2=1-np.sum((q-q2)**2)/max(np.sum((q-q.mean())**2),1e-12)
    return dict(loss=float(local.fun),flow_r2=float(r2),predicted_tds=float(tds2[-1]),parameters=dict(zip(NAMES,map(float,local.x))))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--tds',type=float); ap.add_argument('--dose',type=float,default=18.); ap.add_argument('--out',default='fit.json'); a=ap.parse_args()
    with open(a.out,'w',encoding='utf-8') as f: json.dump(fit(a.csv,a.tds,a.dose),f,indent=2)
    print(a.out)
if __name__=='__main__': main()
