"""Reproduce the July-2026 Figure 5 endpoint-matching analysis.

Usage: python endpoint_matching.py manifest.csv fits.csv telemetry_dir output_dir
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

# The manuscript comparison uses the same 201-point normalized-progress grid
# as the production Figure 5 analysis.
GRID=np.linspace(0,1,201)
def r2pair(a,b):
    d=2*abs(a-b)/(abs(a)+abs(b)); return d
def same(a,b):
    x,y=str(a.user_id).strip(),str(b.user_id).strip(); return bool(x and y and x==y)
def trace(path):
    d=pd.read_csv(path); t=pd.to_numeric(d.elapsed,errors='coerce').to_numpy(float); p=pd.to_numeric(d.pressure,errors='coerce').to_numpy(float); q=pd.to_numeric(d.flow_weight,errors='coerce').to_numpy(float); ok=np.isfinite(t); x=(t-t[ok][0])/(t[ok][-1]-t[ok][0])
    def ip(y,m):
        z=np.isfinite(x)&m; xx,yy=x[z],y[z]; o=np.argsort(xx); xx,yy=xx[o],yy[o]; u=np.r_[True,np.diff(xx)>0]; xx,yy=xx[u],yy[u]; inside=(GRID>=xx[0])&(GRID<=xx[-1]); out=np.full(GRID.shape,np.nan); out[inside]=np.interp(GRID[inside],xx,yy); return out,inside
    # Apparent resistance is defined only where observed outlet flow exceeds
    # the production-analysis validity threshold (0.1 g s^-1).  No zero
    # replacement or interpolation across invalid resistance samples is made.
    valid_r = np.isfinite(p) & np.isfinite(q) & (q > 0.1)
    return tuple(ip(y,m) for y,m in ((p,np.isfinite(p)),(q,np.isfinite(q)),(p/np.where(q>0,q,np.nan),valid_r)))
def rms(a,b,m):
    z=m&np.isfinite(a)&np.isfinite(b); return float(np.sqrt(np.mean((a[z]-b[z])**2)))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('fits'); ap.add_argument('telemetry_dir'); ap.add_argument('output_dir'); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(exist_ok=True)
    m=pd.read_csv(a.manifest,low_memory=False); f=pd.read_csv(a.fits,usecols=['id','r_squared'],low_memory=False); d=m.merge(f,on='id',validate='one_to_one');
    for c in ('drink_tds','bean_weight','drink_weight','r_squared'): d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.loc[d.drink_tds.gt(0)&d.bean_weight.notna()&d.drink_weight.notna()].set_index('id',drop=False); tr={i:trace(Path(a.telemetry_dir)/(str(i)+'.csv')) for i in d.index}; rows=[]
    for i,j in combinations(d.index,2):
        x,y=d.loc[i],d.loc[j]
        if same(x,y): continue
        tx,ty=tr[i],tr[j]; rows.append({'id1':i,'id2':j,'profile1':x.profile_title,'profile2':y.profile_title,'r_squared1':x.r_squared,'r_squared2':y.r_squared,'strict_pair':x.r_squared>=.95 and y.r_squared>=.95,'delta_tds_pp':abs(x.drink_tds-y.drink_tds),'delta_dose_g':abs(x.bean_weight-y.bean_weight),'delta_mass_g':abs(x.drink_weight-y.drink_weight),'pressure_rms_bar':rms(tx[0][0],ty[0][0],tx[0][1]&ty[0][1]),'flow_rms_g_s':rms(tx[1][0],ty[1][0],tx[1][1]&ty[1][1]),'resistance_rms_bar_s_g':rms(tx[2][0],ty[2][0],tx[2][1]&ty[2][1])})
    p=pd.DataFrame(rows); p.to_csv(out/'all_cross_user_pairs.csv',index=False); s=[]
    mask=(p.delta_tds_pp<=.30)&(p.delta_dose_g<=.5)&(p.delta_mass_g<=1.0); q=p.loc[mask]; s.append({'tds_tolerance_pp':.30,'dose_tolerance_g':.5,'mass_tolerance_g':1.0,'pair_count':len(q),'unique_shot_count':len(set(q.id1)|set(q.id2)),'strict_pair_count':int(q.strict_pair.sum()),'pressure_rms_median_bar':q.pressure_rms_bar.median(),'flow_rms_median_g_s':q.flow_rms_g_s.median(),'resistance_rms_median_bar_s_g':q.resistance_rms_bar_s_g.median()})
    pd.DataFrame(s).to_csv(out/'summary.csv',index=False); print(pd.DataFrame(s).to_string(index=False))
if __name__=='__main__': main()
