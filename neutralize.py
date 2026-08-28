"""Post-fit mechanism neutralization and flow/mass R2 loss."""
import argparse
import numpy as np
import pandas as pd
from reconstruct import load_and_prep_data, simulate_numba_core, PARAMETER_NAMES, MAX_SOLUBLES_FRACTION
NEUTRALIZATIONS={"wetting_connectivity":["t_off","wetting_tau"],"resistance_relief":["alpha_ero"],"structural_terms":["alpha_comp","alpha_ero","beta_swelling"],"swelling":["beta_swelling"],"viscosity_response":["a_visc"],"compaction":["alpha_comp"]}
def r2(y,z): return float(1-np.sum((y-z)**2)/max(np.sum((y-y.mean())**2),1e-12))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("fit_csv"); ap.add_argument("output_csv"); a=ap.parse_args(); f=pd.read_csv(a.fit_csv); out=[]
    for _,row in f.iterrows():
        t,p,q,w,_=load_and_prep_data(str(row.csv_path)); x=np.array([row[n] for n in PARAMETER_NAMES],float); sim=simulate_numba_core(t,p,x,float(row.dose_g),MAX_SOLUBLES_FRACTION*float(row.dose_g)*.70,1.0); qb,wb=sim[0],sim[2]
        for name,zero in NEUTRALIZATIONS.items():
            z=x.copy()
            for n in zero: z[PARAMETER_NAMES.index(n)]=0.
            if name=="wetting_connectivity": z[PARAMETER_NAMES.index("t_off")]=0.; z[PARAMETER_NAMES.index("wetting_tau")]=.1
            sim2=simulate_numba_core(t,p,z,float(row.dose_g),MAX_SOLUBLES_FRACTION*float(row.dose_g)*.70,1.0); q2,w2=sim2[0],sim2[2]; out.append({"csv_path":row.csv_path,"neutralization":name,"flow_r2_loss":r2(q,qb)-r2(q,q2),"mass_r2_loss":r2(w,wb)-r2(w,w2),"baseline_flow_r2":r2(q,qb),"baseline_mass_r2":r2(w,wb)})
    pd.DataFrame(out).to_csv(a.output_csv,index=False); print(f"neutralized={len(out)}")
if __name__=='__main__': main()
