"""Fit every row in a manifest using reconstruct.py."""
import argparse
import pandas as pd
from reconstruct import fit
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest_csv"); ap.add_argument("output_csv"); a=ap.parse_args(); m=pd.read_csv(a.manifest_csv)
    if "csv_path" not in m: raise ValueError("manifest needs csv_path")
    out=[]
    for _,r in m.iterrows():
        t=r.get("tds_percent"); t=None if pd.isna(t) else float(t); dose=float(r.get("dose_g",18.))
        z=fit(str(r.csv_path),tds=t,dose=dose); out.append({"csv_path":r.csv_path,"tds_target_percent":t,"dose_g":dose,"loss":z["loss"],"flow_r2":z["flow_r2"],"predicted_tds_percent":z["predicted_tds"],**z["parameters"]})
    pd.DataFrame(out).to_csv(a.output_csv,index=False); print(f"fitted={len(out)}")
if __name__=='__main__': main()
