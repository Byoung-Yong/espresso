"""Extract ten final Fig. 4 trajectory descriptors and PCA scores."""
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
FEATURES=["wetting","retained_liquid","swelling","resistance_relief","viscosity_factor"]
PEAK=["retained_liquid","swelling","viscosity_factor"]
ONSET=["wetting","resistance_relief"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("trajectory_csv"); ap.add_argument("descriptor_csv"); ap.add_argument("score_csv"); a=ap.parse_args(); d=pd.read_csv(a.trajectory_csv)
    need={"shot_id","progress",*FEATURES}
    if not need.issubset(d): raise ValueError("missing trajectory columns")
    rows=[]
    for sid,g in d.groupby("shot_id",sort=False):
        g=g.sort_values("progress"); row={"shot_id":sid}
        if "program" in g: row["program"]=g.program.iloc[0]
        for f in FEATURES: row[f+"_auc"]=float(np.trapz(g[f],g.progress))
        for f in PEAK: row[f+"_peak"]=float(g[f].max())
        for f in ONSET:
            peak=float(g[f].max()); row[f+"_onset"]=float(g.progress.iloc[np.flatnonzero(g[f].to_numpy(float)>=.05*peak)[0]]) if peak>0 else 1.0
        rows.append(row)
    desc=pd.DataFrame(rows); columns=[f+"_onset" for f in ONSET]+[f+"_peak" for f in PEAK]+[f+"_auc" for f in FEATURES]; X=StandardScaler().fit_transform(desc[columns]); p=PCA().fit(X); keys=["shot_id"]+(["program"] if "program" in desc else []); score=desc[keys].copy(); score["PC1"],score["PC2"]=p.transform(X)[:,:2].T; desc.to_csv(a.descriptor_csv,index=False); score.to_csv(a.score_csv,index=False); print("PC1_PC2_percent",*(100*p.explained_variance_ratio_[:2]))
if __name__=='__main__': main()
