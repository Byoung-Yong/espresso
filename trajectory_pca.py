"""Extract five trajectory AUC features and PCA scores."""
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
FEATURES=["wetting","retained_liquid","swelling","resistance_relief","viscosity_factor"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("trajectory_csv"); ap.add_argument("descriptor_csv"); ap.add_argument("score_csv"); a=ap.parse_args(); d=pd.read_csv(a.trajectory_csv)
    need={"shot_id","progress",*FEATURES}
    if not need.issubset(d): raise ValueError("missing trajectory columns")
    rows=[]
    for sid,g in d.groupby("shot_id",sort=False):
        g=g.sort_values("progress"); row={"shot_id":sid}
        if "program" in g: row["program"]=g.program.iloc[0]
        row.update({f+"_auc":float(np.trapz(g[f],g.progress)) for f in FEATURES}); rows.append(row)
    desc=pd.DataFrame(rows); X=StandardScaler().fit_transform(desc[[f+"_auc" for f in FEATURES]]); p=PCA().fit(X); keys=["shot_id"]+(["program"] if "program" in desc else []); score=desc[keys].copy(); score["PC1"],score["PC2"]=p.transform(X)[:,:2].T; desc.to_csv(a.descriptor_csv,index=False); score.to_csv(a.score_csv,index=False); print("PC1_PC2_percent",*(100*p.explained_variance_ratio_[:2]))
if __name__=='__main__': main()
