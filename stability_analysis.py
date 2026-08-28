"""Summarize completed strict stability metrics exported by the SI analysis.

Usage: python stability_analysis.py same_shot_metrics.csv between_program_metrics.csv summary.csv
"""
import argparse
import pandas as pd
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('same_shot_csv'); ap.add_argument('between_program_csv'); ap.add_argument('output_csv'); a=ap.parse_args()
    w=pd.read_csv(a.same_shot_csv); b=pd.read_csv(a.between_program_csv)
    rows=[]
    for tr in sorted(set(w.trajectory)&set(b.trajectory)):
        x=w.loc[w.trajectory.eq(tr),'rmse_vs_shot_solution_median'].median(); y=b.loc[b.trajectory.eq(tr),'rmse_between_family_median_trajectories'].median()
        rows.append({'trajectory':tr,'within_solution_rmse_median':x,'between_program_rmse_median':y,'between_to_within_rmse_ratio':y/x})
    pd.DataFrame(rows).to_csv(a.output_csv,index=False); print(f"trajectories={len(rows)}")
if __name__=='__main__': main()
