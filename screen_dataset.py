"""Screen a one-row-per-shot summary table before fitting."""
import argparse
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input_csv"); ap.add_argument("output_csv")
    ap.add_argument("--max-duration",type=float,default=80.); ap.add_argument("--max-final-weight",type=float,default=80.); ap.add_argument("--max-onset",type=float,default=20.)
    a=ap.parse_args(); d=pd.read_csv(a.input_csv)
    required={"n_samples","duration_s","final_weight_g","onset_s","max_pressure_bar","max_flow_out_ml_s","monotonicity_pass","apparent_resistance_pass"}
    missing=sorted(required-set(d.columns))
    if missing: raise ValueError("missing screening columns: "+", ".join(missing))
    ok=(d.n_samples>=20)&d.duration_s.between(8,a.max_duration)&d.final_weight_g.between(10,a.max_final_weight)&d.onset_s.between(0,a.max_onset)
    truth=lambda s: s.astype(str).str.lower().isin(["1","true","yes","y"])
    ok &= (d.max_pressure_bar>=0)&(d.max_flow_out_ml_s>=0)&truth(d.monotonicity_pass)&truth(d.apparent_resistance_pass)
    d.assign(screen_pass=ok).to_csv(a.output_csv,index=False); print(f"screened={len(d)} eligible={int(ok.sum())}")
if __name__=='__main__': main()
