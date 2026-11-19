"""
risk_diagnostics.py
Temporary helper: shows what drives the risk score in every saved timeline CSV.
Run from the project root:   python risk_diagnostics.py
It only READS files in outputs\\predictions and changes nothing.
"""
import glob
import os

import pandas as pd

# Same weights as src\risk_analyzer.py (W_PED_VEHICLE, W_VEHICLE_VEHICLE, W_DENSITY, W_SIGN_CONTEXT)
WEIGHTS = {"ped_vehicle": 1.0, "vehicle_vehicle": 0.50, "density": 0.0, "sign_context": 0.10}

files = sorted(glob.glob(os.path.join("outputs", "predictions", "*risk_timeline*.csv")))
if not files:
    print("No timeline CSVs found in outputs\\predictions. Run the pipeline on a video first.")
    raise SystemExit(1)

for f in files:
    d = pd.read_csv(f)
    print("=" * 72)
    print(f"{os.path.basename(f)}  ({len(d)} frames)")
    print(f"objects per frame : mean {d['n_objects'].mean():.1f}, max {d['n_objects'].max()}")
    print("frames per level  : " + ", ".join(
        f"{lv} {100 * (d['level'] == lv).mean():.0f}%" for lv in ("LOW", "MEDIUM", "HIGH")))
    print(f"score             : mean {d['score'].mean():.2f}, min {d['score'].min():.2f}, max {d['score'].max():.2f}")
    print("factors (0 to 1, after the persistence ramp):")
    for c, w in WEIGHTS.items():
        print(f"   {c:<16} mean {d[c].mean():.2f} | above 0.5 in {100 * (d[c] > 0.5).mean():3.0f}% of frames"
              f" | average contribution to raw score {w * d[c].mean():.2f}")
print("=" * 72)