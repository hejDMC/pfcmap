import os
import glob
import h5py
import numpy as np
import pandas as pd
from rich.progress import Progress

# Thresholds
low_th = 0.380
high_th = 0.430

# Paths
basepath = "D:/Carlen/Pete_Rudebeck/Unit_types/"
# NWB directory (adjust if your NWBs are elsewhere)
nwb_dir = "D:/Carlen/Pete_Rudebeck/NWB"
nwb_list = glob.glob(os.path.join(nwb_dir, "*.nwb"))

with Progress() as progress:
    task = progress.add_task("[cyan]Processing files...", total=len(nwb_list))
    for F in nwb_list:
        progress.update(task, description=f"[cyan]Processing {os.path.basename(F)}")
        with h5py.File(F, "r") as hf:
            # read Peak_to_Trough and unit ids from the NWB HDF5
            # dataset path expected: /units/Peak_to_Trough
            if "/units/Peak_to_Trough" in hf:
                p2b = np.array(hf["/units/Peak_to_Trough"])[:]
            else:
                # try alternative key or missing data
                ds = hf.get("/units/Peak_to_Trough") or hf.get("units/Peak_to_Trough") or hf.get("/units/peak_to_trough")
                p2b = np.array(ds)[:] if ds is not None else np.array([])

            # read ids if present
            if "/units/id" in hf:
                ids = np.array(hf["/units/id"])[:]
                # decode bytes if necessary
                if ids.dtype.type is np.bytes_:
                    ids = ids.astype(str)
            else:
                # fallback: generate numeric ids
                ids = np.arange(len(p2b)).astype(str)

        # coerce to numpy array and ensure lengths match
        p2b = np.asarray(p2b, dtype=float)
        ids = np.asarray(ids)

        if p2b.size != ids.size:
            # try to trim or pad to match lengths; prefer trimming
            n = min(p2b.size, ids.size)
            p2b = p2b[:n]
            ids = ids[:n]

        nw = (p2b < low_th).astype(int)
        ww = (p2b > high_th).astype(int)

        # build suffix from filename parts (skip the first part)
        bn = os.path.basename(F)
        parts = bn.split("_")
        suffix = "_".join(parts[1:]) if len(parts) > 1 else parts[0]

        out_nw = os.path.join(basepath, "nw", f"nw_{suffix}")
        out_ww = os.path.join(basepath, "ww", f"ww_{suffix}")

        # ensure .csv extension
        if not out_nw.lower().endswith('.csv'):
            out_nw = out_nw + '.csv'
        if not out_ww.lower().endswith('.csv'):
            out_ww = out_ww + '.csv'

        pd.DataFrame({"id": ids, "nw": nw}).to_csv(out_nw, index=False)
        pd.DataFrame({"id": ids, "ww": ww}).to_csv(out_ww, index=False)
        progress.advance(task)

print("Conversion complete")
