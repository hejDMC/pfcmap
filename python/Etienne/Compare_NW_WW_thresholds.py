import os
import glob
import h5py
import numpy as np
import pandas as pd
from rich.progress import Progress

nwb_path = "D:/Carlen/Pete_Rudebeck/NWB"
Pierre_path = "D:/Carlen/Pete_Rudebeck/Unit_types_Pierre"

# Thresholds
low_th = 0.380
high_th = 0.430

# Paths
basepath = "D:/Carlen/Pete_Rudebeck/Unit_types/"
# NWB directory (adjust if your NWBs are elsewhere)
nwb_dir = "D:/Carlen/Pete_Rudebeck/NWB"
nwb_list = glob.glob(os.path.join(nwb_dir, "*.nwb"))

max_p2b_nw = 0.
min_p2b_ww = np.inf
with Progress() as progress:
    task = progress.add_task("[cyan]Processing files...", total=len(nwb_list))
    for F in nwb_list:
        progress.update(task, description=f"[cyan]Processing {os.path.basename(F)}")
        with h5py.File(F, "r") as hf:
            # read Peak_to_Trough and unit ids from the NWB HDF5
            # dataset path expected: /units/Peak_to_Trough
            p2b = np.array(hf["/units/PTR"])[:]

            # read ids if present
            if "/units/id" in hf:
                ids = np.array(hf["/units/id"])[:]
                # decode bytes if necessary
                if ids.dtype.type is np.bytes_:
                    ids = ids.astype(str)
            else:
                # fallback: generate numeric ids
                ids = np.arange(len(p2b)).astype(str)

        # Read Pierre's file to check how NW and WW have been classified
        pierre_nw = os.path.join(Pierre_path, "nw", "nw_" + os.path.basename(F).split(".nwb")[0] + ".csv")
        pierre_ww = os.path.join(Pierre_path, "ww", "ww_" + os.path.basename(F).split(".nwb")[0] + ".csv")

        df_pierre_nw = pd.read_csv(pierre_nw, header=0)
        df_pierre_ww = pd.read_csv(pierre_ww, header=0)

        # Find the maximum p2b value for NW and minimum for WW according to Pierre's classification
        if np.any(df_pierre_nw["nw"].values):
            max_p2b_nw = np.maximum(max_p2b_nw, np.max(p2b[df_pierre_nw["nw"].values]))
        if np.any(df_pierre_ww["ww"].values):
            min_p2b_ww = np.minimum(min_p2b_ww, np.min(p2b[df_pierre_ww["ww"].values]))

        progress.advance(task)

print(f"threshold for NW: {max_p2b_nw}")
print(f"threshold for WW: {min_p2b_ww}")
