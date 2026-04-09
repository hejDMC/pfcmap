import os
import glob
import h5py
import numpy as np
import pandas as pd
import itertools

def write_h5_dataset(h5file, path, data):
    # Ensure groups exist and write dataset (overwrite if exists)
    parts = [p for p in path.split("/") if p]
    grp = h5file
    for p in parts[:-1]:
        grp = grp.require_group(p)
    name = parts[-1]
    if isinstance(data, np.ndarray) and data.dtype.type is np.str_:
        dt = h5py.string_dtype('utf-8')
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data.astype('U'), dtype=dt)
    else:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data)

meanvar_dir = r"D:\Carlen\Intermediate\preprocessing\metrics_extraction\quantities_all_meanvar\Pete_Rudebeck"
metrics_dir = r"D:\Carlen\Intermediate\metrics_files\metric_files_Pete_Etienne"

meanvar_files = glob.glob(os.path.join(meanvar_dir, "*.h5"))

metrics = ["B","LvR","M","rate"]
statistics = ["mean", "std"]

for F in meanvar_files:
    metric_F = os.path.join(metrics_dir, "metrics_" + os.path.basename(F).split("__")[0] + ".h5")
    tintdur = F.split("prestim")[1].split("_")[0]
    with h5py.File(F, "r") as hf:
        with h5py.File(metric_F, "a") as fid:
            for M, S in itertools.product(metrics, statistics):
                s = hf[f"seg/{M}_{S}"][:]
                write_h5_dataset(fid, f"interval_metrics/{M}/prestim/active/all/dur{tintdur}/{S}/", s.astype(float))
