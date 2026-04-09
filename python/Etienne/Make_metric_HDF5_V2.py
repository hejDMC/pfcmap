import os
import glob
import h5py
import numpy as np
import pandas as pd


Dataset = "Pete_Rudebeck"  # "Carlen", "IBL_Passive" "Pete_Rudebeck" "Ben_Hayden"

# Dest to save the hdf5 files
dest = "/Volumes/labs/dmclab/Pierre/PFCmap/metric_files_" + Dataset + "/"
os.makedirs(dest, exist_ok=True)

# Make the metric list
src = "/Volumes/labs/dmclab/Pierre/PFCmap/data/"
dir_list = sorted(os.listdir(src))
name2rmv = [
    ".DS_Store",
    "FiringRate_detail",
    "FiringRate_detail2",
    "GLM",
    "PFC_hierarchy_score.csv",
    "TCA",
    "TCA_factor_list.csv",
    "TCA_factor_list.xls",
    "TCA_factor_list.xlsx",
    "burst",
    "cv2-pierre",
    "dataset.csv",
    "dataset2.csv",
    "dataset3.csv",
    "Dataset_IBL_Passive.csv",
    "Dataset_Carlen.csv",
    "lick",
    "Anatomy.zip",
    "Bad_Waveform_list_Carlen.csv",
    "Bad_Waveform_list_Carlen2.csv",
    "Bad_Waveform_list_IBL.csv",
    "Bad_Waveform_list_Steinmetz.csv",
    "waveform_quality",
    "waveforms",
    "zeta",
    "zeta_passive",
]
dir_list = [d for d in dir_list if d not in name2rmv]

# The original Julia used fixed indices into dir_list. Keep same index mapping (1-based -> 0-based)
def pick(indices):
    return [dir_list[i] for i in indices]

# Indices adjusted from Julia (1-based) to Python (0-based)
dir_ls_param = pick([0, 34, 49, 40])
dir_ls_wav = pick([11, 20, 24, 25, 26, 28, 29, 30, 35, 36, 37, 38, 43, 44, 47])
dir_ls_int = pick([1, 8, 9, 41])
dir_ls_other = pick([23, 27, 31, 32])


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


if Dataset == "Carlen":
    src1 = "/Volumes/labs/dmclab/Pierre/NPX_Database/mPFC/"
    src2 = "/Volumes/labs/dmclab/Pierre/NPX_Database/AUD/"
    src3 = "/Volumes/labs/dmclab/Pierre/NPX_Database/SS/"
    src4 = "/Volumes/labs/dmclab/Pierre/NPX_Database/MO/"
    src5 = "/Volumes/labs/dmclab/Pierre/NPX_Database/HPC/"
    filelist1 = glob.glob(os.path.join(src1, "*", "*.nwb"))
    filelist2 = glob.glob(os.path.join(src2, "*", "*.nwb"))
    filelist3 = glob.glob(os.path.join(src3, "*", "*.nwb"))
    filelist4 = glob.glob(os.path.join(src4, "*", "*.nwb"))
    filelist5 = glob.glob(os.path.join(src5, "*", "*.nwb"))
    nwb_ls = filelist1 + filelist2 + filelist3 + filelist4 + filelist5
elif Dataset == "IBL_Passive":
    src_nwb = "/Volumes/labs/dmclab/Pierre/NPX_External_Databases/" + Dataset + "/"
    nwb_ls = glob.glob(os.path.join(src_nwb, "*.nwb"))
elif Dataset in ("Pete_Rudebeck", "Ben_Hayden"):
    src_nwb = "/Volumes/labs/dmclab/Pierre/NPX_External_Databases/" + Dataset + "/NWB/"
    nwb_ls = glob.glob(os.path.join(src_nwb, "*.nwb"))
else:
    nwb_ls = []


if Dataset == "Carlen":
    src_int = (
        "/Volumes/labs/dmclab/katharina/PFCmap/DATA/results/timescales/quantities_all_meanvar/Carlen_quietactive/"
    )
    int_list = glob.glob(os.path.join(src_int, "*.h5"))
elif Dataset == "IBL_Passive":
    src_int = (
        "/Volumes/labs/dmclab/katharina/PFCmap/DATA/results/timescales/quantities_all_meanvar/IBL_Passive/"
    )
    int_list = glob.glob(os.path.join(src_int, "*.h5"))
elif Dataset in ("Pete_Rudebeck", "Ben_Hayden"):
    src_int = "/Volumes/labs/dmclab/Pierre/PFCmap_NHP/quantities_meanvar/"
    int_list = glob.glob(os.path.join(src_int, "*.h5"))
else:
    src_int = ""
    int_list = []


for F in nwb_ls:
    with h5py.File(F, "r") as NWB:
        filename = os.path.splitext(os.path.basename(F))[0]
        print(f"Processing file {filename}")
        fid_path = os.path.join(dest, f"metrics_{filename}.h5")
        with h5py.File(fid_path, "w") as fid:
            # Dataset
            write_h5_dataset(fid, "Dataset/", np.array([Dataset], dtype='S'))

            # Task
            if Dataset == "Carlen":
                csv_path = "/Volumes/labs/dmclab/Pierre/PFCmap/data/dataset2.csv"
                csv = pd.read_csv(csv_path, header=0)
                # find row where second column equals filename
                matches = csv.iloc[:, 1] == filename
                idx = matches[matches].index
                if len(idx) > 0:
                    task = str(csv.iloc[idx[0], 3])
                else:
                    task = ""
                write_h5_dataset(fid, "Task/", np.array([task], dtype='S'))
            elif Dataset == "IBL_Passive":
                write_h5_dataset(fid, "Task/", np.array(["IBL_Passive"], dtype='S'))
            elif Dataset == "Pete_Rudebeck":
                write_h5_dataset(fid, "Task/", np.array(["Pete_Rudebeck"], dtype='S'))
            elif Dataset == "Ben_Hayden":
                write_h5_dataset(fid, "Task/", np.array(["Ben_Hayden"], dtype='S'))

            # Units identifiers
            uids = NWB["units/id"][:]
            write_h5_dataset(fid, "uids/", uids.astype(np.int64))

            # Anatomy and other metadata per-dataset handling
            if Dataset == "Carlen":
                fid.require_group("anatomy")
                csv_anat = os.path.join(src, dir_ls_param[0], Dataset, f"{dir_ls_param[0]}_{filename}.csv")
                if os.path.isfile(csv_anat):
                    df = pd.read_csv(csv_anat, header=0)
                    df = df.iloc[:, 1:]
                    for nn in df.columns:
                        col = df[nn]
                        if pd.api.types.is_numeric_dtype(col.dtype):
                            write_h5_dataset(fid, f"anatomy/{nn}", col.to_numpy(dtype=float))
                        else:
                            write_h5_dataset(fid, f"anatomy/{nn}", col.astype(str).to_numpy())

                # Unit type
                for g, G in enumerate(dir_ls_param[1:3], start=1):
                    csv_path = os.path.join(src, G, "All", f"{G}_{filename}.csv")
                    if os.path.isfile(csv_path):
                        df = pd.read_csv(csv_path, header=0)
                        df = df.iloc[:, 1:]
                        for nn in df.columns:
                            col = df[nn]
                            if pd.api.types.is_numeric_dtype(col.dtype):
                                write_h5_dataset(fid, f"unit_type/{nn}", col.to_numpy(dtype=np.int64))
                            else:
                                write_h5_dataset(fid, f"unit_type/{nn}", col.astype(str).to_numpy())

                # Quality
                csv_q = os.path.join(src, dir_ls_param[3], Dataset, f"{dir_ls_param[3]}_{filename}.csv")
                if os.path.isfile(csv_q):
                    df = pd.read_csv(csv_q, header=0).iloc[:, 1:]
                    for nn in df.columns:
                        col = df[nn]
                        if pd.api.types.is_numeric_dtype(col.dtype):
                            write_h5_dataset(fid, f"quality/{nn}", col.to_numpy(dtype=np.int64))
                        else:
                            write_h5_dataset(fid, f"quality/{nn}", col.astype(str).to_numpy())

                # Waveform metrics
                for g, G in enumerate(dir_ls_wav):
                    csv_w = os.path.join(src, dir_ls_wav[g], "All", f"{dir_ls_wav[g]}_{filename}.csv")
                    if os.path.isfile(csv_w):
                        df = pd.read_csv(csv_w, header=0).iloc[:, 1:]
                        for nn in df.columns:
                            col = df[nn]
                            if pd.api.types.is_numeric_dtype(col.dtype):
                                write_h5_dataset(fid, f"waveform_metrics/{nn}", col.to_numpy(dtype=float))
                            else:
                                write_h5_dataset(fid, f"waveform_metrics/{nn}", col.astype(str).to_numpy())

            elif Dataset in ("Pete_Rudebeck", "Ben_Hayden"):
                fid.require_group("anatomy")
                ede = NWB["general/extracellular_ephys/electrodes/location"][:]
                main_ch = NWB["units/electrodes"][:] + 1
                # select locations
                loc = ede[main_ch]
                df = pd.DataFrame({"location": loc})
                for nn in df.columns:
                    col = df[nn]
                    if pd.api.types.is_numeric_dtype(col.dtype):
                        write_h5_dataset(fid, f"anatomy/{nn}", col.to_numpy(dtype=float))
                    else:
                        write_h5_dataset(fid, f"anatomy/{nn}", col.astype(str).to_numpy())

                # Unit type
                for g, G in enumerate(dir_ls_param[1:3], start=1):
                    if Dataset == "Pete_Rudebeck":
                        src2 = "/Volumes/labs/dmclab/Pierre/NPX_External_Databases/Pete_Rudebeck/Unit_types/"
                        csv_path = os.path.join(src2, G, f"{G}_{filename}.csv")
                        if os.path.isfile(csv_path):
                            df = pd.read_csv(csv_path, header=0).iloc[:, 1:]
                            for nn in df.columns:
                                col = df[nn]
                                if pd.api.types.is_numeric_dtype(col.dtype):
                                    write_h5_dataset(fid, f"unit_type/{nn}", col.to_numpy(dtype=np.int64))
                                else:
                                    write_h5_dataset(fid, f"unit_type/{nn}", col.astype(str).to_numpy())
                    elif Dataset == "Ben_Hayden":
                        # no type put defaults
                        if g == 1:
                            arr = np.ones(len(loc), dtype=np.int64)
                            write_h5_dataset(fid, f"unit_type/ww", arr)
                        elif g == 2:
                            arr = np.zeros(len(loc), dtype=np.int64)
                            write_h5_dataset(fid, f"unit_type/nw", arr)

                # Quality
                write_h5_dataset(fid, "quality/quality", np.ones(len(loc), dtype=np.int64))

                # Waveform metrics
                for g, G in enumerate(dir_ls_wav):
                    if Dataset == "Pete_Rudebeck":
                        src2 = "/Volumes/labs/dmclab/Pierre/NPX_External_Databases/Pete_Rudebeck/Unit_types/"
                        csv_w = os.path.join(src2, dir_ls_wav[g], f"{dir_ls_wav[g]}_{filename}.csv")
                        if os.path.isfile(csv_w):
                            df = pd.read_csv(csv_w, header=0).iloc[:, 1:]
                            for nn in df.columns:
                                col = df[nn]
                                if pd.api.types.is_numeric_dtype(col.dtype):
                                    write_h5_dataset(fid, f"waveform_metrics/{nn}", col.to_numpy(dtype=float))
                                else:
                                    write_h5_dataset(fid, f"waveform_metrics/{nn}", col.astype(str).to_numpy())
                    elif Dataset == "Ben_Hayden":
                        # create empty/NaN strings for missing waveform info
                        df = pd.DataFrame()
                        if g < 16:
                            df[dir_ls_wav[g]] = ["NaN"] * len(loc)
                        for nn in df.columns:
                            col = df[nn]
                            write_h5_dataset(fid, f"waveform_metrics/{nn}", col.astype(str).to_numpy())

                # Waveform quality
                write_h5_dataset(fid, "waveform_metrics/waveform_quality", np.ones(len(loc), dtype=np.int64))

            # Interval metrics handling
            if Dataset == "Carlen":
                for g, G in enumerate(dir_ls_int):
                    # dur1 active all
                    fpath = os.path.join(src_int, f"{filename}__TSELprestim1__STATEactive__all_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur1/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur1/std/", s.astype(float))

                    fpath = os.path.join(src_int, f"{filename}__TSELprestim1__STATEactive_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur1/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur1/std/", s.astype(float))

                    # dur2
                    fpath = os.path.join(src_int, f"{filename}__TSELprestim2__STATEactive__all_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur2/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur2/std/", s.astype(float))

                    fpath = os.path.join(src_int, f"{filename}__TSELprestim2__STATEactive_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur2/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur2/std/", s.astype(float))

                    # dur3
                    fpath = os.path.join(src_int, f"{filename}__TSELprestim3__STATEactive__all_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/std/", s.astype(float))

                    fpath = os.path.join(src_int, f"{filename}__TSELprestim3__STATEactive_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur3/std/", s.astype(float))

            elif Dataset == "IBL_Passive":
                for g, G in enumerate(dir_ls_int):
                    fpath = os.path.join(src_int, f"{filename}__TSELpassive3_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/std/", s.astype(float))

            elif Dataset == "Pete_Rudebeck":
                for g, G in enumerate(dir_ls_int):
                    fpath = os.path.join(src_int, f"{filename}__TSELprestim3_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/std/", s.astype(float))

            elif Dataset == "Ben_Hayden":
                for g, G in enumerate(dir_ls_int):
                    fpath = os.path.join(src_int, f"{filename}__TSELactive3s_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/active/all/dur3/std/", s.astype(float))

                    fpath = os.path.join(src_int, f"{filename}__TSELpassive3s_quantities_meanvar.h5")
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, "r") as ff:
                            m = ff[f"seg/{dir_ls_int[g]}_mean"][:]
                            s = ff[f"seg/{dir_ls_int[g]}_std"][:]
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur3/mean/", m.astype(float))
                        write_h5_dataset(fid, f"interval_metrics/{dir_ls_int[g]}/prestim/passive/all/dur3/std/", s.astype(float))


if __name__ == "__main__":
    print("Translation complete. Run the script to generate HDF5 metric files.")
