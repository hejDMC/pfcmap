using Plots
using HDF5
using NaNStatistics
using Statistics

# MAIN PATH
DANDIPATH = "Your/Path/To/DandiSet/"

# FIGURE PATH
FIGPATH = "Your/path/to/figures"

# UTILS FUNCTIONS REQUIRED
include("Utils/simple_raster_units.jl")
include("Utils/simple_PSTH.jl")

# PARAMETERS FRO PLOTTING
# Regions
pfc_regions = ["MOs" "FRP" "ACAd" "ACAv" "PL" "ILA" "ORBm" "ORBvl" "ORBl" "AId" "AIv"]
# Colors
pfc_colors = [colorant"#92A691",  colorant"#4B6A2E", colorant"#e8cd00", 
              colorant"#e5a106", colorant"#CE6161", colorant"#9F453B", 
              colorant"#5A8DAF", colorant"#3D6884", colorant"#505770", 
              colorant"#BA80B6", colorant"#775B8A", colorant"#81AC84"]
# Bin size
bin_sz = 0.01
#window to plot in sec, values here around a sound. 
#Not sure it is the same one as the main figure here though... 
time_in = 798 
time_out = 808
# sampling rates
sr_spikes = 30000
sr_EMG = 1000
# Poster boy recording used in the main figure
file = DANDIPATH*"PL026_20190430-probe0.nwb"

# Get file name and load the NWB
filename = split(file, "/")[end]
println("Opening "*filename)
prefix = split(filename, ".")[1]
nwb = h5open(file, "r")

# Get units spike times, Load jagged arrays
unit_times_data = read(nwb["units/spike_times"]);
unit_times_idx = read(nwb["units/spike_times_index"]);
pushfirst!(unit_times_idx,1);
unit_ids = read(nwb["units/id"]);
spk_times = [unit_times_data[unit_times_idx[i]+1:unit_times_idx[i+1]] for i in 1:length(unit_ids)]
# pushing in first spike time
pushfirst!(spk_times[1],unit_times_data[unit_times_idx[1]])

# Get sound timestamps (10kHz Pure Tones, it will change if you change recording)       
sound = read(nwb["stimulus/presentation/passive_10kHz/data"])
sound_ts = read(nwb["stimulus/presentation/passive_10kHz/timestamps"])
sound_id = fill(10,length(sound_ts))

# get ede labels
ede = read(nwb["general/extracellular_ephys/electrodes/location"])
main_ch = read(nwb["units/electrodes"]).+1

# get DV coordinates
dv = read(nwb["general/extracellular_ephys/electrodes/DV"])
dv_ch = dv[main_ch] 

# get EMG
emg = vec(read(nwb["processing/ecephys/EMG/ElectricalSeries/data"]))
emg = emg[time_in*1000:time_out*1000]
timevec = LinRange(time_in,time_out,(time_out-time_in)*1000+1)

# sort units by DV
new_ii = sortperm(dv_ch)
ylab = ede[main_ch[new_ii]]

# Compute simple raster
X,Y = simple_raster_units(spk_times[new_ii], 0.0, time_in, time_out)
Y = Int.(Y)

# Make PSTH
cnt = histcounts(X,time_in:bin_sz:time_out)
psth = cnt./length(spk_times)./bin_sz
timevec2 = LinRange(time_in,time_out,(time_out-time_in)*100)

# Find delim values in regions
ss = fill(0,length(pfc_colors))
    for (R,r) in enumerate(pfc_regions)
        ss[R] = length( findall(contains(pfc_regions[R]),ylab))
    end
css = fill(length(spk_times),length(pfc_colors)) - cumsum(ss)

# Plot spike raster
sc = scatter(X,Y, 
        mc=:black, ms=2, markerstrokewidth=0,
        #xlabel= "time [s]", 
        yticks=(1:10:length(spk_times),
        ylab[1:10:length(spk_times)]),
        ylabel= "region",
        legend=false,
        grid = false)

    for (R,r) in enumerate(pfc_regions)
        for ii in findall(contains(pfc_regions[R]),ylab)
        scatter!(X[Y.==ii],Y[Y.==ii], mc=pfc_colors[R], ms=2, markerstrokewidth=0) 
        end
        hline!(css[R],color=:black)
    end

    if !isempty(findall(x->x>time_in&&x<time_out, sound_ts))
        for tsi in findall(x->x>time_in&&x<time_out, sound_ts)
            vline!([sound_ts[tsi]], color=:black)
            vline!([sound_ts[tsi].-3], color=:grey)
        end
    end

# PLot EMG
emg_plt = plot(timevec,emg,color=:black,
               ylabel="EMG", xlabel="time [s]",legend = false)
               if !isempty(findall(x->x>time_in&&x<time_out, sound_ts))
                    for tsi in findall(x->x>time_in&&x<time_out, sound_ts)
                        vline!([sound_ts[tsi]], color=:black)
                        vline!([sound_ts[tsi].-3], color=:grey)
                    end
                end


# Assemble the complete figure
plt = plot(sc,emg_plt,layout=grid(2,1, heights=[0.95,0.05]),
            size=(1200,800))
display(plt)    

# save figure
savefig(plt,FIGPATH*"figure.svg")
