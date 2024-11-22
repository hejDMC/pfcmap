using Plots
using ProgressMeter
using Glob
using HDF5
using GaussianMixtures
using StatsBase
using Distributions
using NaNStatistics
using Random
Random.seed!(42)

# MAIN PATH
ZENODOPATH = "Your/Path/To/Zenodo/Data" 

# FIGURE PATH
FIGPATH = "Your/path/to/figures"

# Hard-coded values used for the Nw Ww split
# Current values
low_th = 0.000380
high_th = 0.000430

# Get filelist in ZENODO
filelist = glob("*.h5",ZENODOPATH)

# Filtering Parameters
Qual = 1;
WaveformQual = 1;

# initialize array
FR = Array{Float64}(undef, 1)
P2T = Array{Float64}(undef, 1)
AB = Array{Float64}(undef, 1)
WW = Array{Int64}(undef, 1)
NW = Array{Int64}(undef, 1)

prog = Progress(length(filelist), 1, "Looping through metric files...", length(filelist))


    # Main loop
    for file in filelist
            #println(file)
            #file = filelist[1]
            metrics = h5open(file,"r")

            if haskey(metrics,"interval_metrics/rate/prestim/active/all/dur3/mean")

            p2t = read(metrics["waveform_metrics/peak2Trough"])
            ab = read(metrics["waveform_metrics/abRatio"])
            rate = read(metrics["interval_metrics/rate/prestim/active/all/dur3/mean"])
            b_b = read(metrics["interval_metrics/B/prestim/active/all/dur3/mean"])
            m_b = read(metrics["interval_metrics/M/prestim/active/all/dur3/mean"])
            ww = read(metrics["unit_type/ww"])
            nw = read(metrics["unit_type/nw"])

            # filter
            q = read(metrics["quality/quality"])
            wq = read(metrics["waveform_metrics/waveform_quality"])
        
            # remove with quality, waveform qaulity, waveform type and any NaN in the metrics 
            idx2rm = (q.!=Qual) .| (wq.!=WaveformQual) .| (isnan.(rate).==1) .| (isnan.(b_b).==1) .| (isnan.(m_b).==1)
            
            deleteat!(p2t, idx2rm)
            deleteat!(ab, idx2rm)
            deleteat!(rate, idx2rm)
            deleteat!(b_b, idx2rm)
            deleteat!(m_b, idx2rm)
            deleteat!(ww, idx2rm)
            deleteat!(nw, idx2rm)
        
            append!(P2T, p2t)
            append!(AB, ab)
            append!(FR, rate)
            append!(WW, ww)
            append!(NW, nw)

            else

                println(file*" has no interval metrics")

            end
            
            next!(prog)

    end

# delete first values
popfirst!(P2T)
popfirst!(AB)
popfirst!(FR)
popfirst!(WW)
popfirst!(NW)

# remove NaN
remove_idx = isnan.(P2T).==1
deleteat!(WW,remove_idx)
deleteat!(NW,remove_idx)
deleteat!(FR,remove_idx)
deleteat!(AB,remove_idx)
deleteat!(P2T,remove_idx)

# Plot
h1 = histogram(P2T, xlims=(0,0.0015), bins=0:0.0015/150:0.0015, color="#C49A6C", ylabel="unit count", xlabel="peak-to-through duration [s]", label = "unclassified")
histogram!(P2T[(P2T.>high_th).==1], xlims=(0,0.0015), bins=0:0.0015/150:0.0015, color=:black, label = "ww")
histogram!(P2T[(P2T.<low_th).==1], xlims=(0,0.0015), bins=0:0.0015/150:0.0015, color="#A53F97", label = "nw")
plt = plot(h1)
display(plt)

# save figure
savefig(plt,FIGPATH*"figure.svg")