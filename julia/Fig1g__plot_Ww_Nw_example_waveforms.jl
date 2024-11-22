using Plots
using ProgressMeter
using Glob
using HDF5
using Statistics
using Random
Random.seed!(42)

# MAIN PATH
ZENODOPATH = "Your/Path/To/Zenodo/Data" 
DANDIPATH = "Your/Path/To/DandiSet/"
DANDIPATH2 = "Your/Path/To/DandiSet2/"

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
# Number of waveform to plot in each plot
number_of_wfs_2_plot = 100;

# initialize array
FR = Array{Float64}(undef, 1)
P2T = Array{Float64}(undef, 1)
AB = Array{Float64}(undef, 1)
WW = Array{Int64}(undef, 1)
NW = Array{Int64}(undef, 1)
WF = Array{Int64}(undef, 82, 1)

prog = Progress(length(filelist), 1, "Looping through metric and NWB files...", length(filelist))

    # Main loop
    for file in filelist
            #println(file)
            #file = filelist[1]
            metrics = h5open(file,"r")

            if haskey(metrics,"interval_metrics/rate/prestim/active/all/dur3/mean")    

            task = read(metrics["Task"])

            filename = split(split(file,'/')[end],'_')[2]*"_"*split(split(split(file,'/')[end],'_')[3],'.')[1]
            
            if isfile(DANDIPATH*"/"*filename*".nwb")
                nwb = h5open(DANDIPATH*"/"*filename*".nwb","r")
            else
                nwb = h5open(DANDIPATH2*"/"*filename*".nwb","r")
            end

            p2t = read(metrics["waveform_metrics/peak2Trough"])
            ab = read(metrics["waveform_metrics/abRatio"])
            rate = read(metrics["interval_metrics/rate/prestim/active/all/dur3/mean"])
            b_b = read(metrics["interval_metrics/B/prestim/active/all/dur3/mean"])
            m_b = read(metrics["interval_metrics/M/prestim/active/all/dur3/mean"])
            ww = read(metrics["unit_type/ww"])
            nw = read(metrics["unit_type/nw"])

            # load waveforms form raw NWB file
            if haskey(nwb,"units/waveform_means") 
                wf_mean = read(nwb["units/waveform_means"])
            elseif haskey(nwb,"units/waveform_mean")
                wf_mean = read(nwb["units/waveform_mean"])    
            end
            #downsample wf_mean
            subsample_wf = [wf_mean[i,j] for i=1:1000:size(wf_mean,1), j = 1:size(wf_mean,2)]

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
            filter_wf = subsample_wf[:,idx2rm.==0]
        
            append!(P2T, p2t)
            append!(AB, ab)
            append!(FR, rate)
            append!(WW, ww)
            append!(NW, nw)
            global WF = hcat(WF, filter_wf)

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
WF = WF[:,2:end]

# remove NaN
remove_idx = isnan.(P2T).==1
deleteat!(WW,remove_idx)
deleteat!(NW,remove_idx)
deleteat!(FR,remove_idx)
deleteat!(AB,remove_idx)
deleteat!(P2T,remove_idx)
WF = WF[:,remove_idx.==0]

# Plots
norm_factor = maximum(WF,dims=1).-minimum(WF,dims=1)
NORM_WF = WF./norm_factor
# WW
WF_WW = NORM_WF[:,WW.==1]
nbofww = size(WF_WW,2)
idxww = Int.(round.(rand(number_of_wfs_2_plot).*nbofww))
p1 = plot(WF_WW[:,idxww], color=:gray, legend = false)
plot!(mean(WF_WW,dims=2), color=:black, label ="ww")
#NW
WF_NW = NORM_WF[:,NW.==1]
nbofnw = size(WF_NW,2)
idxnw = Int.(round.(rand(number_of_wfs_2_plot).*nbofnw))
p2 = plot(WF_NW[:,idxnw], color=:gray, legend = false)
plot!(mean(WF_NW,dims=2), color="#A53F97", label ="nw")

plt = plot(p1,p2, layout=grid(2,1))
display(plt)

# save figure
savefig(plt,FIGPATH*"figure.svg")