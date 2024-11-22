using Glob
using HDF5
using ProgressMeter
using DataFrames
using MixedModels, StatsModels, Statistics
using StatsPlots
using RCall, JellyMe4 #Get R packages to interpret the LME
R"library(lme4)"
R"library(emmeans)"
R"library(pbkrtest)"
R"library(MuMIn)"

# MAIN PATH
ZENODOPATH = "Your/Path/To/Zenodo/Data" 

# FIGURE PATH
FIGPATH = "Your/path/to/figures"

# Filtering Parameters
Min_FR = 0.1;
Qual = 1;
WaveformQual = 1;

# states to compare
states = ["active" "passive"]

# Get filelist in ZENODO
filelist = glob("*.h5",ZENODOPATH)

prog = Progress(length(filelist), 1, "Collecting metrics...")

for (f,F) in enumerate(filelist)


    h = h5open(F,"r")
    filename = split(split(F,'/')[end],'.')[1][9:end]

    for s in eachindex(states)

        state = states[s]

        if haskey(h,"interval_metrics/rate/prestim/"*state*"/all/dur3/mean")
            
            rate = read(h["interval_metrics/rate/prestim/"*state*"/all/dur3/mean"])
            b = read(h["interval_metrics/B/prestim/"*state*"/all/dur3/mean"])
            m = read(h["interval_metrics/M/prestim/"*state*"/all/dur3/mean"])
            q = read(h["quality/quality"])
            wq = read(h["waveform_metrics/waveform_quality"])
            nw = read(h["unit_type/nw"])
            ww = read(h["unit_type/ww"])
            sub_br = read(h["anatomy/location"])
            br = string.([split(sub_br[i],r"1|2|5|6")[1] for i in 1:length(sub_br)])
            ap = read(h["anatomy/ap"])
            ml = read(h["anatomy/ml"])
            dv = read(h["anatomy/dv"])
            u = read(h["anatomy/U"])
            v = read(h["anatomy/V"])
            roi = read(h["anatomy/roi"])
            mouse = string.(split(split(split(split(F,'/')[8],"metrics_")[2],'.')[1],"-")[1])
            probe = string.(split(split(split(F,'/')[8],"metrics_")[2],'.')[1])
            dataset = read(h["Dataset"])
            task = read(h["Task"])

            # filter units with min FR and quality
            idx2rm = (rate.<Min_FR) .| (q.!=Qual) .| (wq.!=WaveformQual) .| (isnan.(rate).==1) .| (isnan.(b).==1) .| (isnan.(m).==1)

            deleteat!(rate, idx2rm)
            deleteat!(b, idx2rm)
            deleteat!(m, idx2rm)
            deleteat!(sub_br, idx2rm)
            deleteat!(br, idx2rm)
            deleteat!(ap, idx2rm)
            deleteat!(ml, idx2rm)
            deleteat!(dv, idx2rm)
            deleteat!(u, idx2rm)
            deleteat!(v, idx2rm)
            deleteat!(roi, idx2rm)
            deleteat!(ww, idx2rm)
            deleteat!(nw, idx2rm)

            # Append to Dataframe
            df1 = DataFrame(Mouse=mouse, Region=br, SubRegion=sub_br, AP=ap, ML=ml, DV=dv, U=u, V=v, ROI=roi, Dataset=dataset, Task=task, State=state, Rate=rate, B=b, M=m, WW=string.((ww.+(nw.*-1)).+1))
            # Initialize DataFrame
            if f == 1
                global df = df1
            else
                df = append!(df, df1)
            end

        else
                
            println("Missing interval_metrics/rate/prestim/"*state*"/all in "* F)

        end

    end

    next!(prog)

end


# remove NaN before going further
filter!(row -> all(x -> !(x isa Number && isnan(x)), row), df)
filter!(row -> all(x -> !(x isa Number && isinf(x)), row), df)

#WW/NW comparison
# Random effect from mice on Rate with unit type as a fixed effect
fm4 = @formula(Rate ~ 1 + WW + (1|Mouse))
m4 = fit(LinearMixedModel, fm4, df)
# post hoc comaprison
# Degrees-of-freedom method: kenward-roger 
# P value adjustment: tukey method for comparing a family of 11 estimates (Regions)
m4_df = (m4, df)
@rput m4_df
R"summary(m4_df)"
# get the contrast result into a dataframe
emm4 = rcopy(R"summary(emmeans(m4_df, pairwise ~ WW, pbkrtest.limit = 20000)$contrasts)")

# Random effect from mice on Burstiness with unit type as a fixed effect
fm5 = @formula(B ~ 1 + WW + (1|Mouse))
m5 = fit(LinearMixedModel, fm5, df)
# post hoc comaprison
# Degrees-of-freedom method: kenward-roger 
# P value adjustment: tukey method for comparing a family of 11 estimates (Regions)
m5_df = (m5, df)
@rput m5_df
R"summary(m4_df)"
# get the contrast result into a dataframe
emm5 = rcopy(R"summary(emmeans(m5_df, pairwise ~ WW, pbkrtest.limit = 20000)$contrasts)")

# Random effect from mice on Memory with unit type as a fixed effect
fm6 = @formula(M ~ 1 + WW + (1|Mouse))
m6 = fit(LinearMixedModel, fm6, df)
# post hoc comaprison
# Degrees-of-freedom method: kenward-roger 
# P value adjustment: tukey method for comparing a family of 11 estimates (Regions)
m6_df = (m6, df)
@rput m6_df
R"summary(m4_df)"
#R"emmeans(m2_df, pairwise ~ Region, pbkrtest.limit = 20000)"
# get the contrast result into a dataframe
emm6 = rcopy(R"summary(emmeans(m6_df, pairwise ~ WW, pbkrtest.limit = 20000)$contrasts)")

# Plot
d1 = density(df.WW[df.WW.=="2"],df.Rate[df.WW.=="2"], xlabel="log10[FR]", ylabel="probability", label="ww", color=:black, bandwidth=0.05)
density!(df.WW[df.WW.=="0"],df.Rate[df.WW.=="0"],xlabel="log10[FR]", ylabel="probability", label="nw", color="#A53F97", bandwidth=0.05)

d2 = density(df.WW[df.WW.=="2"],df.B[df.WW.=="2"], xlabel="B", ylabel="probability", label="ww", color=:black, bandwidth=0.05)
density!(df.WW[df.WW.=="0"],df.B[df.WW.=="0"],xlabel="B", ylabel="probability", label="nw", color="#A53F97", bandwidth=0.05)

d3 = density(df.WW[df.WW.=="2"],df.M[df.WW.=="2"], xlabel="M", ylabel="probability", label="ww", color=:black, bandwidth=0.05)
density!(df.WW[df.WW.=="0"],df.M[df.WW.=="0"],xlabel="M", ylabel="probability", label="nw", color="#A53F97", bandwidth=0.05)

plt = plot(d1,d2,d3,layout=grid(1, 3))
plot!(size=(1000,400))
display(plt)

# save figure
savefig(plt,FIGPATH*"figure.svg")