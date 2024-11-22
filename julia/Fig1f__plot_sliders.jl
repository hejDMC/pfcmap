using HDF5
using Plots

# MAIN PATH
ZENODOPATH = "Your/Path/To/Zenodo/Data" 

# FIGURE PATH
FIGPATH = "Your/path/to/figures"

# Units and Mice selection for figure
unit_nbs = [257 173 296 278]
mice = ["PL027_20190511-probe0" "216301_20200521-probe0" "PL031_20190524-probe0" "PL054_20191122-probe0"]

r = fill(NaN,4)
b = fill(NaN,4)
m = fill(NaN,4)

for u in 1:length(unit_nbs)

    src = ZENODOPATH*"metrics_"*mice[u]*".h5"
    h5 = h5open(src,"r")

    rate = read(h5["interval_metrics/rate/prestim/active/all/dur3/mean"])
    B = read(h5["interval_metrics/B/prestim/active/all/dur3/mean"])
    M = read(h5["interval_metrics/M/prestim/active/all/dur3/mean"])

    r[u] = rate[unit_nbs[u] + 1]
    b[u] = B[unit_nbs[u] + 1]
    m[u] = M[unit_nbs[u] + 1]

end

# Plot
s1 = scatter([1 2 3 4]', r, color=:black, ylims=(-0.2,1.3),ylabel="log10(rate)",legend=false)
s2 = scatter([1 2 3 4]', b, color=:black, ylims=(-0.35,0.35),ylabel="B",legend=false)
s3 = scatter([1 2 3 4]', m, color=:black, ylims=(-0.35,0.35),ylabel="M",legend=false)
plt = plot(s1,s2,s3, layout=grid(1,3),size=(300,400))
display(plt)

# save figure
savefig(plt,FIGPATH*"figure.svg")