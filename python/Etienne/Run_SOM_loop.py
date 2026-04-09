import sys, os
#os.chdir('python')
PATHS = 'PATHS/filepaths_Pete.yml'
nclust = 5
method = 'ward'
for somsettings in ['runPete1','runPete2','runPete3']:
    #os.system(f"python SOM/calc_SOM.py {PATHS} {somsettings}")
    #os.system(f"python SOM/clusterSOM/clusterplot_SOM.py {PATHS} {somsettings} {nclust} {method}")
    #os.system(f"python enrichments/calc_statsdict_Pete.py {PATHS} {somsettings} {nclust} {method}")
    #os.system(f"python enrichments/statsdict_Pete.py {PATHS} {somsettings} {nclust} {method}")
    #os.system(f"python enrichments/plot_enrichments_Pete.py {PATHS} {somsettings} {nclust} {method}")
    os.system(f"python flatmaps/statsdict_Pete.py {PATHS} {somsettings} {nclust} {method} 0 regions")
    os.system(f"python flatmaps/Pete_plot.py {PATHS} {somsettings} {nclust} {method} regions")

