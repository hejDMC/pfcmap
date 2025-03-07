Order of script calling for goal directed behavior tuning (IBL task) analysis
==================


#extracting tuning variables
1.1 python/goalDirectedBehavior_IBL/extract_tuning.py
1.2 python/goalDirectedBehavior_IBL/extract_taskresp.py

#co-enrichment matrices
2.1 python/goalDirectedBehavior_IBL/calc_statsdict_tuning.py
2.2 python/goalDirectedBehavior_IBL/plot_tuning_vs_spont.py #plots co-enrichment matrix of spontanous categories vs tuning

#flatmaps of IBL tuning variables
3.1 python/goalDirectedBehavior_IBL/flatmap_tessellationIBL.py #tessellate PFC according to available IBL task data --> use IBLflatmap_PFC_ntesselated_obeyRegions_res200.h5 (deep, ww)
3.2 python/goalDirectedBehavior_IBL/calc_statsdict_flatmap_tuning.py
3.3 python/goalDirectedBehavior_IBL/plot_tuning_flatmaps.py #plots flatmap rois

#correlation with Gao hierarchy
4.1 python/goalDirectedBehavior_IBL/calc_statsdict_gaoRois_tuning.py #calculates enrichment matrix for tuning vars in gaoROIs
4.2 python/goalDirectedBehavior_IBL/gao_vs_enrichment_tuning.py #plots correlation

#plot examples
5. python/goalDirectedBehavior_IBL/plot_tuning_examples.py # re-plots example raster plots plotted in 1.1 using desired figure format
