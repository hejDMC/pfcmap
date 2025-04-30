Custom code used in Le Merre and Heining et al. (2024): "A Prefrontal Cortex Map based on Single Neuron Activity" (currently available at [bioRxiv](https://doi.org/10.1101/2024.11.06.622308)).

The custom code was written in **julia** and **python** and is contained in the respective subfolders. 
To run the code and for further details, consult the respective README files in the julia and python folder.

### Data used

- NWB files of the KI dataset  available at DANDI archive [dandiset 001260](https://dandiarchive.org/dandiset/001260) and [dandiset 000473](https://dandiarchive.org/dandiset/000473).
- International Brain Lab (IBL) dataset available [here](https://registry.opendata.aws/ibl-brain-wide-map).
- Intermediate results and preprocessing data available at [Zenodo](https://zenodo.org/records/14205018).



# Code for Figures

## Figure 1: Dataset
Describing the dataset, contrasting nw and ww units.
* [Fig1c__plot_raster_and_EMG_from_NWB.jl](julia/Fig1c__plot_raster_and_EMG_from_NWB.jl)
* [Fig1f__plot_sliders.jl](julia/Fig1f__plot_sliders.jl)
* [Fig1g__plot_Ww_Nw_example_waveforms.jl](julia/Fig1g__plot_Ww_Nw_example_waveforms.jl)
* [Fig1h__plot_Ww_Nw_histogram.jl](julia/Fig1h__plot_Ww_Nw_histogram.jl)
* [Fig1i__MixModel_Statistics.jl](julia/Fig1i__MixModel_Statistics.jl)
* [spont_raster_examples.py](python/examples_illustrations/spont_raster_examples.py) --> example raster plots
* [quiet_active_featurecontrast.py](python/dataset_general/quiet_active_featurecontrast.py) --> distributions nw/ww

## Figure 2: Spontaneous activity patterns - enrichment in brain (sub)regions
Charaterizing single unit's spontaneous firing properties and the (sub-)regional enrichment of spontaneous firing classes 
using self-organizing maps (SOMs) and hierarchical clustering.

#### Self-organizing maps (SOMs):
* [calc_SOM.py](python/SOM/calc_SOM.py) --> calculate SOMs
* [plot_SOM.py](python/SOM/plot_SOM.py) --> plot SOM characteristics, like e.g. component planes
* [plot_SOM_brain.py](python/SOM/plot_SOM_brain.py) --> plot regional hit-maps
* [plot_SOMhits_resolved.py](python/SOM/plot_SOMhits_resolved.py) --> plot subregional hit-maps

#### Clustering of SOM nodes to obtain unit categories:
* [cluster_SOM.py](python/SOM/clusterSOM/cluster_SOM.py) --> partition the SOM's prototype vectors (hexagons) into clusters
* [identify_nclust_SOM.py](python/SOM/clusterSOM/identify_nclust_SOM.py) --> control plots to identify suitable number of clusters
* [clusterplot_SOM.py](python/SOM/clusterSOM/clusterplot_SOM.py) --> plot clusters
* [plot_clusterlabels.py](python/SOM/clusterSOM/plot_clusterlabels.py) --> plot scatter labels

#### Stability of categories:
* [plot_stability.py](python/stability/plot_stability.py) --> plot stability and coincidence coefficient

#### (Sub-)Regional category enrichment:
* [calc_statsdict.py](python/enrichments/calc_statsdict.py) --> calculate enrichments
* [plot_enrichments.py](python/enrichments/plot_enrichments.py) --> plot enrichments (also graph display)
* [compare_nwwwIBLCarlen_statsmode.py](python/enrichments/compare_nwwwIBLCarlen_statsmode.py) --> compare IBL and KI dataset in terms of regional enrichment
* [compare_nw_ww.py](python/enrichments/compare_nw_ww.py) --> correlate nw/ww category enrichment across (sub-)regions

## Figure 3: Spontaneous activity in brain (sub)regions vs. cortical hierarchy
Correlation between spontaneous category enrichment and cortical hierarchy score.
* [harris_vs_enrichment_ENHequalaxes.py](python/connectivity_correlations/harris_vs_enrichment_ENHequalaxes.py) --> category enrichment vs Harris' hierarchy score
* [harriscorrelation_compare.py](python/connectivity_correlations/harriscorrelation_compare.py) --> bar graph comparing correlation strengths

## Figure 4: PFC map - spontaneous activity
Activity defined map of the PFC and correlation between spontaneous category enrichment and intra-PFC hierarchy.

#### Category enrichments of flatmap ROIs:
* [flatmap_tessellation.py](python/flatmaps/flatmap_tessellation.py) --> partition the PFC into flatmap ROIs 
* [calc_flatmap_statsdict.py](python/flatmaps/calc_flatmap_statsdict.py) --> calculat enrichment of flatmap ROIs; apply *roi_tag = 'dataRois'*
* [plot_flatmap_enrichments.py](python/flatmaps/plot_flatmap_enrichments.py) --> plot enrichment flatmaps, matrix and modules; apply *roi_tag = 'dataRois'*
#### Correlation of category enrichment with Gao's (intra-PFC) hierarchy:
* [calc_flatmap_statsdict.py](python/flatmaps/calc_flatmap_statsdict.py) --> calculate enrichment of flatmap ROIs; apply *roi_tag = 'gaoRois'*
* [gao_vs_enrichment.py](python/connectivity_correlations/gao_vs_enrichment.py) --> spontaneous category enrichment vs Gao's hierarchy score


## Figure 5: PFC map - evoked activity & comparison
Mapping tone responses and tuning to goal directed behavior in the PFC;
Correlation of tone responsiveness and task tuning with Gao's intra-PFC hierarchy; 
Comparisons of spontaneous and evoked activity in space (cosine similarity) and at single-unit level (mutual enrichment).  


#### Tone responses and overall tone responsiveness:
* [response_examples.py](python/examples_illustrations/response_examples.py) --> plot example response raster plot and PSTH 
* [plot_PSTH_and_zeta.py](python/SOM/clusterSOM/plot_PSTH_and_zeta.py) --> plot median PSTH per category
* [calc_flatmap_statsdict.py](python/flatmaps/calc_flatmap_statsdict.py) --> calculate enrichment matrix for ROIs vs response categories
* [plot_flatmap_enrichments.py](python/flatmaps/plot_flatmap_enrichments.py) --> plot tone response enrichment flatmaps and enrichment modules
* [plot_clusterlabels_response.py](python/SOM/clusterSOM/plot_clusterlabels_response.py) --> median PSTH labels to the enrichment matrix
* [plot_flat_zeta_responder_and_hierarchy.py](python/flatmaps/plot_flat_zeta_responder_and_hierarchy.py) --> plot overall responsiveness flatmap and correlate ROI enrichment to intra-PFC hierarchy

#### Tuning to behavioral variables (IBL task):
* [extract_tuning.py](python/goalDirectedBehavior_IBL/extract_tuning.py) --> ccCP values per unit and task component (stim, choice, feedback)
* [plot_tuning_examples.py](python/goalDirectedBehavior_IBL/plot_tuning_examples.py) --> example raster plots
* [flatmap_tessellationIBL.py](python/goalDirectedBehavior_IBL/flatmap_tessellationIBL.py) --> tessellate PFC into ROIs of similar unit count for IBL-task
* [calc_statsdict_flatmap_tuning.py](python/goalDirectedBehavior_IBL/calc_statsdict_flatmap_tuning.py) --> compute enrichment for IBL data ROIs in units tuned to task components
* [plot_tuning_flatmaps.py](python/goalDirectedBehavior_IBL/plot_tuning_flatmaps.py) --> plot task tuning enrichments on flatmaps
* [calc_statsdict_gaoRois_tuning.py](python/goalDirectedBehavior_IBL/calc_statsdict_gaoRois_tuning.py) --> compute enrichment for gaoROIs in units tuned to task components
* [gao_vs_enrichment_tuning.py](python/goalDirectedBehavior_IBL/gao_vs_enrichment_tuning.py) --> correlate enrichment in tuned units with Gao's intra-PFC hierarchy

#### Comparison of spontaneous and evoked activity (spatial similarity and mutual enrichment):
* [flatmap_rasterization.py](python/flatmaps/flatmap_rasterization.py) --> rasterize flatmaps as a prerequiste for spatial similarity estimation
* [compare_flatmaps_spont_resp_task.py](python/flatmaps/compare_flatmaps_spont_resp_task.py) --> spatial (i.e. cosine) similarities between spontaneous maps and task-tuning maps, and tone responsiveness map
* [response_vs_spont.py](python/enrichments/response_vs_spont.py)--> calculate and plot mutual enrichment of tone response categories and spontaneous categories
* [calc_statsdict_tuning.py](python/goalDirectedBehavior_IBL/calc_statsdict_tuning.py) --> calculate mutual enrichment of task-tuning and spontaneous categories
* [plot_tuning_vs_spont.py](python/goalDirectedBehavior_IBL/plot_tuning_vs_spont.py) --> plot mutual enrichment of task-tuning and spontaneous categories

