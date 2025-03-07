Custom code used in Le Merre and Heining et al. (2024): "A Prefrontal Cortex Map based on Single Neuron Activity" (currently available at [bioRxiv](https://doi.org/10.1101/2024.11.06.622308)).

The custom code was written in **julia** and **python** and is contained in the respective subfolders. 
To run the code and for further details, consult the respective README files in the julia and python folder.

### Data used

- NWB files of the KI dataset  available at DANDI archive [dandiset 001260](https://dandiarchive.org/dandiset/001260) and [dandiset 000473](https://dandiarchive.org/dandiset/000473).
- International Brain Lab (IBL) dataset available [here](https://registry.opendata.aws/ibl-brain-wide-map).
- Intermediate results and preprocessing data available at [Zenodo](https://zenodo.org/records/14205018).



# Code for Figures


## Figure 1 
Describing the dataset, nw/ww contract
* [Fig1c__plot_raster_and_EMG_from_NWB.jl](julia/Fig1c__plot_raster_and_EMG_from_NWB.jl)
* [Fig1f__plot_sliders.jl](julia/Fig1f__plot_sliders.jl)
* [Fig1g__plot_Ww_Nw_example_waveforms.jl](julia/Fig1g__plot_Ww_Nw_example_waveforms.jl)
* [Fig1h__plot_Ww_Nw_histogram.jl](julia/Fig1h__plot_Ww_Nw_histogram.jl)
* [Fig1i__MixModel_Statistics.jl](julia/Fig1i__MixModel_Statistics.jl)
* [spont_raster_examples.py](python/examples_illustrations/spont_raster_examples.py) --> example raster plots
* [quiet_active_featurecontrast.py](python/dataset_general/quiet_active_featurecontrast.py) --> distributions nw/ww

## Figure 2

#### Self-organizing maps (SOMs):
* [calc_SOM.py](python/SOM/calc_SOM.py)
* [plot_SOM.py](python/SOM/plot_SOM.py)
* [plot_SOM_brain.py](python/SOM/plot_SOM_brain.py) --> regional hitmaps
* [plot_SOMhits_resolved.py](python/SOM/plot_SOMhits_resolved.py) --> subregional hitmaps

#### Clustering of SOM nodes --> unit categories:
* [cluster_SOM.py](python/SOM/clusterSOM/cluster_SOM.py)
* [identify_nclust_SOM.py](python/SOM/clusterSOM/identify_nclust_SOM.py) --> control plots to identify suitable number of clusters
* [clusterplot_SOM.py](python/SOM/clusterSOM/clusterplot_SOM.py) --> plot clusters
* [plot_clusterlabels.py](python/SOM/clusterSOM/plot_clusterlabels.py) --> plot scatter labels

#### Stability of categories:
* [plot_stability.py](python/stability/plot_stability.py) --> plots stability and coincidence coefficient

#### (Sub-)Regional category enrichment:
* [calc_statsdict.py](python/enrichments/calc_statsdict.py)
* [plot_enrichments.py](python/enrichments/plot_enrichments.py) --> also plots the graph display
* [compare_nwwwIBLCarlen_statsmode.py](python/enrichments/compare_nwwwIBLCarlen_statsmode.py) --> compares IBL and KI dataset in terms of regional enrichment
* [compare_nw_ww.py](python/enrichments/compare_nw_ww.py) --> correlates nw/ww category enrichment across (sub-)regions

## Figure 3
* [harris_vs_enrichment_ENHequalaxes.py](python/connectivity_correlations/harris_vs_enrichment_ENHequalaxes.py) --> category enrichment vs Harris' hierarchy score
* [harriscorrelation_compare.py](python/connectivity_correlations/harriscorrelation_compare.py) --> bar graph comparing correlation strengths

## Figure 4

#### category enrichments of flatmap ROIs
* [flatmap_tessellation.py](python/flatmaps/flatmap_tessellation.py) --> partitions the PFC into flatmap ROIs 
* [calc_flatmap_statsdict.py](python/flatmaps/calc_flatmap_statsdict.py) --> calculate enrichment of flatmap ROIs; apply *roi_tag = 'dataRois'*
* [plot_flatmap_enrichments.py](python/flatmaps/plot_flatmap_enrichments.py) --> plot enrichment matrix and modules; apply *roi_tag = 'dataRois'*

#### correlation of category enrichment with Gao's hierarchy
* [calc_flatmap_statsdict.py](python/flatmaps/calc_flatmap_statsdict.py) --> calculate enrichment of flatmap ROIs; apply *roi_tag = 'gaoRois'*
* [gao_vs_enrichment.py](python/connectivity_correlations/gao_vs_enrichment.py) --> category enrichment vs Gao's hierarchy score


## Figure 5
####