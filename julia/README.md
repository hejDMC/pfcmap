# pfcmap/julia


Custom julia code used in Le Merre and Heining et al. (2024): *"A Prefrontal Cortex Map based on Single Neuron Activity"* (currently at [bioRxiv](https://doi.org/10.1101/2024.11.06.622308)).

All code was run on a MacOS (Ventura 13.7.1) using Julia version 1.8 and higher.

To run the code navigate to the code folder and call all scripts from there.


## Setting file paths

To run the code, replace the following paths according to your settings:

DANDIPATH --> where the NWB files of the KI dataset are stored. Download available at DANDI archive [dandiset 001260](https://dandiarchive.org/dandiset/001260) and [dandiset 000473](https://dandiarchive.org/dandiset/000473).

ZENODOPATH --> where the intermediate results are stored. Download available at [Zenodo] (https://zenodo.org/records/14205018).

FIGPATH --> directory where figures generated should be saved (e.g. /home/username/pfcmap/figures)

IBLPATH --> directory where you stored the IBL dataset and nwbs. todo: LINK


## Setting RCall

The code uses RCall to run the Linear Mixed Model statistics (lme4, emmeans, pbkrtest, and MuMIn R packages). Instructions about how to set up RCall.jl can be found [here](https://github.com/JuliaInterop/RCall.jl).



## Allen CCF Julia tools

The code also calls tools to work with the Allen Brain Institute CCFv3 data. These tools are currently under development [here](https://github.com/PierreLeMerre/allenCCF_julia).