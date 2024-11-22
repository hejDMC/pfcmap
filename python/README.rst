

pfcmap python code
====================

Custom python code used in Le Merre and Heining et al. (2024): "A Prefrontal Cortex Map based on Single Neuron Activity" (currently at bioRxiv https://doi.org/10.1101/2024.11.06.622308).


All code was run on Linux (Ubuntu 22.04) using Python version 3.8

Requirements are given in requirements.txt.


To run the code, download the repo, navigate to the code folder (path_to_your_directory/pfcmap) and call scripts from there.

Setting file paths
####################
To run the code, replace the following paths according to your settings:

- **DANDIPATH** --> where nwbfiles of the KI dataset are stored. https://dandiarchive.org/dandiset/001260
- **ZENODOPATH** --> where intermediate results are stored. https://zenodo.org/records/14205018
- **CODEPATH** --> directory to which you downloaded the "pfcmap" repository (e.g. /home/username/workspace)
- **LOGPATH** --> directory in which file loggers are saved (e.g. /home/username/logfolder)
- **FIGDIR** --> directory where figures generated are saved (e.g. /home/username/pfcmap/figures)
- **IBLPATH** --> directory where you stored the nwb files extracted from the IBL dataset (note: nwbs not provided directly) https://registry.opendata.aws/ibl-brain-wide-map
