

pfcmap python code
====================

Custom python code used in Le Merre and Heining et al. (2024): "A Prefrontal Cortex Map based on Single Neuron Activity" (currently at bioRxiv https://doi.org/10.1101/2024.11.06.622308).


All code was run on a linux (Ubuntu 22.04) using Python version 3.8

Requirements are given in requirements.txt


Setting file paths
####################
To run the code, replace the following paths according to your settings
**DANDIPATH** --> where the nwbfiles of the KI dataset are stored. https://dandiarchive.org/dandiset/001260?pos=1
**ZENODOPATH** --> where the intermediated results are stored. todo: LINK
**CODEPATH** --> directory in which "pfcmap" was downloaded locally (e.g. /home/username/workspace)
**LOGPATH** --> directory in which file loggers should be saved (e.g. /home/username/logfolder)
**FIGDIR** --> directory where figures generated should be saved (e.g. /home/username/pfcmap/figures)
**IBLPATH** --> directory where you stored the IBL dataset and nwbs. todo: LINK
