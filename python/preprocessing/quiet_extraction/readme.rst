##############

.. _detect_quiet:

Detect quiet episodes
##############

.. note::
    prerequisites:
        * `NWB --> LFP/info/units exports <nwb_to_lfp>`_
        * if task contained licking: `lick extraction <lickextraction>`_

#. detect down-states from spikes
    * if possible, for probes dominated by L2/3 or ORB (inherently bursty) it might be hard
    * call code: quiet_extraction/call_offdetection.py
    * run code: quiet_extraction/run_offperiod_detection.py
    * plot code: quiet_extraction/run_offperiod_plotting.py
    * inspect the plots, then...
    * set whether the spike-date offdetection is usable: quiet_extraction/set_unusable_manually_from_table.py

#. detect transients (sharp waves/delta waves?) in the LFP
    * call code: quiet_extraction/call_transient_from_table.py
    * run code: quiet_extraction/extract_transients.py
    * plot and check one by one: quiet_extraction/plot_transients_and_off.py

#. collect/merge quiet candidates from LFP transients and spike down-states
    * call code: quiet_extraction/call_transoffextract_from_table.py
    * run code: quiet_extraction/extract_from_off_and_trans.py
