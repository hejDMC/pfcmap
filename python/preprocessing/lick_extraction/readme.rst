
==================================
extracting licktimes
==================================


ONLY use when experiment contained licking
depends on whether a usable piezo-trace is present or not

**If usable piezo trace is present:**

* call code: preprocessing/lick_extraction/call_lickdet_piezo.py
* run code: preprocessing/lick_extraction/lickarts_from_piezo.py


**Elif licktraces show nicely in the LFP:**

* call code: preprocessing/lick_extraction/call_lickart_detection.py
* run scripts: preprocessing/lick_extraction/run_licktrace_icadet.py
* then: SEMI-AUTOMATIC lick detection as command-line script: preprocessing/lick_extraction/run_lickextraction_fromtrace.py


AFTER lick extraction:

#. update info-files
#. redo example plotting --> as a quality control
