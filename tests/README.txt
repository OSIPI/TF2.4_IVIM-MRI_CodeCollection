This document describes the testing done in the OSIPI TF2.4 IVIM repository


-- Outline --
1. Testing philosophy
2. Testing structure
3. Testing results



-- Testing philosophy --
Testing is integral to the repository.
There are many different contributions from many different people and only through diligent testing can they all be ensured to work correctly together.
Automated testing happens on different platforms and versions.
There are 3 major types of tests we conduct.
1. Requirements
    - Runs on each commit
    - Must pass to merge
    - All algorithms have the same requirements
    - E.g. bounds honored, code runs reasonably, properly integrated
    - Would prevent a merge if not passing
    - Flexibile input/output
    - Categories for testing
        -- Contributions - Some aspects currently tested as unit_test
            --- Initial bounds are respected
                ---- Needs implemening
            --- Initial guess is respected
                ---- Needs implemening - may no be possible
            --- Runs reasonably
                ---- Needs implemening: reduced data size, broadened limits
            --- Contains information about algorithm
        -- Wrapper
            --- Initial guess is in bounds
            --- Reasonable default bounds - f: [0 1], D >= 0 & D < D*, D* >= 0
            --- Input size is respected - result is same size as input
            --- Dictionary is returned - worth explicit testing?
        -- Phantom - lower priority
            --- Data can be pulled
2. Expectations
    - Run on each merge
    - Considered warnings
    - Should not necessarily prevent a merge
    - Categories for testing
        -- Determine performance changes from reference run
            --- Currently implemented but could be made easier to interact with
            --- Could be made easier and faster
3. Characterization
    - Run on demand
    - Performance of the algorithms
    - The accuracy and precision of the results
    - The speed of the generated results
    - Human readable report of the wrapped algorithms
    - Categories for testing
        -- Simulations
            --- Voxels from tissue and characterize algorithms
            --- Visualize parameter maps
        -- True data
            --- Visualize parameter maps
            --- Correlations between algorithms - plot the results and differences

-- Testing structure --

* The testing is controlled in several places.
* The testing itself is done with pytest which parses files for "test_" and runs the appropriate tests.
* The pytest testing can be done on your own machine by running "python -m pytest".
** This is configured with pytests.ini and conftest.py
* Testing on github is controlled by the github actions which are in the workflows folder.
* Each workflow performs a series of tests, and is defined by the yml file.
* Each workflow can run at specified times and with specified outputs.
* Currently the major testing workflows are unit_test.yml and analysis.yml.
* The unit_test workflow is done frequently and is relativly fast and does some basic algorithm testing.
* The analysis workflow is done more infrequently with merges to the main branch and does more Expecation testing.


-- Testing results --
The test results are written to several files, most notably an xml file for machine parsing, and a web page for code coverage.
The "analysis" tests are written to a csv file as well.
