This document describes the testing done in the OSIPI TF2.4 IVIM repository


-- Outline --
1. Testing philosophy
2. Testing structure
3. Testing results



-- Testing philosophy --
Testing is integral to the repository.
There are many different contributions from many different people and only through diligent testing can they all be ensured to work correctly together.
There are 3 major types of tests we conduct.
1. Requirements
    - Which are required to pass
    - All algorithms have the same requirements
    - E.g. bounds honored, code runs reasonably
2. Expectations
    - Considered warnings
    - Should not necessarily prevent a merge
    - Should cover performance changes
    - E.g. performance changes
3. Performance
    - The accuracy of the results
    - The speed of the generated results

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
