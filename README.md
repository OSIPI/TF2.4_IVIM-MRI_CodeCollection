# TF2.4_IVIM_code_collection

The ISMRM Open Science Initiative for Perfusion Imaging (OSIPI) is an initiative of the ISMRM Perfusion Study Group, founded in 2019 after a membership survey, and with a mission to: 

>“promote the sharing of perfusion imaging software in order to eliminate the practice of duplicate development, improve the reproducibility of perfusion imaging research, and speed up the translation into tools for discovery science, drug development, and clinical practice”

This **IVIM code collection** code library is maintained by OSIPI [Taskforce 2.4](https://osipi.ismrm.org/task-forces/task-force-2-4/) and aims to collect, test and share open-source code related to intravoxel incoherent motion (IVIM) analysis of diffusion encoded MRI data to be used in research and software development. Code contributions can include any code related IVIM analysis (denoising, motion correction, model fitting, etc.), but at an initial phase, development of tests and other features of the repository will predominantly focus on fitting algorithms. A goal of the IVIM OSIPI task force is to develop a fully tested and harmonized code library, building upon the contributions obtained through this initiative. Documentation and analysis are available on the [OSIPI TF2.4](https://osipi.github.io/TF2.4_IVIM-MRI_CodeCollection/).

We have some useful tools and further documentation on https://osipi.github.io/TF2.4_IVIM-MRI_CodeCollection/.

## How to contribute

If you would like to get involve in OSIPI and work within the task force, please email the contacts listed on our website.

If you would like to contribute with code, please follow the instructions below:

*   [Setting up Git](doc/setting_up_git.md)
*   [How to create a copy of the respository and contribute changes to the repository](doc/create_local_copy_of_repository.md)
*   [Guidelines for IVIM code contribution](doc/guidelines_for_contributions.md)
*   [Guidelines to creating a test file](doc/creating_test.md) 

If you would like to use code from the repository and/or are new to Github or IVIM, please see the jupyter notebook below:
*   [Introduction to TF2.4_IVIM-MRI_CodeCollection github and IVIM Analysis using Python](doc/Introduction_to_TF24_IVIM-MRI_CodeCollection_github_and_IVIM_Analysis_using_Python.ipynb)

If you would like to use MATLAB-based algorithms or apply the testing framework to one of these algorithms, please have a look at these instructions:
* [MATLAB-related instructions](doc/matlab_instructions.md)

## Repository Organization

The repository is organized in four main folders along with configuration files for automated testing. 

The **doc** folder contains all documentation related to the repository of task force 2.4.

The **src** folder contains source code contributed by the the community. Within **src**, the **original** folder contains the code to be tested, and the **wrappers** folder contains code for harmizing the calls the different code contributions. Within the **original** folders, contributions are stored in Initials_Institution, e.g. src/original/OGC_AmsterdamUMC.

The **test** folder contains the test files corresponding to the contributed code in **src**. *to be structured*

The **utils** folder contains various helpful tools.


## View Testing Reports
[![Unit tests](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/unit_test.yml/badge.svg?branch=main)](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/unit_test.yml)
[![Algorithm Analysis](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/analysis.yml/badge.svg?branch=main)](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/analysis.yml)
[![Build & Deploy Website](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/website.yml/badge.svg?branch=main)](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection/actions/workflows/website.yml)
