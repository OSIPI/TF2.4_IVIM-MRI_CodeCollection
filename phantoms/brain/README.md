# Code for generating ivim phantoms of the brain

Two sets of ground truth data are used for simulation:
- Simulation in the diffusive regime (IVIM parameters D, f and D*): reference values from Ryghög et al. 2014 and b-values from Federau et al. 2012
- Simulation in the ballistic regime (IVIM parameters D, f and vd): reference values and sequence parameters from Ahlgren et al. 2016

The segmentation used by the simulations is the ICBM 2009a nonlinear symmetric 3T atlas (https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/), the same as in e.g. ASLDRO (https://asldro.readthedocs.io/en/stable/).