## MATLAB-related instructions

A large portion of IVIM-related code has historically been written in the MATLAB programming language. For this reason, we have included the possibility to call and test MATLAB-based code from the Python-based code developed here. This requires a few additional steps to get going

- In addition to the Python packages listed requirements.txt, one also need to install matlab.engine, for example with pip: `python -m pip install matlabengine`. This requires a corresponding version of MATLAB to be installed.
- To be able to call the MATLAB code in this repository, the src/original folder must be added to the MATLAB path
- Testing of MATLAB-based code only runs offline (not on Github). This is done by running `python -m pytest --withmatlab`