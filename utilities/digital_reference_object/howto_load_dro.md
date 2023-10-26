The digital reference object (DRO.npy) is a numpy array containing the following fields:
- `DRO['D']`: value of diffusion coefficient used for generating signal
- `DRO['f']`: value of perfusion fraction used for generating signal
- `DRO['Dp']`: value of pseudo-diffusion coefficient used for generating signal
- `DRO['S0']`: value of signal at b=0 used for generating signal
- `DRO['bvals']`: numpy array of b-values used for generating signal
- `DRO['signals']`: numpy array of signals generated using the above parameters

The DRO can be loaded using the following code:
data = np.load('DRO.npy', allow_pickle=True)
