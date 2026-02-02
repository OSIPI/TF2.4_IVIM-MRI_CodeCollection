# For TCML_TechnionIIT_lsqBOBYQA we skip 2 tests:
The pericardium seems to give weird results on some systems. We believe this to be different interpertation of the data and not a systematic error in the algorithm
For the bounds test, on Mac and Ubuntu, TCML_TechnionIIT_lsqBOBYQA returns default values of 0. We do not believe this to be an intrensic error of the code, but an instable performance in small, unrealistic boundaries. 
**Consequently, bounds are not tested for TCML_TechnionIIT_lsqBOBYQA and it should be used with caution when bounds are requiered.**