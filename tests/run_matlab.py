import matlab.engine

def run(x):
    eng = matlab.engine.start_matlab()
    a = eng.sqrt(x)
    eng.quit()
    return(a)