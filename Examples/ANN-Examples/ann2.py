# Generate ANDNOT function using McCulloch-Pitts neural net by a python program.
import numpy as np

def mc_culloch_pitts_andnot(x1,x2):
    w1,w2=1,-1
    threshold=0.5
    weighted_sum = w1 * x1 + w2 * x2
    if weighted_sum > threshold:
        return 1
    else:
        return 0

print("ANDNOT(0, 0) = ", mc_culloch_pitts_andnot(0, 0))
print("ANDNOT(0, 1) = ", mc_culloch_pitts_andnot(0, 1))
print("ANDNOT(1, 0) = ", mc_culloch_pitts_andnot(1, 0))
print("ANDNOT(1, 1) = ", mc_culloch_pitts_andnot(1, 1))