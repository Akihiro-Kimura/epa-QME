"""Model definitions for the manuscript-reproduction notebooks.

The numerical solver in QME.py is N-site generic and does not import this file.
"""
import numpy as np
from QME import chain_hamiltonian


def three_site_model(V12, V23, delta_E12, delta_E23):
    """Three-site Hamiltonian in Eq. (32), using E3=0 as energy reference."""
    E3 = 0.0
    E2 = delta_E23
    E1 = delta_E12 + delta_E23
    return chain_hamiltonian([E1, E2, E3], [V12, V23])


def fmo_7site_hamiltonian():
    """Seven-site FMO Hamiltonian used for manuscript Fig. 5 (cm^-1)."""
    return np.array([
        [12410.0, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
        [  -87.7,12530.0, 30.8,   8.2,   0.7,  11.8,   4.3],
        [    5.5,  30.8,12210.0, -53.5,  -2.2,  -9.6,   6.0],
        [   -5.9,   8.2, -53.5,12320.0, -70.7, -17.0, -63.3],
        [    6.7,   0.7,  -2.2, -70.7,12480.0,  81.1,  -1.3],
        [  -13.7,  11.8,  -9.6, -17.0,  81.1,12630.0,  39.7],
        [   -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7,12440.0],
    ], dtype=float)
