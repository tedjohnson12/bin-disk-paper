"""
polar fraction as a function of $j$
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import simpson
import warnings

from polar_disk_freq.analytic import Zanazzi2018, MartinLubow2019

from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma
import helpers
import paths
import colors

plt.style.use('bmh')

OUTFILE = paths.figures / 'fpol_j.pdf'

N_E_ANALYTIC = 247
N_I_ANALYTIC = 247

JS = np.linspace(0,2,80)

def prob_analytic(j: float) -> float:
    """
    Compute the polar fraction assuming isotropic angular momentum distribution and random eccentricity.
    
    Uses Martin & Lubow 2019.
    """
    eccentricities = np.linspace(0.001,0.999,N_E_ANALYTIC)
    polar_fracs = [MartinLubow2019.frac_polar(j,e,N_I_ANALYTIC) for e in eccentricities]
    fp  = simpson(y=polar_fracs,x=eccentricities)
    return fp

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax: plt.Axes
    fps = []
    for j in tqdm(JS):
        fps.append(prob_analytic(j))
    ax.plot(JS, fps, c=colors.yellow)
    ax.axhline(0.37, ls='--', c=colors.slate)
    ax.set_xlabel('$j$')
    ax.set_ylabel('$f_{\\rm polar}$\n (upper\nlimit)',rotation='horizontal',labelpad=22)

    fig.savefig(OUTFILE)