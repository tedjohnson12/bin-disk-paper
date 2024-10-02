"""
polar fraction as a function of $j$
"""

from typing import Callable
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

N_E_ANALYTIC = 101
N_I_ANALYTIC = 101

JS = np.linspace(0,2,80)
ECCENTRICITIES = np.linspace(0.001,0.999,N_E_ANALYTIC)

def prob_e(alpha: float) -> Callable[[float],float]:
    def func(e: float) -> float:
        return (alpha+1) * e**alpha
    return func

def rev_prob_e(alpha: float) -> Callable[[float],float]:
    def func(e: float) -> float:
        return (alpha+1) * (1-e)**alpha
    return func

def exp_prob_e(sigma: float) -> Callable[[float],float]:
    def func(e: float) -> float:
        return -1/sigma/(np.exp(-1/sigma) - 1) * np.exp(-e/sigma)
    return func

def flat_cutoff_prob_e(x:float) -> Callable[[float],float]:
    def func(e: np.ndarray) -> np.ndarray:
        return np.where(e < x, 1/x, 0)
    return func

def prob_analytic(j: float,
                  prob_e = prob_e(0)
                  ) -> float:
    """
    Compute the polar fraction assuming isotropic angular momentum distribution and random eccentricity.
    
    Uses Martin & Lubow 2019.
    """
    
    polar_fracs = [MartinLubow2019.frac_polar(j,e,N_I_ANALYTIC) for e in ECCENTRICITIES]
    weights = prob_e(ECCENTRICITIES)
    fp  = simpson(y=polar_fracs*weights,x=ECCENTRICITIES)
    return fp

def single_e_analytic(j:float,e: float) -> float:
    return MartinLubow2019.frac_polar(j,e,N_I_ANALYTIC)

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax: plt.Axes
    fig.subplots_adjust(right=0.7)
    inax = ax.inset_axes([0.6, 0.15, 0.2, 0.2])
    
    def plot(func,color,label):
        fps = []
        for j in tqdm(JS):
            fps.append(prob_analytic(j,prob_e=func))
        ax.plot(JS, fps, c=color, label=label)
        inax.plot(ECCENTRICITIES, func(ECCENTRICITIES), c=color)
    def plot_single(e,color,label):
        fps = []
        for j in tqdm(JS):
            fps.append(single_e_analytic(j,e))
        ax.plot(JS, fps, c=color, label=label,ls='--')
        
    
    plot(prob_e(1),colors.dark_orange,'Thermal')
    
    plot(prob_e(0),colors.yellow,'Flat')
    
    # plot(rev_prob_e(1),colors.teal,'Inverse Thermal')
    
    plot(flat_cutoff_prob_e(0.1),colors.teal,'$e_\\text{b} < 0.1$')
    
    for e in [0.01,0.3,0.5,0.7,0.99]:
        color = plt.cm.viridis(e)
        plot_single(e,color,f'$e_\\text{{b}} = {e:.2f}$')
    
    
    ax.axhline(0.37, ls='--', c=colors.slate, label='Kozai-Lidov')
    ax.set_xlabel('$j$')
    ax.set_ylabel('$f_{\\rm polar}$',rotation='horizontal',labelpad=22)
    ax.legend(loc=(1.05,0.15))
    
    inax.set_xlabel('$e_\\text{b}$')
    inax.set_ylabel('$p_\\text{e}(e_\\text{b})$')

    fig.savefig(OUTFILE)