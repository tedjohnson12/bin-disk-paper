"""
Compare the phase diagram for the massless case to that of the massive case.
"""


import matplotlib.pyplot as plt
import numpy as np

import paths
import colors

from misaligned_cb_disk import system, mc, search
from misaligned_cb_disk.utils import STATE_LONG_NAMES

color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}

OUTFILE = paths.figures / 'compare_massive.pdf'
DB_PATH = paths.data / 'mc.db'

MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
MASS_PLANET = 1e-2
SEP_PLANET = 1
NU = 0
ECC_PLANET = 0
ARG_PARIAPSIS = 0
GR = False
ECC_BIN = 0.5

N_ANALYTIC = 1000
SEARCH_PRECISION = np.pi / 180 * 0.5 * 5 # 5 degree
CONFIDENCE_INTERVAL_WIDTH = 0.1
CONFIDENCE_LEVEL = 0.95

def get_sampler(ecc: float, _mass: float)->mc.Sampler:
    """
    Get a sampler for a given eccentricity and mass.
    
    Parameters
    ----------
    ecc : float
        The eccentricity of the binary.
    _mass : float
        The mass of the planet.
    
    Returns
    -------
    sampler : mc.Sampler
        The monte carlo sampler.
    """
    return mc.Sampler(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ecc,
        mass_planet=_mass,
        semimajor_axis_planet=SEP_PLANET,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        gr=GR,
        rng=None,
        db_path=DB_PATH
    )

def do_mc(ecc: float, _mass: float):
    """
    Run a monte carlo simulation for a given eccentricity.

    Parameters
    ----------
    ecc : float
        The eccentricity of the binary.
    mass : float
        The mass of the planet.

    Returns
    -------
    low : float
        The lower end of the confidence interval.
    high : float
        The upper end of the confidence interval.
    """
    _sampler = get_sampler(ecc, _mass)
    _sampler.sim_until_precision(CONFIDENCE_INTERVAL_WIDTH, batch_size=20)
    result = _sampler.bootstrap(
        system.LIBRATING, confidence_level=CONFIDENCE_LEVEL)
    return result.confidence_interval.low, result.confidence_interval.high

def do_search(ecc:float, _mass: float):
    """
    Do a binary search for a given eccentricity.
    
    Parameters
    ----------
    ecc : float
        The eccentricity of the binary.
    
    Returns
    -------
    transitions : list
        The list of transitions.
    """
    _searcher = search.Searcher(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ecc,
        mass_planet=_mass,
        semimajor_axis_planet=SEP_PLANET,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        lon_ascending_node_planet=np.pi/2,
        precision=SEARCH_PRECISION,
        gr=GR,
        db_path=DB_PATH
    )
    _searcher._integration_max_orbits = 10000
    _transitions = _searcher.search()
    return _transitions

def plot_full(_ax: plt.Axes, ecc: float, inclination:float,_mass: float):
    """
    Plot the full path for a particular setup.
    
    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on.
    ecc : float
        The eccentricity of the binary.
    inclination : float
        The initial inclination of the planet.
    _mass : float
        The mass of the planet.
    """
    sys = system.System.from_params(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ecc,
        mass_planet=_mass,
        semimajor_axis_planet=SEP_PLANET,
        inclination_planet=inclination,
        lon_ascending_node_planet=np.pi/2,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        gr=GR,
        sim=None
    )
    sys.integrate_to_get_path(step=5,max_orbits=10000,capture_freq=1)
    incls = sys.inclination
    lon_asc_node = sys.lon_ascending_node
    x = incls*np.cos(lon_asc_node)
    y = incls*np.sin(lon_asc_node)
    state = sys.state
    ls = '-' if _mass == 0 else '--'
    _ax.plot(x,y, c=color[state], lw=1,ls=ls)

def plot_scatter(_ax: plt.Axes,_sampler: mc.Sampler):
    """
    Make a scatter plot from the monte carlo sampler.
    
    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on.
    sampler : mc.Sampler
        The monte carlo sampler.
    """
    lon_asc_node = np.array(_sampler.lon_ascending_nodes)
    incl = np.array(_sampler.inclinations)
    states = np.array(_sampler.states)
    x = incl*np.cos(lon_asc_node)
    y = incl*np.sin(lon_asc_node)
    for state in [system.LIBRATING, system.RETROGRADE, system.PROGRADE, system.UNKNOWN]:
        _ax.scatter(x[states==state], y[states==state], c=color[state], s=10,label=STATE_LONG_NAMES[state],marker='x')
    _ax.set_aspect('equal')
    _ax.set_xlabel('$i \\cos{\\Omega}$')
    _ax.set_ylabel('$i \\sin{\\Omega}$')
    _ax.legend()
    _ax.set_xlim(-np.pi, np.pi)
    _ax.set_ylim(-np.pi, np.pi)

if __name__ in '__main__':
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax: plt.Axes
    sampler = get_sampler(ECC_BIN, MASS_PLANET)
    plot_scatter(ax,sampler)
    transitions = do_search(ECC_BIN,_mass=0)
    for transition in transitions:
        for i in [transition.low_value, transition.high_value]:
            plot_full(ax,ECC_BIN,i,0)
            if system.LIBRATING in [transition.low_state, transition.high_state]:
                plot_full(ax,ECC_BIN,-i,0)
    transitions = do_search(ECC_BIN,_mass=MASS_PLANET)
    for transition in transitions:
        for i in [transition.low_value, transition.high_value]:
            plot_full(ax,ECC_BIN,i,MASS_PLANET)
            if system.LIBRATING in [transition.low_state, transition.high_state]:
                plot_full(ax,ECC_BIN,-i,MASS_PLANET)
    fig.savefig(OUTFILE)

