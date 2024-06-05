"""
Compare the phase diagram for the massless case to that of the massive case.
"""


import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import paths
import helpers
import colors

from polar_disk_freq import system, mc, search
from polar_disk_freq.utils import STATE_LONG_NAMES
from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma

color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}

plt.style.use('bmh')

OUTFILE = paths.figures / 'mc_example.pdf'
DB_PATH = paths.data / 'mc.db'

MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
ECC_BIN = 0.4
SEP_PLANET = 5*SEP_BINARY
JCRIT = helpers.j_critical(ECC_BIN)
J = 0
ECC_PLANET = 0

NU = 0
ARG_PARIAPSIS = 0
GR = False


N_ANALYTIC = 1000
SEARCH_PRECISION = np.pi / 180 * 0.5 * 5 # 5 degree
CONFIDENCE_INTERVAL_WIDTH = 0.1
CONFIDENCE_LEVEL = 0.95

def get_sampler(ecc: float, _j: float)->mc.Sampler:
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
    mass = helpers.get_mass_planet(
        _j=_j,
        m_bin=MASS_BINARY,
        f_bin=FRAC_BINARY,
        e_bin=ECC_BIN,
        a_bin=SEP_BINARY,
        a_planet=SEP_PLANET,
        e_planet=ECC_PLANET
    )
    return mc.Sampler(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ecc,
        mass_planet=mass,
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
        _ax.scatter(x[states==state], y[states==state], c=color[state], s=50,label=STATE_LONG_NAMES[state],marker='.')

if __name__ in '__main__':
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    # fig.subplots_adjust(left=0.15,right=0.6)
    
    for i,(j,_ax) in enumerate(zip([0,0.5,1.0],[ax1,ax2,ax3])):
        sampler = get_sampler(ECC_BIN, j)
        sampler.sim_until_precision(CONFIDENCE_INTERVAL_WIDTH, batch_size=4,confidence_level=CONFIDENCE_LEVEL)
        plot_scatter(_ax,sampler)
    
        print(sampler.bootstrap(system.LIBRATING))
        _ax.set_aspect('equal')
        _ax.set_xlabel('$i~\\cos{\\Omega}$',fontsize=12)
        _ax.set_ylabel('$i~\\sin{\\Omega}$',fontsize=12)
        if i==0:
            _ax.legend(fontsize=12,loc=(1.2,0.2))
        _ax.set_xlim(-np.pi,np.pi)
        _ax.set_ylim(-np.pi,np.pi)
        _ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        _ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        _ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
        _ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
        _ax.text(0.8*np.pi,1.15*np.pi,f'$j={helpers.represent_j(j)}$',ha='right',va='top',fontsize=12)
    
    
    # n = 1000
    # m = 1000
    # i_arr = np.linspace(0,np.pi,n)
    # omega_arr = np.linspace(0,2*np.pi,m)
    # state_arr = np.zeros((n,m),dtype=int)
    # gamma = get_gamma(ECC_BIN,J)
    # state_mapper = {
    #     'u':0,
    #     'p':1,
    #     'l':2,
    #     'r':3
    # }
    # for _n, i in tqdm(enumerate(i_arr),desc='Running grid',total=n):
    #     for _m, omega in enumerate(omega_arr):
    #         x,y,z = init_xyz(i,omega)
    #         _,_,_,_,_,state = integrate(0,0.01,x,y,z,ECC_BIN,gamma,1.0,1e-10)
    #         state_arr[_n,_m] = state_mapper[state]
    
    # ax.contour(omega_arr,i_arr,state_arr,levels=[0.5,1.5,2.5],colors=['k','k','k','k'],linewidths=1.5)
    
    fig.tight_layout()
    
    fig.savefig(OUTFILE)

