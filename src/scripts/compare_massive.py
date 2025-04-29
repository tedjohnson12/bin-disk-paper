"""
Compare the phase diagram for the massless case to that of the massive case.
"""


import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import simpson

import paths
import helpers
import colors

from polar_disk_freq import system, mc, search
from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma



plt.style.use('bmh')

color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}

OUTFILE = paths.figures / 'compare_massive.pdf'
DB_PATH = paths.data / 'mc.db'
mc_out = paths.output/'mc_out.txt'
rk_out = paths.output/'rk_out.txt'
rk_err = paths.output/'rk_err.txt'

STATE_LONG_NAMES = {
    system.UNKNOWN : 'Unknown',
    system.PROGRADE: 'Circulation',
    system.RETROGRADE: 'Retrograde Circulation',
    system.LIBRATING: 'Libration'
}


MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
ECC_BIN = 0.4
SEP_PLANET = 5*SEP_BINARY
JCRIT = helpers.j_critical(ECC_BIN)
J = JCRIT
ECC_PLANET = 0

MASS_PLANET = helpers.get_mass_planet(
    _j=J,
    m_bin=MASS_BINARY,
    f_bin=FRAC_BINARY,
    e_bin=ECC_BIN,
    a_bin=SEP_BINARY,
    a_planet=SEP_PLANET,
    e_planet=ECC_PLANET
)
NU = 0
ARG_PARIAPSIS = 0
GR = False


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

def plot_full(_ax: plt.Axes, ecc: float, inclination:float,_mass: float, omega:float = np.pi/2,capture_freq=1,
              true_anomaly=NU,
              arg_pariapsis=ARG_PARIAPSIS,
              eccentricity_planet=ECC_PLANET,
              gr=GR,
              a_factor=1.0):
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
        mass_planet=_mass/np.sqrt(a_factor),
        semimajor_axis_planet=SEP_PLANET * a_factor,
        inclination_planet=inclination,
        lon_ascending_node_planet=omega,
        true_anomaly_planet=true_anomaly,
        eccentricity_planet=eccentricity_planet,
        arg_pariapsis_planet=arg_pariapsis,
        gr=gr,
        sim=None
    )
    sys.integrate_to_get_path(step=5,max_orbits=10000,capture_freq=capture_freq)
    # sys.integrate_orbits(n_orbits=1000,capture_freq=capture_freq)
    incls = sys.inclination
    lon_asc_node = sys.lon_ascending_node
    x = incls*np.cos(lon_asc_node)
    y = incls*np.sin(lon_asc_node)
    state = sys.state
    ls = '-'
    _ax.plot(x,y, c=color[state], lw=0.5,ls=ls)

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
        if (states==state).sum() == 0:
            pass
        else:
            _ax.scatter(lon_asc_node[states==state], incl[states==state], c=color[state], s=50,label=STATE_LONG_NAMES[state],marker='.')
def plot_discrepancy(_ax: plt.Axes, _ax_polar: plt.Axes,_sampler: mc.Sampler):
    lon_asc_node = np.array(_sampler.lon_ascending_nodes)
    incl = np.array(_sampler.inclinations)
    states = np.array(_sampler.states)
    gamma = get_gamma(ECC_BIN, J)
    for s, omega, i in zip(states, lon_asc_node, incl):
        x,y,z = init_xyz(i,omega)
        _,_,_,_,_,_s = integrate(0,0.01,x,y,z,ECC_BIN,gamma,1.0,1e-10)
        if s != _s and s != 'u':
            print(f'omega={omega}, i={i}, s={s}, _s={_s}')
            # plot_full(_ax, ECC_BIN, i, MASS_PLANET,omega,capture_freq=20)
            # _ax_polar.scatter(omega, i, c='k', s=5,marker='X',zorder=100)

def plot_audit(_ax: plt.Axes, _ax_polar: plt.Axes, i: float, omega: float):
    _ax_polar.scatter(omega, i, c='k', s=5,marker='X',zorder=100)
    for f in [0.6,1]:
        print(f)
        plot_full(_ax, ECC_BIN,i,MASS_PLANET,omega,capture_freq=2.,a_factor=f)
        

if __name__ in '__main__':
    fig = plt.figure(figsize=(4,5))
    extent = [0.2,0.3,0.7,0.6]
    ax_cartesian: plt.Axes = fig.add_axes(extent)
    # ax_cartesian.patch.set_alpha(0.0)
    ax_polar = fig.add_axes(extent,projection='polar',frameon=False)
    ax_polar.patch.set_agg_filter(0.0)
    fig.subplots_adjust(left=0.15,right=0.6)
    sampler = get_sampler(ECC_BIN, MASS_PLANET)
    sampler.sim_until_precision(CONFIDENCE_INTERVAL_WIDTH, batch_size=4,confidence_level=CONFIDENCE_LEVEL)
    plot_scatter(ax_polar,sampler)
    # transitions = do_search(ECC_BIN,_mass=0)
    # for transition in transitions:
    #     for i in [transition.low_value, transition.high_value]:
    #         plot_full(ax,ECC_BIN,i,0)
    #         if system.LIBRATING in [transition.low_state, transition.high_state]:
    #             plot_full(ax,ECC_BIN,-i,0)
    # transitions = do_search(ECC_BIN,_mass=MASS_PLANET)
    # for transition in transitions:
    #     for i in [transition.low_value, transition.high_value]:
    #         plot_full(ax,ECC_BIN,i,MASS_PLANET)
    #         if system.LIBRATING in [transition.low_state, transition.high_state]:
    #             plot_full(ax,ECC_BIN,-i,MASS_PLANET)
    print(sampler.bootstrap(system.LIBRATING))
    # ax.set_xlim(-np.pi*1.05,np.pi*1.05)
    # ax.set_ylim(-np.pi*1.05,np.pi*1.05)
    ax_polar.set_aspect('equal')
    ax_cartesian.set_xlabel('$i~\\cos{\\Omega}$',fontsize=12)
    ax_cartesian.set_ylabel('$i~\\sin{\\Omega}$',fontsize=12)
    ax_cartesian.set_xlim(-np.pi,np.pi)
    ax_cartesian.set_ylim(-np.pi,np.pi)
    ax_cartesian.set_aspect('equal')
    ax_cartesian.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax_cartesian.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax_cartesian.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    ax_cartesian.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    ax_polar.legend(fontsize=12,loc=(0.25,-0.55))
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    # ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=24)
    # ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=24)
    # plt.tick_params(labelsize=24)
    ax_cartesian.text(-0.9*np.pi,1.15*np.pi,f'$j={helpers.represent_j(J)}$',ha='left',va='top',fontsize=12)
    
    res = sampler.bootstrap(system.LIBRATING,0.68)
    lo,hi = res.confidence_interval
    means = res.bootstrap_distribution
    std = res.standard_error
    mean = 0.5*(lo+hi)
    s = f'{mean:.2f} \\pm {std:.2f}'
    with open(mc_out,'w',encoding='utf-8') as f:
        f.write(s)
    # plot_discrepancy(ax_cartesian, ax_polar,sampler)
    
    
    
    frac_low = None
    frac_high = None
    for _i,(n, m) in enumerate(zip([400,800],[400,800])):
        i_arr = np.linspace(0,np.pi,n+1,endpoint=True)
        omega_arr = np.linspace(0,2*np.pi,m+1)
        state_arr = np.zeros((n,m),dtype=int)
        gamma = get_gamma(ECC_BIN,J)
        state_mapper = {
            'u':0,
            'p':1,
            'l':2,
            'r':3
        }
        for _n, (i0,i1) in tqdm(enumerate(zip(i_arr[:-1],i_arr[1:])),total=n):
            for _m, (omega0,omega1) in enumerate(zip(omega_arr[:-1],omega_arr[1:])):
                i = (i0+i1)/2
                omega = (omega0+omega1)/2
                x,y,z = init_xyz(i,omega)
                _,_,_,_,_,_,state = integrate(0,0.01,x,y,z,ECC_BIN,gamma,1.0,1e-10)
                state_arr[_n,_m] = state_mapper[state]
        i_cen = 0.5*(i_arr[:-1]+i_arr[1:])
        omega_cen = 0.5*(omega_arr[:-1]+omega_arr[1:])
        
        ax_polar.contour(omega_cen,i_cen,state_arr,levels=[0.5,1.5,2.5],colors=['k','k','k','k'],linewidths=1.5)
        
        oo,ii = np.meshgrid(omega_cen,i_cen)
        area = np.sin(ii)
        is_polar = state_arr==state_mapper['l']
        integrand2d = area*is_polar
        integrand1d = simpson(y=integrand2d,x=omega_cen,axis=1)
        frac = simpson(y=integrand1d,x=i_cen)/(4*np.pi)
        if _i == 0:
            frac_low = frac
        else:
            frac_high = frac
    error = abs(frac_high-frac_low)
    with open(rk_out,'w',encoding='utf-8') as f:
        f.write(f'{frac_low:.2f}')
    with open(rk_err,'w',encoding='utf-8') as f:
        f.write(f'10^{{{round(np.log10(error)):.0f}}}')
    
    # audit_i = [3.013643]
    audit_i = [0.88]
    audit_omega = [4.068365]
    audit_omega = [np.pi/2]
    # for i,omega in zip(audit_i,audit_omega):
    #     plot_audit(ax_cartesian,ax_polar,i,omega)
    
    
    
    fig.savefig(OUTFILE)

