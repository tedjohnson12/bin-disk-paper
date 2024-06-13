
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

OUTFILE = paths.figures / 'compare_ebin.pdf'


N_ANALYTIC = 100
ECCENTRICITIES_ANALYTIC = np.linspace(0.001,0.999,N_ANALYTIC)

N_INC = 100
N_OMEGA = 100
N_GRID = 41
ECCENTRICITIES_GRID = np.linspace(0.01,0.99,N_GRID)

JS = [0.0,0.1,0.25,0.5,1.0,2.0,4.0]

STATE_MAPPER = {
    'u':0,
    'p':1,
    'l':2,
    'r':3
}

TAU_INIT = 0.0
DTAU_INIT = 0.01
WALLTIME = 1 # seconds per run (usually 1/50,000 s)
EPSILON = 1e-10 # Error allowance for integration

CMAP = colors.cm_teal_cream_orange
COLORS = CMAP(np.linspace(0, 1, len(JS)))



def plot_analytic(_ax: plt.Axes):
    """
    Plot the analytic solution from Zanazzi 2018.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on.
    """
    polar_fracs = [Zanazzi2018.frac_polar(e) for e in ECCENTRICITIES_ANALYTIC]
    _ax.plot(ECCENTRICITIES_ANALYTIC, polar_fracs,
             c=colors.yellow, label=Zanazzi2018.citet,lw=3,ls='--')
    _j = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar_fracs = [MartinLubow2019.frac_polar(_j,e) for e in ECCENTRICITIES_ANALYTIC]
    _ax.plot(ECCENTRICITIES_ANALYTIC, polar_fracs,
             c=colors.slate, label=f'{MartinLubow2019.citet},\n$j={helpers.represent_j(_j)}$',lw=3,ls='--')
    _j = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar_fracs = [MartinLubow2019.frac_polar(_j,e) for e in ECCENTRICITIES_ANALYTIC]
    _ax.plot(ECCENTRICITIES_ANALYTIC, polar_fracs,
             c=colors.slate, label=f'{MartinLubow2019.citet},\n$j={helpers.represent_j(_j)}$',lw=3,ls=':')


def run_grid(
    n_i:int,
    n_omega:int,
    e_bin:float,
    _j:float
):
    """
    Run a grid of simulations and return the polar fraction.
    """
    i_arr = np.linspace(0,np.pi,n_i,endpoint=False)
    omega_arr = np.linspace(0,2*np.pi,n_omega,endpoint=False)
    state_arr = np.zeros((n_i,n_omega),dtype=int)
    gamma = get_gamma(e_bin,_j)
    for _n, i in enumerate(i_arr):
        for _m, omega in enumerate(omega_arr):
            x,y,z = init_xyz(i,omega)
            _,_,_,_,_,state = integrate(TAU_INIT,DTAU_INIT,x,y,z,e_bin,gamma,WALLTIME,epsilon=EPSILON)
            state_arr[_n,_m] = STATE_MAPPER[state]
    _,ii = np.meshgrid(omega_arr,i_arr)
    area = np.sin(ii)
    is_polar = state_arr==STATE_MAPPER['l']
    integrand2d = area*is_polar
    integrand1d = simpson(y=integrand2d,x=omega_arr,axis=1)
    _frac = simpson(y=integrand1d,x=i_arr)/(4*np.pi)
    return _frac

def get_frac_func_of_ebin(
    n_i:int,
    n_omega:int,
    e_bin:np.ndarray,
    _j:float
):
    """
    Get the polar fraction as a function of binary eccentricity for a single value of j.
    """
    frac_arr = np.zeros_like(e_bin)
    for _i, e in tqdm(enumerate(e_bin),desc=f'j={_j:.2f}',total=len(e_bin)):
        frac_arr[_i] = run_grid(n_i,n_omega,e,_j)
    return frac_arr

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(right=0.7)
    ax: plt.Axes
    plot_analytic(ax)
    eccentricities = np.array(ECCENTRICITIES_GRID)
    for j, color in zip(JS, COLORS):
        frac = get_frac_func_of_ebin(N_INC,N_OMEGA,eccentricities,j)
        ax.plot(eccentricities, frac, c=color, label=f'$j={helpers.represent_j(j)}$')
    
    # frac_crit = np.zeros_like(eccentricities)
    # for i,e in enumerate(eccentricities):
    #     jcrit = helpers.j_critical(e)
    #     frac_crit[i] = run_grid(N_INC,N_OMEGA,e,jcrit)
    
    # ax.plot(eccentricities, frac_crit, c='k', label='$j_{crit}$',ls='--',lw=3)
    
    
    ax.set_xlabel('Binary Eccentricity')
    ax.set_ylabel('Polar Fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc=(1.05,0.15))

    fig.savefig(OUTFILE)