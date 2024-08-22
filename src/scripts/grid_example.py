import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Rectangle
import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import simpson

from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma

import helpers
import paths
import colors

plt.style.use('bmh')



OUTFILE = paths.figures / 'grid_example.pdf'

TAU_INIT = 0.0
DTAU_INIT = 0.01
WALLTIME = 1 # seconds per run (usually 1/50,000 s)
EPSILON = 1e-10 # Error allowance for integration
STATE_MAPPER = {
    'u':0,
    'p':1,
    'l':2,
    'r':3
}
CMAP = matplotlib.colors.ListedColormap(
    [colors.state_colors[s] for s in STATE_MAPPER.keys() if s != 'u']
)

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
    omega_arr = np.linspace(0,2*np.pi,n_omega,endpoint=True)
    state_arr = np.zeros((n_i,n_omega),dtype=int)*np.nan
    gamma = get_gamma(e_bin,_j)
    for _n, i in tqdm(enumerate(i_arr[1:]),total=n_i-1):
        for _m, omega in enumerate(omega_arr[:-1]):
            x,y,z = init_xyz(i,omega)
            _,_,_,_,_,state = integrate(TAU_INIT,DTAU_INIT,x,y,z,e_bin,gamma,WALLTIME,epsilon=EPSILON)
            state_arr[_n,_m] = STATE_MAPPER[state]
    _,ii = np.meshgrid(omega_arr,i_arr)
    area = np.sin(ii)
    is_polar = state_arr==STATE_MAPPER['l']
    integrand2d = area*is_polar
    integrand1d = simpson(y=integrand2d,x=omega_arr,axis=1)
    _frac = simpson(y=integrand1d,x=i_arr)/(4*np.pi)
    return i_arr, omega_arr, _frac, state_arr


if __name__ == '__main__':
    N_INC = 1000
    N_OMEGA = 1001
    E_BIN = 0.5
    J = 0.1
    i_arr, omega_arr,frac, state_arr = run_grid(N_INC,N_OMEGA,E_BIN,J)
    
    fig = plt.figure(figsize=(4,4))
    extent = [0.2,0.15,0.7,0.6]
    ax_cartesian: plt.Axes = fig.add_axes(extent)
    # ax_cartesian.patch.set_alpha(0.0)
    ax_polar = fig.add_axes(extent,projection='polar',frameon=False)
    ax_polar.patch.set_agg_filter(0.0)
    fig.subplots_adjust(left=0.15,right=0.6)
    im = ax_polar.pcolormesh(omega_arr,i_arr,state_arr,rasterized=True,cmap=CMAP,linewidth=0.0)
    
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
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    # fig.text(0.55,0.5,'Prograde',fontsize=10,ha='center',va='center')
    # fig.text(0.55,0.7,'Polar',fontsize=10,ha='center',va='center')
    # fig.text(0.55,0.9,'Retrograde',fontsize=10,ha='center',va='center')
    
    patches = [
        Rectangle((0,0),1,1,facecolor=colors.state_colors[s],alpha=1) for s in STATE_MAPPER.keys() if s!='u'
    ]
    ax_cartesian.legend(
        patches,
        ['Prograde','Polar','Retrograde'],
        loc=(0.,1.05)
    )
    ax_cartesian.text(0.6, 1.05, f'$f_{{\\rm polar}} = {frac:.3f}$',transform=ax_cartesian.transAxes)
    
    fig.savefig(OUTFILE)
    