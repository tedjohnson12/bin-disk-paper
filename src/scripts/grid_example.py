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
    i_arr = np.linspace(0,np.pi,n_i+1,endpoint=True)
    omega_arr = np.linspace(0,2*np.pi,n_omega+1,endpoint=True)
    state_arr = np.zeros((n_i,n_omega),dtype=int)*np.nan
    tau_p_arr = np.zeros((n_i,n_omega),dtype=float)*np.nan
    gamma = get_gamma(e_bin,_j)
    for _n, (i0,i1) in tqdm(enumerate(zip(i_arr[:-1],i_arr[1:])),total=n_i):
        for _m, (omega0,omega1) in enumerate(zip(omega_arr[:-1],omega_arr[1:])):
            i = (i0+i1)/2
            omega = (omega0+omega1)/2
            x,y,z = init_xyz(i,omega)
            _,_,_,_,_,taup,state = integrate(TAU_INIT,DTAU_INIT,x,y,z,e_bin,gamma,WALLTIME,epsilon=EPSILON)
            state_arr[_n,_m] = STATE_MAPPER[state]
            tau_p_arr[_n,_m] = taup
    i_mean = 0.5*(i_arr[:-1]+i_arr[1:])
    omega_mean = 0.5*(omega_arr[:-1]+omega_arr[1:])
    _,ii = np.meshgrid(omega_mean,i_mean)
    area = np.sin(ii)
    is_polar = state_arr==STATE_MAPPER['l']
    integrand2d = area*is_polar
    integrand1d = simpson(y=integrand2d,x=omega_mean,axis=1)
    _frac = simpson(y=integrand1d,x=i_mean)/(4*np.pi)
    return i_arr, omega_arr, _frac, state_arr, tau_p_arr

def tau_to_tOmegab(tau:np,r_over_a: float, f: float) -> np.ndarray:
    """
    Convert a tau value to t * Omega_b
    """
    return 4/3 * tau * r_over_a**(7/2) * 1/(f*(1-f)*np.sqrt(1+f))


if __name__ == '__main__':
    N_INC = 1000
    N_OMEGA = 1001
    E_BIN = 0.4
    J = 0.1
    T_DISK = 0.5e6
    AP_OVER_AB = 15
    FB = 0.5
    i_arr, omega_arr,frac, state_arr, tau_p_arr = run_grid(N_INC,N_OMEGA,E_BIN,J)
    
    t_p_omega_b = tau_to_tOmegab(tau_p_arr,AP_OVER_AB,FB)
    
    fig = plt.figure(figsize=(4,4))
    extent = [0.2,0.15,0.7,0.6]
    ax_cartesian: plt.Axes = fig.add_axes(extent)
    # ax_cartesian.patch.set_alpha(0.0)
    ax_polar = fig.add_axes(extent,projection='polar',frameon=False)
    ax_polar.patch.set_agg_filter(0.0)
    fig.subplots_adjust(left=0.15,right=0.6)
    im = ax_polar.pcolormesh(omega_arr,i_arr,state_arr,rasterized=True,cmap=CMAP,linewidth=0.0,vmin=1,vmax=3)
    # ax_polar.pcolormesh(omega_arr,i_arr,np.where(t_p_omega_b>T_DISK,1,np.nan),alpha=1.0,cmap='gray',rasterized=True,linewidth=0.0)
    
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
    
    fnames = ['grid_example_n_sims.txt','grid_example_eb.txt','grid_example_j.txt']
    values = [
        f"$10^{np.log10(N_INC*N_OMEGA):.0f}$",
        f"{E_BIN:.1f}",
        f"{J:.1f}"
    ]
    for fname,value in zip(fnames,values):
        with open(paths.output / fname,'w',encoding='utf-8') as f:
            f.write(value)
    