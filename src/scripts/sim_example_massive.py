"""
An example showing how our simulation works in the massless case.
"""

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import numpy as np
from matplotlib.lines import Line2D
from time import time

from misaligned_cb_disk import params, system
import rebound

import paths
import colors
import helpers

PATH = paths.figures / 'sim_example_massive.pdf'

mb = 1
fb = 0.5
ab = 0.2
eb = 0.4
GR = False

j_crit = helpers.j_critical(eb)
print(f'Critical relative angular momentum: {j_crit}')

j = 1
ap = 5*ab
mp = helpers.get_mass_planet(j,mb,fb,eb,ab,ap,0)
# i_arr = np.array([1.2,1.4,1.55,1.8])*np.pi
i_arr = np.array([-0.99,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])*np.pi

# i_arr = np.array([1.0])*np.pi

color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}

def run(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = system.System(binary,planet,GR,sim=sim)
    _sys.integrate_to_get_state(step=20,capture_freq=0.5,max_orbits=10000)
    
    return _sys
def run_full(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = system.System(binary,planet,GR,sim=sim)
    # _sys.integrate_to_get_path(step=10, capture_freq=30,max_orbits=10000)
    _sys.integrate_orbits(n_orbits=10,capture_freq=30)
    
    return _sys

if __name__ in '__main__':
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax:Axis
    start = time()
    for inclination in i_arr:
        print(f'Running for i = {inclination}')
        sys = run(inclination,eb)
        state = sys.state
        c = color[state]
        # ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2)
        sys = run_full(inclination,eb)
        ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2)
    print(f'Took {time()-start} seconds')
    
    # print('Running one on the outside')
    # binary = params.Binary(mb,fb,ab,eb)
    # planet = params.Planet(mp,ap,np.pi*0.5,0.1,0,0,0)
    # sim = rebound.Simulation()
    # sys = system.System(binary,planet,GR,sim=sim)
    # sys.integrate_to_get_path(step=10, capture_freq=30,max_orbits=10000)
    # sys.integrate_orbits(n_orbits=1000,capture_freq=30)
    # state = sys.state
    # c = color[state]
    # ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2)

    lines = [
        Line2D([0],[0],color=color[system.PROGRADE],lw=2),
        Line2D([0],[0],color=color[system.RETROGRADE],lw=2),
        Line2D([0],[0],color=color[system.LIBRATING],lw=2),
        # Line2D([0],[0],color=colors.slate,lw=2,ls='--'),
    ]
    ax.set_xlim(-np.pi*1.05,np.pi*1.05)
    ax.set_ylim(-np.pi*1.05,np.pi*1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('$i~\\cos{\\Omega}$',fontsize=24)
    ax.set_ylabel('$i~\\sin{\\Omega}$',fontsize=24)
    # ax.legend(lines,["Prograde","Retrograde","Librating"],fontsize=24)
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=24)
    ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=24)
    # plt.tick_params(labelsize=24)
    fig.text(0.9,0.95,f'$j={helpers.represent_j(j)}$',ha='right',va='top',fontsize=24)
    
    fig.savefig(PATH)

        