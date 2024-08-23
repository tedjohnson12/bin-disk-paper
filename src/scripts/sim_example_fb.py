"""
An example showing how our simulation works in the massless case.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from polar_disk_freq import params, system
import rebound

import paths
import colors
import helpers

PATH = paths.figures / 'sim_example_fb.pdf'

plt.style.use('bmh')

mb = 1
ab = 0.2
eb = 0.4
J = 2
ap = 5*ab


i_arr = np.array([0.1,0.2,0.3,0.4,0.5,-0.3,-0.4,-0.5,0.8,0.9,1.0])*np.pi

color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}
def run_full(i:float,e:float, fb:float)->system.System:
    mp = helpers.get_mass_planet(
        _j=J,
        m_bin=mb,
        f_bin=fb,
        e_bin=eb,
        a_bin=ab,
        a_planet=ap,
        e_planet=0
    )
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = system.System(binary,planet,gr=False,sim=sim)
    _sys.integrate_to_get_path(step=10, capture_freq=2,max_orbits=10000)
    
    return _sys

if __name__ in '__main__':
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax: plt.Axes
    
    for inclination in i_arr:
        for ls, lw, fb in zip(['-', '--'],[0.1,0.1], [0.5, 0.95]):
            sys = run_full(inclination,eb,fb)
            state = sys.state
            c = color[state]
            ax.plot(sys.icosomega,sys.isinomega,c=c,lw=lw,ls=ls)

    lines = [
        Line2D([0],[0],color=color[system.PROGRADE],lw=2),
        Line2D([0],[0],color=color[system.RETROGRADE],lw=2),
        Line2D([0],[0],color=color[system.LIBRATING],lw=2),
        Line2D([0],[0],color=colors.slate,lw=2,ls='--'),
    ]
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel('$i~\\cos{\\Omega}$')
    ax.set_ylabel('$i~\\sin{\\Omega}$')
    ax.legend(lines,["Prograde","Retrograde","Librating", "$f_b=0.9$"])
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    
    fig.tight_layout()
    
    fig.savefig(PATH)

        