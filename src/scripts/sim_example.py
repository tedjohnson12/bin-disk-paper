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

PATH = paths.figures / 'sim_example.pdf'

plt.style.use('bmh')

mb = 1
fb = 0.5
ab = 0.2
eb = 0.4
J = 0.05
ap = 5*ab
mp = helpers.get_mass_planet(
    _j=J,
    m_bin=mb,
    f_bin=fb,
    e_bin=eb,
    a_bin=ab,
    a_planet=ap,
    e_planet=0
)

i_arr = np.array([0.1,0.2,0.3,0.4,0.5,-0.3,-0.4,-0.5,0.8,0.9,1.0])*np.pi

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
    _sys = system.System(binary,planet,gr=False,sim=sim)
    _sys.integrate_to_get_state(step=5,capture_freq=2)
    
    return _sys
def run_full(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = system.System(binary,planet,gr=False,sim=sim)
    _sys.integrate_to_get_path(step=10, capture_freq=2)
    
    return _sys

if __name__ in '__main__':
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax: plt.Axes
    
    for inclination in i_arr:
        sys = run_full(inclination,eb)
        state = sys.state
        c = color[state]
        ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2)
        # if abs(inclination) != np.pi/2 or J>0:
        #     ilast = np.argwhere(np.sign(sys.icosomega) != np.sign(sys.icosomega[1]))[-1][0]
        #     ifirst = 0
        #     ax.arrow(
        #         0.5*(sys.icosomega[ilast] + sys.icosomega[ifirst]),
        #         0.5*(sys.isinomega[ilast] + sys.isinomega[ifirst]),
        #         sys.icosomega[ifirst] - sys.icosomega[ilast],
        #         sys.isinomega[ifirst] - sys.isinomega[ilast],
        #         shape='full',
        #         lw=0.1,
        #         length_includes_head=True,
        #         head_width=0.1,
        #         color=c,
        #         zorder=100,
        #         ec='k'
        #     )
        # ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2,ls='--')

    lines = [
        Line2D([0],[0],color=color[system.PROGRADE],lw=2),
        Line2D([0],[0],color=color[system.RETROGRADE],lw=2),
        Line2D([0],[0],color=color[system.LIBRATING],lw=2),
        # Line2D([0],[0],color=colors.slate,lw=2,ls='--'),
    ]
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel('$i~\\cos{\\Omega}$')
    ax.set_ylabel('$i~\\sin{\\Omega}$')
    ax.legend(lines,["Prograde Circulation","Retrograde Circulation","Libration"])
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],fontsize=12)
    
    fig.tight_layout()
    
    fig.savefig(PATH)

        