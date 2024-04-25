"""
An example showing how our simulation works in the massless case.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from misaligned_cb_disk import params, system
import rebound

import paths
import colors

PATH = paths.figures / 'sim_example.pdf'

mb = 1
fb = 0.5
ab = 0.2
eb = 0.4

mp = 0e-3
ap = 1
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
    _sys = system.System(binary,planet,sim)
    _sys.integrate_to_get_state(step=5,capture_freq=2)
    
    return _sys
def run_full(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = system.System(binary,planet,sim)
    _sys.integrate_to_get_path(step=10, capture_freq=2)
    
    return _sys

if __name__ in '__main__':
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    
    for inclination in i_arr:
        sys = run(inclination,eb)
        state = sys.state
        c = color[state]
        ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2)
        sys = run_full(inclination,eb)
        ax.plot(sys.icosomega,sys.isinomega,c=c,lw=2,ls='--')

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
    ax.legend(lines,["Prograde","Retrograde","Librating","Unnecessary"])
    
    fig.savefig(PATH)

        