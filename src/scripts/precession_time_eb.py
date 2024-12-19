"""
Find the precession timescale compared to the massless case.
"""
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma, get_i, get_omega
from polar_disk_freq.precession import get_tp_over_tpj0, get_gamma_2
from helpers import j_critical
import paths
import colors

plt.style.use('bmh')

OUTFILE = paths.figures / 'precession_time_eb.pdf'

TAU_INIT = 0.0
DTAU_INIT = 0.01
WALLTIME = 1 # seconds per run (usually 1/50,000 s)
EPSILON = 1e-12 # Error allowance for integration
STATE_MAPPER = {
    'u':0,
    'p':1,
    'l':2,
    'r':3
}

if __name__ == '__main__':
    f = 0.5
    eb = 0.9
    fivedeg = -5/180*np.pi # 5 degrees
    omega0 = np.pi/2-0.01
    ab_ap = 1/1e6
    # js = np.linspace(0,2,2000)
    js=np.logspace(-2,1.5,2000)
    fig = plt.figure(figsize=(4,4))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0])
    ax: plt.Axes
    
    # coplanar
    for scale in np.linspace(0.1,10,5):
        rats = np.zeros(len(js))
        states = np.zeros(len(js),dtype=int)
        for i,j in enumerate(tqdm(js)):
            i0 =  -fivedeg * scale
            rat, state = get_tp_over_tpj0(eb,j,i0,omega0,f,ab_ap,
                                        TAU_INIT,DTAU_INIT,WALLTIME,EPSILON)
            rats[i] = rat
            states[i] = STATE_MAPPER[state]
        is_prograde = states == STATE_MAPPER['p']
        is_librating = states == STATE_MAPPER['l']
        is_retrograde = states == STATE_MAPPER['r']
        needs_separating = np.any(is_librating) and np.any(is_prograde)
        
        ls = '-'
        lw = 1
        if needs_separating:
            first_prograde = np.argwhere(is_prograde)[0][0]
            is_first_prograde = np.arange(0,len(js)) == first_prograde
            before_prograde = np.arange(0,len(js)) <= first_prograde
            after_prograde = np.arange(0,len(js)) >= first_prograde
            ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls=ls,lw=lw,label='Prograde')
            ax.plot(js[(is_librating & before_prograde) | is_first_prograde],rats[(is_librating & before_prograde)| is_first_prograde],color=colors.state_colors['l'],ls=ls,lw=lw,label='Librating')
            ax.plot(js[is_librating & after_prograde],rats[is_librating & after_prograde],color=colors.state_colors['l'],ls=ls,lw=lw)
        else:
            # ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
            ax.plot(js[is_librating ],rats[is_librating ],color=colors.state_colors['l'],ls=ls,lw=lw,label='$5^\\circ from stationary point$')
            ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls=ls,lw=lw,label='$5^\\circ$ from coplanar')
        coeffs = np.polyfit(np.log(js[-10:]),np.log(rats[-10:]),1)
        slope = coeffs[0]
        print(f'slope={slope}')

    # ax.plot(js[is_retrograde],rats[is_retrograde],color=colors.state_colors['r'],ls='-',lw=2)
    # ax.legend(prop={'size':12,'family':'serif'})
    ax.set_xlabel('$j$')
    ax.set_ylabel('$t_{\\rm p}/t_{{\\rm p}, j=0}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')    
    fig.text(0.65,0.93,'$i=0.9$',fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTFILE)