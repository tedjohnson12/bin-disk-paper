"""
Find the precession timescale compared to the massless case.
"""
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma, get_i, get_omega
from polar_disk_freq.precession import get_tp_over_tpj0
from helpers import j_critical, get_stationary_inclination
import paths
import colors

plt.style.use('bmh')

OUTFILE = paths.figures / 'precession_time.pdf'

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
    eb = 0.4
    i0 = 0.9
    fivedeg = -5/180*np.pi # 5 degrees
    omega0 = np.pi/2-0.01
    ab_ap = 1/5
    jcrit = j_critical(eb)
    js = np.linspace(0,2,2000)
    # js=np.logspace(-2,1.5,2000)
    fig = plt.figure(figsize=(4,8))
    gs = fig.add_gridspec(2,1)
    ax = fig.add_subplot(gs[0])
    ax: plt.Axes
    ax2 = fig.add_subplot(gs[1])
        
    rats = np.zeros(len(js))
    states = np.zeros(len(js),dtype=int)
    for i,j in enumerate(tqdm(js)):
        rat, state = get_tp_over_tpj0(eb,j,i0,omega0,f,ab_ap,
                                      TAU_INIT,DTAU_INIT,WALLTIME,EPSILON)
        x,y,z = init_xyz(i0,omega0)
        gamma = get_gamma(eb,j)
        _,lx,ly,lz,_,_,_ = integrate(TAU_INIT,DTAU_INIT,x,y,z,eb,gamma,WALLTIME,epsilon=EPSILON)
        _inc = get_i(lx,ly,lz)
        _omega = get_omega(lx,ly)
        ax2.plot(_inc*np.cos(_omega),_inc*np.sin(_omega),c='k',alpha=0.1,lw=0.3)
        rats[i] = rat
        states[i] = STATE_MAPPER[state]
    is_prograde = states == STATE_MAPPER['p']
    is_librating = states == STATE_MAPPER['l']
    is_retrograde = states == STATE_MAPPER['r']
    if True:
        first_prograde = np.argwhere(is_prograde)[0][0]
        before_prograde = np.arange(0,len(js)) < first_prograde
        after_prograde = np.arange(0,len(js)) >= first_prograde
        ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        ax.plot(js[is_librating & before_prograde],rats[is_librating & before_prograde],color=colors.state_colors['l'],ls='-',lw=2,label='Librating')
        ax.plot(js[is_librating & after_prograde],rats[is_librating & after_prograde],color=colors.state_colors['l'],ls='-',lw=2)
    else:
        ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        ax.plot(js[is_librating ],rats[is_librating ],color=colors.state_colors['l'],ls='-',lw=2,label='Librating')
        ax.plot(js[is_retrograde],rats[is_retrograde],color=colors.state_colors['r'],ls='-',lw=2)
    ax2.annotate('$(i_0,\\Omega_0)$',xy=(0,i0),arrowprops=dict(arrowstyle='->',color='k',lw=2),clip_on=False,xycoords='data',xytext=(0.2,i0-0.4),fontsize=12)
    ax2.scatter(0,i0,c='r',s=5,marker='X',zorder=100)
    
    
    # stationary
    rats = np.zeros(len(js))
    states = np.zeros(len(js),dtype=int)
    for i,j in enumerate(tqdm(js)):
        i0 = get_stationary_inclination(j,eb) - fivedeg
        rat, state = get_tp_over_tpj0(eb,j,i0,omega0,f,ab_ap,
                                      TAU_INIT,DTAU_INIT,WALLTIME,EPSILON)
        x,y,z = init_xyz(i0,omega0)
        gamma = get_gamma(eb,j)
        _,lx,ly,lz,_,_,_ = integrate(TAU_INIT,DTAU_INIT,x,y,z,eb,gamma,WALLTIME,epsilon=EPSILON)
        _inc = get_i(lx,ly,lz)
        _omega = get_omega(lx,ly)
        # ax2.plot(_inc*np.cos(_omega),_inc*np.sin(_omega),c='k',alpha=0.1,lw=0.3)
        rats[i] = rat
        states[i] = STATE_MAPPER[state]
    is_prograde = states == STATE_MAPPER['p']
    is_librating = states == STATE_MAPPER['l']
    is_retrograde = states == STATE_MAPPER['r']
    if False:
        first_prograde = np.argwhere(is_prograde)[0][0]
        before_prograde = np.arange(0,len(js)) < first_prograde
        after_prograde = np.arange(0,len(js)) >= first_prograde
        ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        ax.plot(js[is_librating & before_prograde],rats[is_librating & before_prograde],color=colors.state_colors['l'],ls='-',lw=2,label='Librating')
        ax.plot(js[is_librating & after_prograde],rats[is_librating & after_prograde],color=colors.state_colors['l'],ls='-',lw=2)
    else:
        # ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        ax.plot(js[is_librating ],rats[is_librating ],color=colors.state_colors['l'],ls='--',lw=2,label='$5^\\circ$ from stationary point')
        # ax.plot(js[is_retrograde],rats[is_retrograde],color=colors.state_colors['r'],ls='-',lw=2)
    
    # coplanar
    rats = np.zeros(len(js))
    states = np.zeros(len(js),dtype=int)
    for i,j in enumerate(tqdm(js)):
        i0 =  fivedeg
        rat, state = get_tp_over_tpj0(eb,j,i0,omega0,f,ab_ap,
                                      TAU_INIT,DTAU_INIT,WALLTIME,EPSILON)
        x,y,z = init_xyz(i0,omega0)
        gamma = get_gamma(eb,j)
        _,lx,ly,lz,_,_,_ = integrate(TAU_INIT,DTAU_INIT,x,y,z,eb,gamma,WALLTIME,epsilon=EPSILON)
        _inc = get_i(lx,ly,lz)
        _omega = get_omega(lx,ly)
        # ax2.plot(_inc*np.cos(_omega),_inc*np.sin(_omega),c='k',alpha=0.1,lw=0.3)
        rats[i] = rat
        states[i] = STATE_MAPPER[state]
    is_prograde = states == STATE_MAPPER['p']
    is_librating = states == STATE_MAPPER['l']
    is_retrograde = states == STATE_MAPPER['r']
    if False:
        first_prograde = np.argwhere(is_prograde)[0][0]
        before_prograde = np.arange(0,len(js)) < first_prograde
        after_prograde = np.arange(0,len(js)) >= first_prograde
        ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        ax.plot(js[is_librating & before_prograde],rats[is_librating & before_prograde],color=colors.state_colors['l'],ls='-',lw=2,label='Librating')
        ax.plot(js[is_librating & after_prograde],rats[is_librating & after_prograde],color=colors.state_colors['l'],ls='-',lw=2)
    else:
        # ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='-',lw=2,label='Prograde')
        # ax.plot(js[is_librating ],rats[is_librating ],color=colors.state_colors['l'],ls='--',lw=2,label='$5^\\circ from stationary point$')
        ax.plot(js[is_prograde],rats[is_prograde],color=colors.state_colors['p'],ls='--',lw=2,label='$5^\\circ$ from coplanar')

    # ax.plot(js[is_retrograde],rats[is_retrograde],color=colors.state_colors['r'],ls='-',lw=2)
    # ax.legend(prop={'size':12,'family':'serif'})
    ax.set_xlabel('$j$')
    ax.set_ylabel('$t_{\\rm p}/t_{{\\rm p}, j=0}$')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_aspect('equal')
    ax2.set_xlabel('$i~\\cos{\\Omega}$')
    ax2.set_ylabel('$i~\\sin{\\Omega}$')
    ax2.set_aspect('equal')
    ax2.set_xlim(0,np.pi * 3 / 4)
    ax2.set_ylim(0,np.pi * 3 / 4)
    ax2.set_xticks([0,np.pi/4,np.pi/2,np.pi*3/4])
    ax2.set_yticks([0,np.pi/4,np.pi/2,np.pi*3/4])
    ax2.set_xticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$'],fontsize=12)
    ax2.set_yticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$'],fontsize=12)
    fig.text(0.65,0.93,'$i=0.9$',fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTFILE)