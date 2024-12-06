"""
Make a cool 3d plot
"""

import numpy as np
import matplotlib.pyplot as plt

from misaligned_cb_disk import params, system, utils

import paths
import colors
import helpers

PATH = paths.figures / 'plot3d.pdf'

mb = 1
fb = 0.5
ab = 0.2
eb = 0.4
GR = False

j_crit = helpers.j_critical(eb)
print(f'Critical relative angular momentum: {j_crit}')

j = j_crit*0
ap = 5*ab
mp = helpers.get_mass_planet(j,mb,fb,eb,ab,ap,0)
# i_arr = np.array([-0.99,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])*np.pi
i_arr = np.array([0.15,0.4,-0.4,0.85])*np.pi
color = {
    system.UNKNOWN : colors.brown,
    system.PROGRADE: colors.yellow,
    system.RETROGRADE: colors.dark_orange,
    system.LIBRATING: colors.teal
}

get_zorder = {
    1: 100,
    5: 100,
    2:0,
    6:0,
    3:-100,
    7:-100,
    4: 0,
    8: 0
}

def is_in_octant(
    x:np.ndarray,
    y:np.ndarray,
    z:np.ndarray,
    i:int
):
    if i == 1:
        return (x>=0) & (y>=0) & (z>=0)
    elif i == 2:
        return (x<0) & (y>=0) & (z>=0)
    elif i == 3:
        return (x<0) & (y<0) & (z>=0)
    elif i == 4:
        return (x>=0) & (y<0) & (z>=0)
    elif i == 5:
        return (x>=0) & (y>=0) & (z<0)
    elif i == 6:
        return (x<0) & (y>=0) & (z<0)
    elif i == 7:
        return (x<0) & (y<0) & (z<0)
    elif i == 8:
        return (x>=0) & (y<0) & (z<0)
    else:
        raise ValueError(f'Invalid i: {i}')


if __name__ == '__main__':
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    
    for i in i_arr:
        sys = system.System.from_params(
            mass_binary=mb,
            mass_fraction=fb,
            semimajor_axis_binary=ab,
            eccentricity_binary=eb,
            mass_planet=mp,
            semimajor_axis_planet=ap,
            inclination_planet=i,
            lon_ascending_node_planet=np.pi/2,
            true_anomaly_planet=0,
            eccentricity_planet=0,
            arg_pariapsis_planet=0,
            gr=GR,
        )
        # sys.integrate_to_get_path(step=50, capture_freq=3,max_orbits=100000)
        sys.integrate_orbits(n_orbits=800,capture_freq=3)
        
        phi = sys.lon_ascending_node - np.pi/2
        theta = sys.inclination
        r = 1
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)
        cut=2
        # x,y,z = x[:-2],y[:-2],z[:-2]
        for i in range(1,8+1):
            zorder = get_zorder[i]
            reg = is_in_octant(x,y,z,i)
            _phi = phi[reg]
            order = np.argsort(np.unwrap(_phi))
            _x = x[reg][order]
            _y = y[reg][order]
            _z = z[reg][order]
            ax.plot(_x,_y,_z,zorder=zorder,c=color[sys.state],alpha=1,lw=2)
            ax.quiver(
                [0],[0],[0],_x,_y,_z,arrow_length_ratio=0.0, color=color[sys.state],alpha=0.05
            )
        ax.quiver(
            [0],[0],[0],x[-1],y[-1],z[-1],arrow_length_ratio=0.1, color=color[sys.state],alpha=1
        )
    x, y, z = np.array([[-2,0,0],[0,-2,0],[0,0,-2]])
    u, v, w = np.array([[4,0,0],[0,4,0],[0,0,4]])
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    r=0.99
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color=colors.cream, alpha=0.2, linewidth=0)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_zlim(-1.1,1.1)
    ax.set_aspect('equal')
    ax.view_init(20, 45)
    fsize=32
    ax.text(0.0,0.1,1.2,'${\\bf\\ell}_b$',fontdict={'size':fsize})
    ax.text(1.9,0.0,0.3,'${\\bf e}_b$',fontdict={'size':fsize})
    ax.text(1.5,0.0,1.4,'Prograde',fontdict={'size':fsize},color=color[system.PROGRADE])
    ax.text(-0.85,0.25,0.5,'Librating',fontdict={'size':fsize},color=color[system.LIBRATING])
    ax.text(0,0.2,-1.2,'Retrograde',fontdict={'size':fsize},color=color[system.RETROGRADE])

    
    ax.set_axis_off()
    fig.savefig(PATH)