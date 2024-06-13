
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import paths
import colors
from compare_ebin import run_grid


plt.style.use('bmh')

OUTFILE = paths.figures / 'ej_mesh.pdf'

N_INC = 100
N_OMEGA = 100
N_ECC = 25
N_J = 29

ECCENTRICITIES = np.linspace(0.01,0.99,N_ECC)
# JS = np.logspace(np.log10(0.1),np.log10(4.0+0.1),N_J)
# JS = JS - JS[0]
JS = np.linspace(0,2,N_J)


fracs = np.zeros((N_ECC,N_J))
for i,e in enumerate(tqdm(ECCENTRICITIES,desc='eccentricities',total=N_ECC)):
    for k,j in enumerate(JS):
        fracs[i,k] = run_grid(N_INC,N_OMEGA,e,j)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

im = ax.pcolormesh(ECCENTRICITIES,JS,fracs.T,cmap=colors.cm_teal_cream_orange)
# ax.set_yscale('log')
fig.colorbar(im,ax=ax,label='$f_{\\rm polar}$')
ax.set_xlabel('$e_{\\rm b}$')
ax.set_ylabel('$j$')

fig.savefig(OUTFILE)