"""
Do a numerical integral in the high-j limit
"""

from scipy.integrate import simpson
import numpy as np

import paths

outfile = paths.output / 'high_j_integral.txt'

N = 1001

x = np.linspace(0,np.pi,N)
i_crit_low = np.arcsin(np.sqrt(2/5))
i_crit_high = np.pi - i_crit_low
omega = np.where((x<i_crit_low) | (x>i_crit_high),np.pi/2,np.arcsin(np.sqrt(2/5)/np.sin(x)))
y = (1 - 2/np.pi *omega) * np.sin(x) * 0.5

frac = simpson(y=y,x=x)

print(f'Fraction in high j limit: {frac}')

with open(outfile,'w',encoding='utf-8') as f:
    f.write(f'{frac:.2f}')