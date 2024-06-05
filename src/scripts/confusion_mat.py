
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from polar_disk_freq.rk4 import integrate, init_xyz, get_gamma

from polar_disk_freq.db import get_var
from polar_disk_freq.params import G

import paths

DB_PATH = paths.data / 'mc.db'
OUTFILE = paths.figures / 'confusion_mat.pdf'


ytrue = []
ypred = []

conn = sqlite3.connect(DB_PATH)

res = get_var(conn,'*',has_gr=False)

for row in tqdm(res,desc='Going through rows',total=len(res)):
    mb,fb,ab,eb,mp,ap,_,ep,_,i,omega,_,state = row
    jp = mp * np.sqrt(G*(mb+mp)*ap)
    jb = mb*(1-fb) * fb * np.sqrt(G * mb * ab * (1-eb**2))
    j = jp/jb
    gamma = get_gamma(eb,j)
    x,y,z = init_xyz(i,omega)
    _, _, _, _, _, state_pred = integrate(0,0.01,x,y,z,eb,gamma,1.0,1e-10)
    ytrue.append(state)
    ypred.append(state_pred)

are_equal = np.equal(ytrue,ypred)
print(np.sum(are_equal)/len(are_equal))
mat = confusion_matrix(ytrue,ypred,labels=['p','r','l','u'])
disp = ConfusionMatrixDisplay(mat,display_labels=[
    'Prograde','Retrograde','Polar','Unknown'
])
disp.plot()

plt.xlabel('RK4')
plt.ylabel('REBOUND')
plt.savefig(OUTFILE) 