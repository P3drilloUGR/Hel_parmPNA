#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pytraj as pt
from scipy.optimize import minimize
from numpy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Hel_parm_funcs import Triads, hel_parms
from alive_progress import alive_bar; import time
from scipy.spatial.transform import Rotation as R
import argparse
def min_normal(normal, points): #Funciona bastante bien
    distance=np.abs(np.dot(points,normal))

    return np.sum(distance**2)
    
                                        #                                                                                   #      
#Bases=["R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","R","R","Y","R","R","Y","R","R","Y","Y","R","Y","R","R","Y","R","R","Y"] #Poner el tipo de base, pirimidica (Y) o puridica (R)(ej. A/G puridica, C,U,T pirimídica), esto influye en la triada de referencia
#Bases=["R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y"] #Probar las bases pequeñas
Bases=["Y","R","Y","Y","R","Y"]

# We are going to analyze the helical parameters of a generic nucleic acid, following the instructions given by: https://doi.org/10.1006/jmbi.1997.1346.
# For this project we will use pytraj, scipy and numpy to determine all the parameters.
# 


step=1
parser=argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="traj name in pdb format")
parser.add_argument("-d","--directory",help="Directory folder")
arg=parser.parse_args()

dir=str(arg.directory)



with open(dir+"_new_250ns/"+str(arg.filename),"r") as file:
    filedata=file.read()
    
filedata=filedata.replace("*","'")

with open(dir+"_new_250ns/"+str(arg.filename), 'w') as file:
  file.write(filedata)
del filedata
pna_traj=pt.load(dir+"_new_250ns/"+str(arg.filename),stride=1)

pna_traj=pt.autoimage(pna_traj)






parejas=int(len(Bases)/2)
NB=len(Bases)
lim_sup=NB+3
T_bp_R=np.zeros((pna_traj.n_frames,parejas,3,3))
T_b_R=np.zeros((pna_traj.n_frames,parejas,3,3))

r_bp,T_bp,r_b,T_b=Triads(pna_traj,step,NB,Bases,lim_sup,parejas)





omega,SSR,twist_g=hel_parms(r_bp,T_bp,r_b,T_b,pna_traj,step,NB,Bases,lim_sup,parejas)
        


# Ya tengo todo calculado, ahora puedo ir analizando los parámetros importantes.




prom=np.zeros(pna_traj.n_frames)
prom_g=np.zeros(pna_traj.n_frames)
for k in range(0,pna_traj.n_frames):
    prom[k]=sum(omega[k][:5])/5
    prom_g[k]=sum(twist_g[k][:5])/5


np.savetxt(dir+"_new_250ns/heltwist_"+str(arg.filename)+".txt",(prom_g))

np.savetxt(dir+"_new_250ns/heltwist_local"+str(arg.filename)+".txt",(prom))
#plt.hist(prom,20)


"""Distribución de rise"""

prom_R=np.zeros(pna_traj.n_frames)

for k in range(0,pna_traj.n_frames):
    prom_R[k]=sum(np.abs(SSR[k][:5,2]))/5

np.savetxt(dir+"_new_250ns/Rise"+str(arg.filename)+".txt",prom_R)

