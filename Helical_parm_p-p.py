#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pytraj as pt
from Hel_parm_funcs_pp import Triads,hel_parms
import matplotlib.pyplot as plt
import argparse

from scipy.spatial.transform import Rotation as R
def min_normal(normal, points): #Funciona bastante bien
    distance=np.abs(np.dot(points,normal))

    return np.sum(distance**2)
    
                                        #                                                                                   #      
#Bases=["R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y","R","R","Y","Y","R","Y","Y","R","Y"] #Poner el tipo de base, pirimidica (Y) o puridica (R)(ej. A/G puridica, C,U,T pirimídica), esto influye en la triada de referencia
#Bases=["R","Y","Y","R","Y","Y","R","Y","Y","R","Y","Y","R","Y"] #Probar las bases pequeñas
Bases=["Y","R","Y","Y","Y","R","Y","Y"]
parser=argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="traj name in pdb format")
parser.add_argument("-d","--directory",help="Directory folder")
arg=parser.parse_args()
# We are going to analyze the helical parameters of a generic nucleic acid, following the instructions given by: https://doi.org/10.1006/jmbi.1997.1346.
# For this project we will use pytraj, scipy and numpy to determine all the parameters.
# 

# In[9]:


dir=str(arg.directory)

"""ARREGLAR PARA PNA-PNA, NO SE PORQUÉ NO VA"""

with open(dir+"_new_250ns/efecto_borde/"+str(arg.filename),"r") as file:
    filedata=file.read()
    
filedata=filedata.replace("*","'")

with open(dir+"_new_250ns/efecto_borde/"+str(arg.filename), 'w') as file:
  file.write(filedata)

pna_traj=pt.load(dir+"_new_250ns/efecto_borde/"+str(arg.filename),stride=1)
del filedata
pna_traj=pt.autoimage(pna_traj)
# First we load the trajectory

# Then strip the solvent and ions (for other solvent or ions simply change the name in strip(":solvent:ion")).
# 
# Then for the calculatios we have to define some references. First we start with the base-pair triad.
# 
# 

# In[10]:


"""Voy a correr un bucle que me haga la selección de parejas de bases y me calcule su correspondiente triada de vectores"""
parejas=int(len(Bases)/2)
NB=len(Bases)
lim_sup=NB+1
step=1
T_bp_R=np.zeros((pna_traj.n_frames,parejas,3,3))
T_b_R=np.zeros((pna_traj.n_frames,parejas,3,3))

r_bp,T_bp,r_b,T_b=Triads(pna_traj,step,NB,Bases,lim_sup,parejas)

        


omega,SSR,twist_g=hel_parms(r_bp,T_bp,r_b,T_b,pna_traj,step,NB,Bases,lim_sup,parejas)

# Ya tengo todo calculado, ahora puedo ir analizando los parámetros importantes.

prom=np.zeros(pna_traj.n_frames)
prom_g=np.zeros(pna_traj.n_frames)
for k in range(0,pna_traj.n_frames):
    prom[k]=sum(omega[k][:3])/3
    prom_g[k]=sum(twist_g[k][:3])/3


np.savetxt(dir+"_new_250ns/efecto_borde/heltwist_"+str(arg.filename)+".txt",(prom_g))

np.savetxt(dir+"_new_250ns/efecto_borde/heltwist_local"+str(arg.filename)+".txt",(prom))
#plt.hist(prom,20)


"""Distribución de rise"""

prom_R=np.zeros(pna_traj.n_frames)

for k in range(0,pna_traj.n_frames):
    prom_R[k]=sum(np.abs(SSR[k][:3,2]))/3

np.savetxt(dir+"_new_250ns/efecto_borde/Rise"+str(arg.filename)+".txt",prom_R)





