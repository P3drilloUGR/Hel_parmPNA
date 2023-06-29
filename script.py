#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pytraj as pt
import subprocess
import matplotlib.pyplot as plt 
"""Script para paralelizar el análisis de los parámetros helicoidales"""
dir="./rCAG-pCGG_amp_250ns/"
traj=pt.load(dir+"hel_parm.pdb")

"""División de las trayectorias en el número de procesos que se desee"""
cores=30

for i in range(0,cores):
    
    if i==0:
        slice=traj.iterframe(1,(i+1)*traj.n_frames//cores,1)
        
        pt.write_traj(dir+"traj"+str(i)+".pdb",traj=slice,options="model",overwrite=True)
    else:
        slice=traj.iterframe((i)*traj.n_frames//cores,(i+1)*traj.n_frames//cores,1)
        
        pt.write_traj(dir+"traj"+str(i)+".pdb",traj=slice,options="model",overwrite=True)
del traj
del slice


# In[82]:


procceses=[]
for i in range(0,cores):
    procces=subprocess.Popen("python Helical_parm_r-p.py -f traj"+str(i)+".pdb -d rCAG-pCGG  &",shell=True)
    procceses.append(procces)    


# In[2]:


#get_ipython().system('rm ./rCAG-pCTG_amp_250ns/traj*')
#get_ipython().system('cat ./rCAG-pCTG_amp_250ns/Risetraj{0..23}.pdb.txt > ./rCAG-pCTG_amp_250ns/Rise.txt')
#get_ipython().system('cat ./rCAG-pCTG_amp_250ns/heltwist_localtraj{0..23}.pdb.txt > ./rCAG-pCTG_amp_250ns/heltwist_local.txt')


# In[3]:


#get_ipython().system('rm ./rCAG-pCTG_amp_250ns/Risetraj*')
#get_ipython().system('rm ./rCAG-pCTG_amp_250ns/heltwisttraj*')
#get_ipython().system('rm ./rCAG-pCTG_amp_250ns/heltwist_localtraj*')


# 
