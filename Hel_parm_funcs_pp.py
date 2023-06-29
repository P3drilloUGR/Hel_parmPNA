import numpy as np
import pytraj as pt 
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from numba import jit,vectorize,prange

@jit(nopython=True)
def min_normal(normal, points): #Funciona bastante bien
    
    distance=np.abs(np.dot(points,normal))
    
    return np.sum(distance**2)


def Triads(pna_traj,step,NB,Bases,lim_sup,parejas):
    n_frames=pna_traj.n_frames
    r_bp=np.zeros((pna_traj.n_frames,parejas,3))
    r_b=np.zeros((pna_traj.n_frames,NB,3))
    T_bp=np.zeros((pna_traj.n_frames,parejas,3,3))
    T_b=np.zeros((pna_traj.n_frames,NB,3,3))


    for k in prange(0,n_frames,step):
        for i in range(1,NB//2+1): 

            traj_bp=pt.strip(pna_traj,"!(:"+str(i)+",:"+str(lim_sup-i)+")")
            traj_bp=pt.strip(traj_bp,"@/H,@C1',P,O3',O5',C3',C4',C5',C2',OP1,OP2,O4',O2',@C7',O7',C3',N4',C',O1',C5',C2',N1',C8',C2'H,1'H,C3'H,1'")

        
            """Selección de tipo de base"""

            if((Bases[i-1]=="Y") and (Bases[NB-i]=="Y")):
                C8_C6=pt.strip(traj_bp,"!(:1@C6,:2@C6)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_bp,[":1@C6 :2@C6"])
                index_C8C6=pt.select_atoms(":1@C6 :2@C6",traj_bp.top)
                
            elif(Bases[i-1]=="Y") and (Bases[NB-i]=="R"):
                
                C8_C6=pt.strip(traj_bp,"!(:1@C6,:2@C8)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_bp,[":1@C6 :2@C8"])
                index_C8C6=pt.select_atoms(":1@C6 :2@C8",traj_bp.top)

            elif(Bases[i-1]=="R" and Bases[NB-i]=="Y"):
                C8_C6=pt.strip(traj_bp,"!(:1@C8,:2@C6)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_bp,[":1@C8 :2@C6"])
                index_C8C6=pt.select_atoms(":1@C8 :2@C6",traj_bp.top)
            
            elif(Bases[i-1]=="R" and Bases[NB-i]=="R"):
                C8_C6=pt.strip(traj_bp,"!(:1@C8,:2@C8)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_bp,[":1@C8 :2@C8"])
                index_C8C6=pt.select_atoms(":1@C8 :2@C8",traj_bp.top)
            else:
                continue


            Originbp=(C8_C6[k][0]+C8_C6[k][1])/2
            coordinates=traj_bp.xyz
            #view=ng.show_pytraj(traj_bp)   
            #view                            
            result=minimize(min_normal,(1,1,1),args=coordinates[k]-Originbp)
            z_0=result.x


        

            y_0=(C8_C6[k][0]-C8_C6[k][1])
            z_0=z_0/np.linalg.norm(z_0)
            y_0=y_0/np.linalg.norm(y_0)
            

            zy=np.dot(z_0,y_0)
            angzy=np.arccos(zy)
            z_0=z_0*np.sin(angzy)
            
            x_0=np.cross(y_0,z_0)
            
            
            """Guardo los valores de origin en un vector de posición r, para cada base pair o cada base"""
            r_bp[k][i-1]=Originbp
            T=np.asmatrix([x_0,y_0,z_0])
            T_bp[k][i-1]=np.transpose(T)
            
        "Ahora calculo los parámetros de las bases "
        
        #traj_b=pt.strip(pna_traj,":ACE,:NME")
        for j in range(1,NB+1):
            traj_b=pt.strip(pna_traj,":ACE,:NME")
            traj_b=pt.strip(traj_b,"!(:"+str(j)+")")
            traj_b=pt.strip(traj_b,"@/H,@C1',P,O3',O5',C3',C4',C5',C2',OP1,OP2,O4',O2',@C7',O7',C3',N4',C',O1',C5',C2',N1',C8',C2'H,1'H,C3'H,1'")
            #view=ng.show_pytraj(traj_b)
            #view
            
            if(Bases[j-1]=="Y"):
                N1_C4=pt.strip(traj_b,"!(:1@N3,:1@C6)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_b,[":1@N3 :1@C6"])
                
                Originb=(N1_C4[k][0]+N1_C4[k][1])/2

                y_0b=(N1_C4[k][0]-N1_C4[k][1])    
                
            elif(Bases[j-1]=="R"):
                    
                N1_C6=pt.strip(traj_b,"!(:1@N1,C6)").xyz #Coordenas de C8 y C6 para escoger el eje
                dist=pt.distance(traj_b,[":1@N1 :1@C6"])
                
                Originb=(N1_C6[k][0]+N1_C6[k][1])/2
                
                y_0b=(N1_C6[k][0]-N1_C6[k][1])
            
            else:
                continue
            coordinates=traj_bp.xyz
            #view=ng.show_pytraj(traj_bp)   
            #view                            
            result=minimize(min_normal,(1,1,1),args=coordinates[k]-Originb)
            z_0b=result.x


            

            
            z_0b=z_0b/np.linalg.norm(z_0b)
            y_0b=y_0b/np.linalg.norm(y_0b)
                

            zy=np.dot(z_0b,y_0b)
            angzy=np.arccos(zy)
            z_0b=z_0b*np.sin(angzy)
            
            x_0b=np.cross(y_0b,z_0b)

            r_b[k][j-1]=Originb
            T=np.asmatrix([x_0b,y_0b,z_0b])
            T_b[k][j-1]=np.transpose(T)
    return r_bp,T_bp,r_b,T_b


def hel_parms(r_bp,T_bp,r_b,T_b,pna_traj,step,NB,Bases,lim_sup,parejas):

    RollTilt_ang=np.zeros((pna_traj.n_frames,parejas,1))
    rt=np.zeros((pna_traj.n_frames,parejas,3))
    T_bp_Ri=np.zeros((pna_traj.n_frames,parejas,3,3))
    T_bp_Ri1=np.zeros((pna_traj.n_frames,parejas,3,3))
    T_mst=np.zeros((pna_traj.n_frames,parejas,3,3))
    r_mst=np.zeros((pna_traj.n_frames,parejas,3))

    T_b_Ri=np.zeros((pna_traj.n_frames,NB//2,3,3))
    T_b_Rii=np.zeros((pna_traj.n_frames,NB//2,3,3))
    T_mbt=np.zeros((pna_traj.n_frames,NB//2,3,3))
    r_mbt=np.zeros((pna_traj.n_frames,NB//2,3))
    T_i=np.zeros((pna_traj.n_frames,parejas,3,3))
    T_i1=np.zeros((pna_traj.n_frames,parejas,3,3))

    """Base-step"""
    omega=np.zeros((pna_traj.n_frames,parejas,1))
    phi=np.zeros((pna_traj.n_frames,parejas,1))
    #roll=np.zeros((pna_traj.n_frames,parejas,1))
    #tilt=np.zeros((pna_traj.n_frames,parejas,1))
    SSR=np.zeros((pna_traj.n_frames,parejas,3))

    """Base-pair"""
    BuOp_ang=np.zeros((pna_traj.n_frames,NB//2,1))
    BuOp=np.zeros((pna_traj.n_frames,NB//2,3))
    #propeller=np.zeros((pna_traj.n_frames,NB//2,1))
    #phi_b=np.zeros((pna_traj.n_frames,NB//2,1))
    #buckle=np.zeros((pna_traj.n_frames,NB//2,1))
    #opening=np.zeros((pna_traj.n_frames,NB//2,1))
    #SSS=np.zeros((pna_traj.n_frames,NB//2,3))

    """Global parameters"""
    z_glob=np.zeros((pna_traj.n_frames,parejas,3))
    nabla=np.zeros((pna_traj.n_frames,parejas,1))
    ti=np.zeros((pna_traj.n_frames,parejas,3))
    phi_g=np.zeros((pna_traj.n_frames,parejas,1))
    tip=np.zeros((pna_traj.n_frames,parejas,1))
    inclination=np.zeros((pna_traj.n_frames,parejas,1))
    twist_g=np.zeros((pna_traj.n_frames,parejas,1))
    ddd=np.zeros((pna_traj.n_frames,parejas,3))
    """En esta parte se realiza el cálculo de los base-step parameters"""


    for k in range(0,pna_traj.n_frames):
        for i in range(0,parejas-1):
            RollTilt_ang[k][i]=np.arccos(np.dot(T_bp[k][i][:,2],T_bp[k][i+1][:,2]))
            rt[k][i]=np.cross(T_bp[k][i][:,2],T_bp[k][i+1][:,2])
            rt[k][i]=rt[k][i]/np.linalg.norm(rt[k][i])

            Rot_i=R.from_rotvec(rt[k][i]*(RollTilt_ang[k][i]/2))
            T_bp_Ri[k][i]=Rot_i.apply(T_bp[k][i])
            Rot_i1=R.from_rotvec(rt[k][i]*(-1*RollTilt_ang[k][i]/2))
            T_bp_Ri1[k][i]=Rot_i1.apply(T_bp[k][i+1])

            T_mst[k][i]=(T_bp_Ri[k][i]+T_bp_Ri1[k][i])/2 #No se si está normalizada
            
            T_mst[k][i]=np.asmatrix([T_mst[k][i][0,:]/np.linalg.norm(T_mst[k][i][0,:]),T_mst[k][i][1,:]/np.linalg.norm(T_mst[k][i][1,:]),T_mst[k][i][2,:]/np.linalg.norm(T_mst[k][i][2,:])])
        
            #r_mst[k][i]=(r_bp[k][i]+r_bp[k][i+1])/2
            #r_mst[k][i]=r_mst[k][i]/np.linalg.norm(r_mst[k][i])

            """Cálculo del local-twist"""

            if (np.dot(np.cross(T_bp_Ri[k][i][:,1],T_bp_Ri1[k][i][:,1]),T_mst[k][i][:,2]))>0:
                omega[k][i]=np.degrees(np.arccos(np.dot(T_bp_Ri[k][i][:,1],T_bp_Ri1[k][i][:,1]))) 
            else:
                omega[k][i]=np.degrees(np.arccos(np.dot(T_bp_Ri[k][i][:,1],T_bp_Ri1[k][i][:,1]))) 
            """Cálculo del ángulo entre el Roll-tilt axis y el mst y-axis"""

            #if (np.dot(np.cross(rt[k][i],T_mst[k][i][:,1]),T_mst[k][i][:,2]))>0:
            #    phi[k][i]=(np.arccos(np.dot(rt[k][i],T_mst[k][i][:,1])))
            #else:
            #    phi[k][i]=-(np.arccos(np.dot(rt[k][i],T_mst[k][i][:,1])))
            """Cálculo roll and tilt"""

            #roll[k][i]=np.degrees(RollTilt_ang[k][i]*np.cos(phi[k][i]))
            #tilt[k][i]=np.degrees(RollTilt_ang[k][i]*np.sin(phi[k][i]))     #Esto funciona perfecto, es simple

            """Cálculo shift,slide,rise"""

            SSR[k][i]=np.dot((r_bp[k][i+1]-r_bp[k][i]),T_mst[k][i]) #Resultado raro, mirarlo

        """Pasamos al cálculo de los base-pair parameters"""

        for i in range(0,NB//2):
            """Buckle-opening angle"""

            #BuOp_ang[k][i]=np.arccos(np.dot(T_b[k][NB-1-i][:,1],T_b[k][i][:,1]))

            """Bucle-opening axis"""

            #BuOp[k][i]=np.cross(T_b[k][NB-1-i][:,1],T_b[k][i][:,1])
            #BuOp[k][i]=BuOp[k][i]/np.linalg.norm(BuOp[k][i])
            
            """Rotación triada base-pair"""

            #Rot_i=R.from_rotvec(BuOp[k][i]*(BuOp_ang[k][i]/2))
            #T_b_Ri[k][i]=Rot_i.apply(T_b[k][i])
            #Rot_ii=R.from_rotvec(BuOp[k][i]*(-1*BuOp_ang[k][i]/2))
            #T_b_Rii[k][i]=Rot_ii.apply(T_b[k][13-i])

            """Cálculo mbt"""

            #T_mbt[k][i]=(T_b_Ri[k][i]+T_b_Rii[k][i])/2  #Lo mismo, mirar si está normalizado

            #T_mbt[k][i]=np.asmatrix([T_mbt[k][i][0,:]/np.linalg.norm(T_mbt[k][i][0,:]),T_mbt[k][i][1,:]/np.linalg.norm(T_mbt[k][i][1,:]),T_mbt[k][i][2,:]/np.linalg.norm(T_mbt[k][i][2,:])])
            #r_mbt[k][i]=(r_b[k][i]+r_b[k][NB-1-i])/2

            #r_mbt[k][i]=r_mbt[k][i]/np.linalg.norm(r_mbt[k][i])
            """Propeller"""
            #if(np.dot(np.cross(T_b_Rii[k][i][:,0],T_b_Ri[k][i][:,0]),T_mbt[k][i][:,1]))>0:
            #    propeller[k][i]=np.arccos(np.dot(T_b_Rii[k][i][:,0],T_b_Ri[k][i][:,0])) #pONER CONVENIO DE SIGNOS
            #else:
            #    propeller[k][i]=-propeller[k][i]
            """Angle between transformed x axis and MBT x axis"""
            #if(np.dot(np.cross(BuOp[k][i],T_mbt[k][i][:,0]),T_mbt[k][i][:,1]))>0:
            #    phi_b[k][i]=np.arccos(np.dot(BuOp[k][i],T_mbt[k][i][:,0]))  #Convenio de signos
            #else:
            #    phi_b[k][i]=-phi_b[k][i]
            """Buckle and opening"""

            #buckle[k][i]=BuOp_ang[k][i]*np.cos(phi_b[k][i])
            #opening[k][i]=BuOp_ang[k][i]*np.sin(phi_b[k][i])

            """Displacement"""

            #SSS[k][i]=np.dot((r_b[k][i]-r_b[k][NB-1-i]),T_mbt[k][i])

            """Para el cálculo de los parámetros globales, voy a usar la definición de 3DNA para el helical axis (Mas simple y equivalente https://doi.org/10.1093/nar/gkg680)"""

        for i in range(0,parejas-1):

            """Z global"""

            z_glob[k][i]=np.cross((T_bp[k][i][:,0]-T_bp[k][i+1][:,0]),(T_bp[k][i][:,1]-T_bp[k][i+1][:,1]))
            z_glob[k][i]=z_glob[k][i]/np.linalg.norm(z_glob[k][i])

            nabla[k][i]=np.arccos(np.dot(z_glob[k][i],T_bp[k][i][:,2]))

            """Tip-inclination axis"""

            ti[k][i]=np.cross(z_glob[k][i],T_bp[k][i][:,2])
            ti[k][i]=ti[k][i]/np.linalg.norm(ti[k][i])

            """Rotations"""

            Rot=R.from_rotvec(-1*ti[k][i]*nabla[k][i])
            T_i[k][i]=Rot.apply(T_bp[k][i])
            T_i1[k][i]=Rot.apply(T_bp[k][i+1])

            """Angle between tip-inclination axis and y rotated"""
            #if(np.dot(np.cross(ti[k][i],T_i[k][i][:,1]),z_glob[k][i]))>0:
            #    phi_g[k][i]=np.arccos(np.dot(ti[k][i],T_i[k][i][:,1]))
            #else:
            #    phi_g[k][i]=-phi_g[k][i]

            """Tip and inclination"""

            #tip[k][i]=nabla[k][i]*np.cos(phi_g[k][i])
            #inclination[k][i]=nabla[k][i]*np.sin(phi_g[k][i])

            """Global twist"""
            if(np.dot(np.cross(T_i[k][i][:,1],T_i1[k][i][:,1]),z_glob[k][i]))>0:

                twist_g[k][i]=np.degrees(np.arccos(np.dot(T_i[k][i][:,1],T_i1[k][i][:,1])))
            else:
                twist_g[k][i]=np.degrees(-np.arccos(np.dot(T_i[k][i][:,1],T_i1[k][i][:,1])))
            """Displacements"""

    return omega,SSR,twist_g