import numpy as np

def bdry_int_sig(v1,v2,wn):
    """Returns the boundary interpolation signature between v1 and v2, wn is the wolff normal"""
    return myangleorient(wn,v1) if v1.dot(wn)**2 < v2.dot(wn)**2 else myangleorient(wn,v2)

def myangleorient(v1,v2):
    return 1 if v1[0]*v2[1] > v1[1]*v2[0] else -1



def plaquette_signature(v1,v2,v3,v4, wn):
    return (
        bdry_int_sig(v1,v2,wn),
        bdry_int_sig(v2,v3,wn),
        bdry_int_sig(v3,v4,wn),
        bdry_int_sig(v4,v1,wn),
    )


def get_plaqsig_at(dofs, wn ,x,y):
    sx, sy = dofs.shape[:2]
    v1 = dofs[x,y]
    v2 = dofs[(x+1)%sx,y]
    v3 = dofs[(x+1)%sx, (y+1)%sy]
    v4 = dofs[x, (y+1)%sy]
    return plaquette_signature(v1,v2,v3,v4,wn)

def get_halfvortex_locations(dofs, wn, isinc):
    sx,sy = dofs.shape[:2]
    hvloc=[]
    hvploc=[]
    hvmloc=[]
    for x in range(sx):
        for y in range(sy):
            psig=np.array(get_plaqsig_at(dofs, wn ,x,y))
            sites =[(x,y) , ((x+1)%sx, y), ((x+1)%sx, (y+1)%sy), (x, (y+1)%sy)]
            j=0;
            psigprev = psig[3]
            vorts = np.zeros(4)
            for i in range(4):
                psigcurrent = psig[i]
                if psigprev == 1 and psigcurrent ==-1:
                    vorts[i]=-0.5
                    sitem = sites[i]
                if psigprev == -1 and psigcurrent ==1:
                    vorts[i]=0.5
                    sitep=sites[i]
                    
                psigprev = psigcurrent
            if 0.5 in vorts and isinc[sitep] != isinc[sitem]:
                plaqcenter = np.array([x+0.5,y+0.5])
                hvloc.append(plaqcenter)
                hvploc.append((0.7*plaqcenter+0.3*np.array(sitep)))
                hvmloc.append((0.7*plaqcenter+0.3*np.array(sitem)))
    return np.array(hvloc), np.array(hvploc), np.array(hvmloc)


def get_halfvortex_clusters(dofs, wn, isinc):
    sx,sy = dofs.shape[:2]
    hvcls=[]
    ahvcls=[]

    for x in range(sx):
        for y in range(sy):
            psig=np.array(get_plaqsig_at(dofs, wn ,x,y))
            sites =[(x,y) , ((x+1)%sx, y), ((x+1)%sx, (y+1)%sy), (x, (y+1)%sy)]
            j=0;
            psigprev = psig[3]
            vorts = np.zeros(4)
            for i in range(4):
                psigcurrent = psig[i]
                if psigprev == 1 and psigcurrent ==-1:
                    vorts[i]=-0.5
                    sitem = sites[i]
                if psigprev == -1 and psigcurrent ==1:
                    vorts[i]=0.5
                    sitep=sites[i]

                psigprev = psigcurrent
            if 0.5 in vorts and isinc[sitep] != isinc[sitem]:
                hvcls.append(isinc[sitep])
                ahvcls.append(isinc[sitem])

    return np.array(hvcls), np.array(ahvcls)

