""" A collection of functions related to vortices in the 2d O(2) model, many of which are compiled with numba."""


from itertools import compress
from numba import njit
import numpy as np
import o2pytools.pylab as pylab


def orientedcover_vec(v1,v2, w):
    alpha_0 = np.sign(v1[:,:,0]*v2[:,:,1] - v1[:,:,1]*v2[:,:,0]);
    alpha_1 = np.sign(v1[:,:,0]*w[1]  - v1[:,:,1]*w[0]);
    alpha_2 = np.sign( w[0]*v2[:,:,1] - w[1]*v2[:,:,0]);
    return alpha_0 * ((alpha_0 == alpha_1) & (alpha_0 == alpha_2) )

def plaquettevorticity_vec(dofs, w=None):
    if w is None:
        w=random_unit_vector()
    v1 = dofs
    v2 = np.roll(dofs, 1, axis=0)
    v3 = np.roll(np.roll(dofs, 1, axis=0), 1,axis=1)
    v4 = np.roll(dofs, 1, axis=1)
    return orientedcover_vec(v1,v2,w)+orientedcover_vec(v2,v3,w)+orientedcover_vec(v3,v4,w)+orientedcover_vec(v4,v1,w)




@njit
def get_plaquette_cs(hookcoords, sx,sy):
    """Returns the coordinates of all site that belong to the plaquette with hook hookcoords"""
    x,y=hookcoords
    plaquettecs = [(x,y), ((x-1)%sx,y), ((x-1)%sx,(y-1)%sy), (x,(y-1)%sy)]
    return plaquettecs 

@njit
def flip_to_refconf(vs, wn):
    """Flip all vectors in vs to the reference configuration (such that $$v \cdot w >0 \forall v \in vs $$)"""
    return [flip(v, wn) if v.dot(wn) <= 0 else v for v in vs]

@njit
def get_signature(vs, wn):
    """Returns the signature of the vo"""
    rc = flip_to_refconf(vs, wn)
    flippedvs = [rc[:i]+[flip(rc[i],wn)]+rc[i+1:] for i in range(len(rc))]
    signature = np.array([plaq_vort(conf) for conf in flippedvs])
    return signature

@njit
def flip(v, wn):
    """flip v with respect to the plane normal to wn"""
    return v-2*v.dot(wn)*wn

@njit
def oc(v1,v2,w):
    """Oriented cover. Returns \pm 1 if w is coverd by the arc from v1 to v2 else 0. Positive if the arc goes counterclockwise, negative if clockwise"""
    alpha_0 = np.sign(v1[0]*v2[1] - v1[1]*v2[0]);
    alpha_1 = np.sign(v1[0]*w[1]  - v1[1]*w[0]);
    alpha_2 = np.sign( w[0]*v2[1] - w[1]*v2[0]);
    return alpha_0 * ((alpha_0 == alpha_1) & (alpha_0 == alpha_2) )

@njit
def random_unit_vector():
    """ a random 2d unit vector"""
    r = np.random.randn(2)
    return r/np.sqrt(np.sum(r**2))


@njit
def plaq_vort(vs):
    """calculates the vorticity of the plaquette containing the vectors listed in vs."""
    r = random_unit_vector()
    occum = 0
    for i in range(len(vs)):
        occum += oc(vs[i], vs[(i+1)%len(vs)], r)
    return occum

@njit
def get_associated_clusters(coords, dofs, isinc, wn):
    """get the clusterids of the clusters associated to the vortex at coords"""
    pcoords = get_plaquette_cs(coords, *isinc.shape)
    vs = [dofs[c] for c in pcoords]
    
    sig = get_signature(vs, wn)
    asc=[]
    for i in range(4):
        if sig[i]!=0:
            asc.append(isinc[pcoords[i]])
    return asc

def free_vn(dofs, isinc, wn):
    pv = pylab.plaquettevorticity_vec(dofs)
    vloc = list(zip(*np.where(pv!=0)))
    clsofvorts = [get_associated_clusters(vc, dofs, isinc, wn) for vc in vloc]
    clswithvorts, nvorts = np.unique(np.array(clsofvorts), return_counts=True)
    return np.sum(nvorts%2==1)/2.
    
def completely_free_vortices(dofs, isinc, wn):
    pv = pylab.plaquettevorticity_vec(dofs)
    vloc = list(zip(*np.where(pv!=0)))
    clsofvorts = [get_associated_clusters(vc, dofs, isinc, wn) for vc in vloc]
    clswithvorts, nvorts = np.unique(np.array(clsofvorts), return_counts=True)
    return np.sum([nvorts[clswithvorts==cl1] == 1 and  nvorts[clswithvorts==cl2]== 1 for cl1,cl2 in clsofvorts] )   
 
def completely_free_vortex_locations(dofs, isinc, wn):
    pv = pylab.plaquettevorticity_vec(dofs)
    vloc = list(zip(*np.where(pv!=0)))
    clsofvorts = [get_associated_clusters(vc, dofs, isinc, wn) for vc in vloc]
    clswithvorts, nvorts = np.unique(np.array(clsofvorts), return_counts=True)
    vortisfree = [nvorts[clswithvorts==cl1] == 1 and  nvorts[clswithvorts==cl2]== 1 for cl1,cl2 in clsofvorts]    
    return  list(compress(vloc, vortisfree))
