"""The Wolff algorithm for the 2d O(2) model in python (mostly) compiled with numba.

Examples:

    To create a 100x100 system with a random state and perform 100 multicluster updates at beta=1.12 one could do:
    ```python
    dofs = random_dofs(100,100)
    mc_update(dofs, 1.12,nsteps=100)
    ```
    Or more verbousely:
    ```python
    dofs = random_dofs(100,100)
    for i in range(100):
        wn = random_unit_vector()
        ph,pv = p_add_bond(dofs, wn,beta)
        bondsh, bondsv = set_bonds(ph,pv) 
        isinc = assign_clusters_from_bonds(bondsh,bondsv)
        flip_clusters_5050(dofs, isinc, wn)
    ```
    The functions can also be used to simulate a two d bond percolation model, here weith bond occupation probability 0.5:
    ```python
    ps = numpy.full( (100,100), 0.5)
    bondsh, bondsv = set_bonds(ps,ps)
    isinc = assign_clusters_from_bonds(bondsh,bondsv)
    ```
"""

from numba import jit
import numba
import numpy as np

@jit(nopython=True)
def random_unit_vector():
    """
    Returns a random 2d unit vector.
    """
    r = np.random.randn(2)
    return r/np.sqrt(np.sum(r**2))

def random_dofs(sx,sy):
    """ Creates a grid of random spins, to be used e.g. for initial conditions.
    Args:
        sx (int): the extent of the system in 1 direction
        sy (int): the extents of the system in 2 direction
    Returns:
        dofs (array): a numpy array of shape [sx,sy,2] containing random 2d unit vectors
    """
    return np.array([[random_unit_vector() for y in range(sy)] for x in range(sx)])

@jit(nopython=True)
def flip_dofs(dofs, wolffnormal):
    """
    Returns dofs flipped at the plane normal to wolffnormal.

    Calculates:
     $$ s_{flipped} = s-2(s \cdot w ) w $$

    This function assumes that dofs is 3d dimensional with the last dimension being of extent 2.
    In order to be easily compilable, there are separate functions for a 1d array of spins
    ( flip_dofs_1d ) and a single spin (flip_dof_0d).

    Args:
        dofs ([: ,:,2] float array): the usual spin array containing the state of the 2d O(2) model.
        wolffnormal ([2] float array): the vector normal to the wolff polane

    Returns:
        dofsf ( flip_dofs_1d ): the flipped degrees of freedom
    """
    dofsf = dofs.copy()
    for i in range(2):
        dofsf[:,:,i] -= 2*(dofs[:,:,0]*wolffnormal[0] + dofs[:,:,1]*wolffnormal[1] )* wolffnormal[i]
        return dofsf 

@jit(nopython = True)
def flip_dofs_1d(dofs, wolffnormal):
    """
    Returns dofs flipped at the plane normal to wolffnormal.

    Calculates:
    $$ s_{flipped} = s-2(s \cdot w ) w $$

    This function assumes that dofs is 2 dimensional with the last dimension being of extent 2.
    In order to be easily compilable, there are separate functions for a 2d array of spins
    ( flip_dofs ) and a single spin (flip_dof_0d).

    Args:
        dofs ([:,2] float array): the usual spin array containing the state of the 2d O(2) model.
        wolffnormal ([2] float array): the vector normal to the wolff polane

    Returns:
        dofsf ( flip_dofs_1d ): the flipped degrees of freedom
    """

    dofsf = dofs.copy()
    for i in range(2):
        dofsf[:,i] -= 2*(dofs[:,0]*wolffnormal[0] + dofs[:,1]*wolffnormal[1] )* wolffnormal[i]
    return dofsf 

@jit(nopython = True)
def flip_dof_0d(dof, wolffnormal):
    """Return the reflection of dof at the plane normal to wolffnormal"""
    doff= dof.copy()
    for i in range(2):
        doff[i] -= 2*(dof[0]*wolffnormal[0] + dof[1]*wolffnormal[1] )* wolffnormal[i]
    return doff 



@jit(nopython = True)
def dofdot(dofs, dofsp,shift=0, shiftaxis=0):
    """Calculates the dot product elementwise between dofs and dofsp, dofsp is shifted.

    This function calculates the elementwise shifted dot product of the last dimension of the arrays dofs, dofsp.

    $$ r_{ij} = s^{(1)}_{ijk}s^{(2)}_{ijk} $$
    if shift is not equal 0 and shiftaxis is 0:
    $$ r_{ij} = s^{(1)}_{ijk}s^{(2)}_{(i+shift)jk} $$
    periodicity is assumed.
    """
    shape = dofs.shape[:2]
    sx,sy=shape
    result=np.zeros(shape)
    if shift==0:
        for i in range(sx):
            for j in range(sy):
                result[i,j] = dofs[i,j,0] * dofsp[i,j,0]+ dofs[i,j,1] * dofsp[i,j,1]
        
    if shiftaxis ==0 :
        for i in range(sx):
            for j in range(sy):
                result[i,j] = dofs[(i),j,0] * dofsp[(i+shift)%sx,j,0]+ dofs[i,j,1] * dofsp[(i+shift)%sx,j,1]
 
    if shiftaxis ==1 :
        for i in range(sx):
            for j in range(sy):
                result[i,j] = dofs[(i),j,0] * dofsp[(i),(j+shift)%sy,0]+ dofs[i,j,1] * dofsp[i,(j+shift)%sy,1]

    return result

@jit(nopython = True)
def p_add_bond(dofs, wn, beta):
    """
    Calculates the probability for each bond to be activated. 

    This function sets the bond probability to assign the wolff clusters in the 2d-O(2) model on a rectangular lattice
    with periodic boundary conditions. 
    $$ p_{<xy>}= 1-\exp(\min(0, -2 \\beta s_x \cdot w s_y \cdot w)) $$

    Args:
        dofs ([sx,sy,2] float array): The spin state of the system.
        wn ([2] float array): The vector normal to the wolffplane.
        beta (float) : The inverse temperature of the system.

    Returns:
        ph ([sx,sy] float array): The array containing th bond probabilties as ph[x_1,x_2] = p_{<(x_1,x_2)(x_1+1, x_2)>} 
        pv ([sx,sy] float array): The array containing th bond probabilties as ph[x_1,x_2] = p_{<(x_1,x_2)(x_1, x_2+1)>} 
    """
    sx = dofs.shape[0]
    sy = dofs.shape[1]
    ph = np.zeros(dofs.shape[:2], dtype=numba.float64)
    pv = np.zeros(dofs.shape[:2], dtype=numba.float64)
    for x in range(sx):
        for y in range(sy):
            dofdotwn = dofs[x,y,0]*wn[0]+dofs[x,y,1]*wn[1]
            dofdotwnh = dofs[(x+1)%sx,y,0]*wn[0]+dofs[(x+1)%sx,y,1]*wn[1]
            dofdotwnv = dofs[x,(y+1)%sy,0]*wn[0]+dofs[x,(y+1)%sy,1]*wn[1]
            zeroa = np.zeros_like(dofdotwnv)
            ph[x,y] = 1.-np.exp(np.minimum(zeroa, -2*beta*dofdotwn*dofdotwnh))
            pv[x,y] = 1.-np.exp(np.minimum(zeroa, -2*beta*dofdotwn*dofdotwnv))
    return ph,pv

@jit(nopython = True)
def set_bonds(ph, pv):
    """
    Sets each bond according to the probabilties specified in ph,pv.

    Args:
        ph ([sx,sy] float array): The probabilities to set the horizontal bonds, as calculated with p_add_bond.
        pv ([sx,sy] float array): The probabilities to set the vertical bonds, as calculated with p_add_bond.
    Returns:
        bondsh ([sx,sy] bool array): bondsh[x_1, x_2] is true if the bond <(x_1,x_2),(x_1+1,x_2)> is set
        bondsv ([sx,sy] bool array): bondsv[x_1, x_2] is true if the bond <(x_1,x_2),(x_1,x_2+1)> is set
    """
    probh = np.random.rand(*ph.shape)
    probv = np.random.rand(*pv.shape)
    bondsh = probh<ph
    bondsv = probv<pv
    return bondsh, bondsv

@jit(nopython = True)
def assign_clusters_from_bonds(bondsh, bondsv):
    """ Assigns each site to a cluster with respect to the connectivity specified in terms of activated bonds.

    This function assigns a unique cluster id to each connected component regarding the connectivity specified int via
    activated bonds. It assumes a rectangular geometry with periodic boundary conditions.

    Args:
        bondsh (bool array): if the bond between x and x + (1,0) is activated, bondsh[x_1, x_2] should be true
        bondsv (bool array): if the bond between x and x + (0,1) is activated, bondsh[x_1, x_2] should be true

    Returns:
        isinc ([sx,sy] int array): an array containing the uniqe id of the cluster each site belongs to.
    """
    sx = bondsh.shape[0]
    sy = bondsh.shape[1]
    clustno=-1.

    isinc=-1*np.ones(bondsh.shape, dtype = numba.int32)

    for x in range(sx):
        for y in range(sy):

            if isinc[x,y]<-0.1:
                clustno=clustno+1
                isinc[x,y]= clustno
                queue=[]
                queue.append((x,y))
                while queue:
                    cx,cy = queue.pop()
                    nbbonds = [bondsh[cx,cy], bondsh[(cx-1)%sx,cy], bondsv[cx,cy], bondsv[cx,(cy-1)%sy] ]
                    nbcoords = [((cx+1)%sx,cy),((cx-1)%sx, cy), (cx,(cy+1)%sy), (cx, (cy-1)%sy)]

                    for bond,coord in zip(nbbonds, nbcoords):
                        if bond and isinc[coord]==-1:
                            isinc[coord] = clustno
                            queue.append(coord)
            
    return isinc


@jit(nopython = True)
def flip_clusters_5050(dofs, isinc, wolffnormal):
    """ Flip each cluster with 50 % probability inplace.

    Args:
        dofs ([sx,sy, 2] float array) The spin state of the system. the spins in this array are flipped.
        isinc ([sx,sy] int array)  An array containing the cluster id to which each site belongs (as generated by assign_clusters_from_bonds)
        wolffnormal ([2] float array) The vector normal to the wolffplane.

    """
    nclusters = int(np.max(isinc))
    clusterids = np.arange(nclusters)
    clusterstoflip = clusterids[np.random.rand(nclusters)>=0.5]

    sx, sy = isinc.shape
    clusterstoflip_set = set(clusterstoflip)
    sitestoflip = np.full(isinc.shape, False)
    for x in range(sx):
        for y in range(sy):
            if isinc[x,y] in clusterstoflip_set:
                dofs[x,y,:] = flip_dof_0d(dofs[x,y,:], wolffnormal)

@jit(nopython = True)
def mc_update(dofs, beta, ntimes = 1):
    """
    Perfomrs a full multicluster update.

    This function assigns a random wolffplane with random_unit_vector, assigns clusters and flips each
    cluster with 50 % probability. The flip is performed inplace.

    Args:
        dofs([sx,sy,2] float array): the spin state of the system. This is modified inplace.
        beta(float): The inverset Temperature.
        ntimes(int): The number of multicluster updates performed, defaults to 1.
    """
    for i in range(ntimes):
        wn = random_unit_vector()
        isinc = assign_clusters(dofs, beta, wn)
        flip_clusters_5050(dofs, isinc, wn)
@jit(nopython =True)
def flip_cluster(dofs, isinc, clid, wolffnormal):
    dofs[isinc==clid] =  flip_dofs_1d(dofs[isinc==clid], wolffnormal)
    return 0



@jit(nopython =True)
def assign_clusters(dofs, beta,wn):
    """Assigns clusters with respect to the specified wolffplane.

    This function is a shortcut for calling p_add_bond, set_bonds and assign_clusters_from_bonds successively.
    """
    ph , pv = p_add_bond(dofs, wn, beta)
    bondsh, bondsv = set_bonds(ph, pv)
    isinc = assign_clusters_from_bonds(bondsh,bondsv)
    return isinc

@jit(nopython=True)
def sc_update(dofs, beta, ntimes =1):
    """Performs a Wolff single cluster update inplace."""
    wn = random_unit_vector()
    ph, pv = p_add_bond(dofs, wn,beta)
    bondsh, bondsv = set_bonds(ph,pv)
    sx,sy = bondsh.shape
    x0,y0 = np.random.randint(0, dofs.shape[0],2)
    start =(x0,y0)
    queue=[]
    queue.append(start)

    isinc=-1*np.ones(bondsh.shape)
    clsize=0

    while queue:
        cx,cy = queue.pop()
        dofs[cx,cy,:] = flip_dof_0d(dofs[cx,cy,:], wn)
        clsize+=1
        nbbonds = [bondsh[cx,cy], bondsh[(cx-1)%sx,cy], bondsv[cx,cy], bondsv[cx,(cy-1)%sy] ]
        nbcoords = [((cx+1)%sx,cy),((cx-1)%sx, cy), (cx,(cy+1)%sy), (cx, (cy-1)%sy)]
        for bond,coord in zip(nbbonds, nbcoords):
            if bond and isinc[coord]==-1:
                isinc[coord] = 1 
                queue.append(coord)


    return clsize 

@jit(nopython=True)
def metropolis_update(dofs, beta, nsteps=1):
    """Performs nsteps metropolis updates inplace."""
    for i in range(nsteps):
	    # choose a random site
        sx,sy = dofs.shape[:2]
        x = np.random.randint(0,sx)
        y = np.random.randint(0,sy)
        nbsites = [((x+1)%sx,y),((x-1)%sx, y), (x,(y+1)%sy), (x,(y-1)%sy)]
        currentspin = dofs[x,y]
        newspin = random_unit_vector()
        prevenergy = 0
        newenergy = 0
        for nb in nbsites:
            prevenergy -= currentspin.dot(dofs[nb])
            newenergy -= newspin.dot(dofs[nb])
            paccept = np.exp(-beta*(newenergy-prevenergy))
        if np.random.rand() < paccept:
                dofs[x,y] = newspin

