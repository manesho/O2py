""" Functions related to tracing of cluster boundaries

Some conventions:

the coordinate of a plaquette is the coordinate of the southwest site of that plaquette

the boundary orientation is such that the interior of the cluster is to the left

if a corss plaquette is encountered: turn left

plaquette site indices:

    3
       |
     3 - 4
 0 - |   | - 2
     0 - 1
       |
       1

"""

from numba import jit
import numba
import numpy as np


@jit(nopython=True)
def get_plaquette_state(x,y, isinc, cid):
    sx,sy = isinc.shape
    pstate = np.zeros(4)
    pstate[0] = isinc[x,y]==cid
    pstate[1] = isinc[(x+1)%sx, y]==cid
    pstate[2] = isinc[(x+1)%sx, (y+1)%sy]==cid
    pstate[3] = isinc[x, (y+1)%sy]==cid
    return pstate


@jit(nopython = True)
def next_move(pstate, prev_exitdir):
    """Returns the direction to move on along the boundary.

   The exit direction is calculated from the state of the plaquette given by pstate in such a way
   that the interior of the cluster remains to the left.
   if a cross plaquette is encountered, always turn left, therefore we need the previous exit directaion as well. If we start a boundary trace this can be signaled by giving -1 as prev_exitdir, then always the second exit is chosen.
   directions are encoded as 0: west, 1:south, 2:east, 3:nord

   Args:
       pstate ([4] bool array): Marking wehter the i'th site is in the cluster.
       prev_exitdir (int): the direction exiting the last plaquette

   Returns:
       the exit direction encoded as 0: west, 1:south, 2:east, 3:north
    """
    exitcounts=0
    for i in range(4):
        if pstate[i] == True and pstate[i-1] ==False:
            exitcounts +=1
            exitdir = i
    # if it is the start always exit at exitdir even if there are two exits
    #if there are two exits, the plaquette is a cross plaquette: always turn left 
    if exitcounts >1:
        exitdir = (prev_exitdir+1)%4
    return exitdir

@jit(nopython = True)
def next_plaquette(x,y, sx,sy, move):
    """Returns the hook coordinate of the next plaquette. 
    Args:
        x (int): x coordinate of current plaquette
        y (int): y coordinate of current plaquette
        sx(int): system extent in x direction
        sy(int): system extent in y direction
        move(int): a number (0...3) encoding the move direction as returned by next_move

    Returns:
        x,y the coordinates of the next plaquette hook
    """
    #go west
    if move == 0:
        return (x-1)%sx, y
    #go south
    if move == 1:
        return x, (y-1)%sy
    #go east
    if move == 2:
        return (x+1)%sx, y
    #go north
    if move == 3:
        return x, (y+1)%sy
 
#@jit nextmove
#@jit(nopython=True)
#def nextmove(plaquette_step, inmove):
#
#
@jit(nopython=True)
def trace_boundary_from_start(start, cid, isinc):
    current = start
    #check that you are on a cluster boundary:
    pstatestart = get_plaquette_state(*current, isinc, cid)
    if pstatestart.sum()==4 or pstatestart.sum()==0:
        raise Exception('Start not on cluster boundary')
    this_move=-1
    i=0
    while True:
        pstate = get_plaquette_state(*current, isinc, cid)
        this_move = next_move(pstate, this_move)
        current = next_plaquette(*current, *isinc.shape, this_move)
        i+=1
        if current == start:
            break
    return i

@jit(nopython=True)
def bdry_int_sig(v1,v2,wn):
    """Returns the boundary interpolation signature between v1 and v2, wn is the wolff normal"""
    return myangleorient(wn,v1) if v1.dot(wn)**2 < v2.dot(wn)**2 else myangleorient(wn,v2)

@jit(nopython=True)
def myangleorient(v1,v2):
    return 1 if v1[0]*v2[1] > v1[1]*v2[0] else -1

@jit(nopython = True)
def move_signature(x,y, sx,sy, move, dofs, wn):
    if move == 0:
        dof1 = dofs[x,y]
        dof2 = dofs[x,(y+1)%sy]
    if move == 1:
        dof1 = dofs[x,y]
        dof2 = dofs[(x+1)%sx,y]
    if move == 2:
        dof1 = dofs[(x+1)%sx,y]
        dof2 = dofs[(x+1)%sx,(y+1)%sy]
    if move == 3:
        dof1 = dofs[(x+1)%sx,(y+1)%sy]
        dof2 = dofs[x,(y+1)%sy]
    return bdry_int_sig(dof1,dof2, wn)

@jit(nopython=True)
def get_bdry_signature(start, cid, dofs, isinc, wn):
    """ Returns the coordinates of the next halfvortex along the boundary of the cid cluster starting from the plaquette at start and following the orientation of the boundary (interior to the left) """
    current = start
    #check that you are on a cluster boundary:
    pstatestart = get_plaquette_state(*current, isinc, cid)
    if pstatestart.sum()==4 or pstatestart.sum()==0:
        raise Exception('Start not on cluster boundary')
    this_move=-1
    first_move = next_move(pstatestart, this_move)
    startsig = move_signature(*current, *isinc.shape, first_move, dofs, wn)
    signature=[]
    i=0
    while True:
        pstate = get_plaquette_state(*current, isinc, cid)
        this_move = next_move(pstate, this_move)
        currentsig = move_signature(*current, *isinc.shape, this_move, dofs, wn)
        signature.append(currentsig)
        current = next_plaquette(*current, *isinc.shape, this_move)
        i+=1
        if current == start:
            break
    return signature

@jit(nopython=True)
def get_next_halfvortex_location(start, cid, dofs, isinc, wn, entrymove):
    """ Returns the coordinates of the next halfvortex along the boundary of the cid cluster starting from the plaquette at start and following the orientation of the boundary (interior to the left) """
    current = start
    #check that you are on a cluster boundary:
    pstatestart = get_plaquette_state(*current, isinc, cid)
    if pstatestart.sum()==4 or pstatestart.sum()==0:
        raise Exception('Start not on cluster boundary')

    this_move= entrymove 
    first_move = next_move(pstatestart, this_move)
    startsig = move_signature(*current, *isinc.shape, first_move, dofs, wn)
    currentsig = startsig
    i=0
    while True:
        i+=1
        pstate = get_plaquette_state(*current, isinc, cid)
#        this_move = next_move_signature_sensitive(pstate, this_move,currentsig,*current,*isinc.shape, dofs, wn)
        this_move = next_move(pstate, this_move)
        currentsig = move_signature(*current, *isinc.shape, this_move, dofs, wn)
        if currentsig != startsig:
            break
        current = next_plaquette(*current, *isinc.shape, this_move)
        #print(current)
        #if current == start:
        #    print('cid = ', cid )
        #    print('start = ', start )
        #    print('i=' , i)
        #    raise Exception('closed loop')

        if i >= isinc.shape[0]*isinc.shape[1]:
            raise Exception('overflow:  points in boundary  > L^2')
    return current






@jit(nopython = True)
def get_start_move(x,y,dofs, isinc, cid,wn,sig2follow):
    sx,sy =isinc.shape
    pstate = get_plaquette_state(x,y,isinc, cid)
    # if it's not a cross plaquette everything is fine and the entry direction is irrelevant
    if not ((pstate[0] == 0 and pstate[1] == 1 and pstate[2] ==0 and pstate[3] ==1 ) or
            (pstate[0] == 1 and pstate[1] == 0 and pstate[2] ==1 and pstate[3] ==0 )    ):
        return -1
    else:
        for move in range(4):
            # if the move signature in the directio move is the signature you want to follow and if
            # the spin to the left of that move, this is the exit you want to take.
            if ( move_signature(x,y,sx,sy, move, dofs, wn) == sig2follow and
                 pstate[move] == 1 and
                 move_signature(x,y,sx,sy,(move+1)%4 ,dofs, wn )!= sig2follow):
                # turn right to get the entry direction for this exit
                return (move-1)%4 



                
@jit(nopython = True)
def construct_boundary_elist(dofs, isinc, wn):
    """
    Construct the vortex-cluster-boundary (proper name still missing) graph from a configuration, specified by dofs, isinc and wn
    Args:
        dofs: the spins
        isinc: the cluster id of each spins
        wn : the current wolff plane

    returns:
        the graph as a list of tuples for all edges.
    """
    elist = []
    hvlocs, cidps, cidms = find_hv_fake_included(dofs, isinc, wn)
    for i in range(len(cidps)):
        hv = hvlocs[i]
        cidp = cidps[i]
        cidm = cidms[i]



        entrymove = get_start_move(*hv, dofs, isinc , cidp,wn, -1)
        nextvert = get_next_halfvortex_location(hv, cidp, dofs,isinc, wn,entrymove)
        elist.append((hv,nextvert))

        entrymove = get_start_move(*hv, dofs, isinc , cidm,wn, 1)
        nextvert = get_next_halfvortex_location(hv, cidm, dofs,isinc, wn,entrymove)
        elist.append((hv,nextvert))

    return elist











@jit(nopython=True)
def find_hv_fake_included(dofs, isinc, wn):

    hvloc = []
    hvcid = []
    hvacid = []

    sx,sy = isinc.shape
    psig = np.zeros(4)

    for x in range(sx):
        for y in range(sy):

            pisinc = get_plaquette_isinc(x,y,isinc)
            psig = get_plaquette_signagture(x,y,*isinc.shape,dofs, wn )

            vorts = np.zeros(4)
            hasvort = False
            sitem=0
            sitep=0

            #print('--------------')
            #print(x,y)
            #print(pisinc)
            #print(psig)

            for i in range(4):
                if psig[i-1] == 1 and psig[i] ==-1:
                    vorts[i]=-0.5
                    sitem =i 
                    hasvort=True
                if psig[i-1] == -1 and psig[i] ==1:
                    vorts[i]=0.5
                    sitep=i


            #print(sitep, sitem)

            if hasvort:
                # if the spins with nonzero vorticity are in different clusters add the halfvortex
                if not pisinc[sitem] == pisinc[sitep]:
                    hvloc.append((x,y))
                    hvcid.append(pisinc[sitep])
                    hvacid.append(pisinc[sitem])

                    # now check for fake half vortices as well:
                    #check weather it is a cross plaquette and add it to the vortex list
                elif ((sitep-sitem)%4 == 2) and not pisinc[(sitep+1)%4] == pisinc[sitep] and not pisinc[(sitem+1)%4] == pisinc[sitem]:
                    #print('whoop')
                    hvloc.append((x,y))
                    hvcid.append(pisinc[sitep])
                    hvacid.append(pisinc[sitem])
    return hvloc, hvcid, hvacid



@jit(nopython=True)
def get_plaquette_signagture(x,y,sx,sy,dofs,wn):
    psig = np.zeros(4)
    for move in range(4):
        psig[move] = move_signature(x,y,sx,sy, (move+1)%4, dofs, wn)
    return psig



@jit(nopython=True)
def get_plaquette_isinc(x,y, isinc):
    sx,sy = isinc.shape
    pstate = np.zeros(4)
    pstate[0] = isinc[x,y]
    pstate[1] = isinc[(x+1)%sx, y]
    pstate[2] = isinc[(x+1)%sx, (y+1)%sy]
    pstate[3] = isinc[x, (y+1)%sy]
    return pstate


@jit(nopython=True)
def get_adj_from_elist(elistint):
#    adj = np.zeros([4,4])
    adj =np.zeros((len(elistint)//2 ,2), dtype=np.int64)
    for i  in range(len(elistint)//2):
        adj[i,0] = int(elistint[2*i][1])
        adj[i,1] = int(elistint[2*i+1][1])
    return adj

@jit(nopython=True)
def connected_components(adja):
    components=[]
    visited = np.zeros(adja.shape[0])
    for v in range(adja.shape[0]):
        if not visited[v]:
            components.append(dfscc(v,visited,adja))
    return components


@jit(nopython=True)
def dfscc(v,visited, adj):
    visited[v] = 1
    count = 1
    for vnb in adj[v]:
        if not visited[vnb]:
            count += dfscc(vnb, visited, adj )
    return count 




def bdryhvg_analysis(dofs,isinc, wn):
    loc, c1, c2 = find_hv_fake_included(dofs, isinc, wn)
    elist = construct_boundary_elist(dofs,isinc,wn)
    vid ={pos:idx for idx,pos in enumerate(loc)}
    elistint = [(vid[e[0]], vid[e[1]]) for e in elist]
    adj= get_adj_from_elist(elistint)
    ccs = connected_components(adj)
    return max(ccs), sum(ccs)







