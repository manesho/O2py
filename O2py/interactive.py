""" An interactive visualization of the 2d O(2) or XY model with matplotlib"""


import numpy as np
import random
from numba import njit
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import O2py.wolffcompiled as pw
import O2py.vortices as o2v


@njit
def get_clo(dofs, isinc, wn):
    """
    Returns the mean orientation of each cluster with respect to the wolff normal vector.

    This function calculates:
    $$ \bar{\omeage}(x) =\frac{1}{|c|} \sum_{y\in c(x)} s_x\cdot w $$

    Parameters:
    dofs([sx,sy,2] float array): Contains the current state of the O2 model.
    isinc([sx,sy] int array): isinc[x,y] contains the number of the cluster the site at x,y belongst to.
    wn([2] float array): the vector normal to the wolff plane

    Returns:
    [sx,sy] float array: 

    """
    clomegas = np.zeros((round(int(np.max(isinc)+1)), 2))
    nsitesinc = np.zeros(round(int(np.max(isinc)+1)))
    clos = np.zeros(isinc.shape, dtype=np.float64)
    isincint = isinc.astype(np.int32)
    for x in range(isincint.shape[0]):
        for y in range(isincint.shape[1]):
            clomegas[isincint[x,y]]+=dofs[x,y]
            nsitesinc[isincint[x,y]]+=1
    for x in range(isincint.shape[0]):
        for y in range(isincint.shape[1]):
            clos[x,y] = (wn[0]*clomegas[isincint[x,y],0]+wn[1]*clomegas[isincint[x,y],1])/nsitesinc[isincint[x,y]]
    return clos


class interactiveo2plot:
    """A class for the interactive visualization of the 2d O(2) model.

    Attributes:
        background (char): a key denoting which background is activated:
              'o' : the mean cluster orientation with respect to the wolffplane  
              'i' : the local degrees of freedom (with repsect to the wolff plane)
              's' : large clusters have a random color, small cluster a random greyscale value.
        show_vortices(bool) : if true, vortices and anti-vortices are plotted
        clustercontour(dict) : a dict of clid:cluster boundary contour plot handle, for all cluster boundary contours that are shown
        axis: a handle to the axis of the plot
        clplot: a handle to the background plot
        pvs: a handle to the vortex plot
        pavs: a handle to the antivortex plot
        pfreevs: a handle to the free vortex plot
    
        dofs([sx,sy,2] float array): the spin state of the model.
        beta(float): the inverse temperature
        wn([2] float array): the current wolff plane
        isinc([sx,sy] int): the current cluster assignment

        clo ([sx,ys] int): the mean orientation of the assigned cluster for each site
 
    """

    manual ="""
    right click: flips the cluster
    keys:
        alt+o : show mean cluster orientation as background
        alt+i : show the clusters with qualitative colors
        alt+s : show the spin orientaitons with respect to the wolff plane

        v : toggle vortices
        b : toggle boundary  of cluster under cursor
        B : remove all boundaries from plot

        u : perform a multicluster update
        r : perform 20 multicluster updates
        m : perform L^2 * 100 metropolis updates

        c : show the vorticity change induced by the last cluster flip (by right click)

        up: increase the inverse temperature by 0.1
        down: decrease the inverse temperature by 0.1
        
    """

    def __init__(self, beta=1., l = 40, dofs=None, wn=pw.random_unit_vector()):

        if dofs  is None:
            dofs = pw.random_dofs(l,l)
        print(self.manual)
        self.bgplotcmds = {'alt+o':self.update_cloplot, 
                           'alt+i':self.plot_isinc,
                           'alt+s':self.plot_orientation}


        self.background = list(self.bgplotcmds.keys())[0]
        self.dofs=dofs
        self.beta=beta
        self.wn = wn
        self.isinc=pw.assign_clusters(dofs,beta, wn)



        self.clo = self.clo_from_dofs()

        self.show_vortices = False
        self.clustercontour = {}

        plt.ion()
        fig, ax = plt.subplots()
        self.axis = ax
        self.set_title()
        self.clplot = ax.imshow(self.clo, cmap = 'coolwarm')
        self.clplot.set_clim(-1,1)
        plt.subplots_adjust(right=0.7)
        self.pvs = plt.plot([0],[0],'^C5', markersize=5, label='Vortices')
        self.pavs = plt.plot([1],[1],'vC8', markersize=5, label='Anti Vortices')
        self.pfreevs = plt.plot([1],[1],'oC2',markersize=7, fillstyle='none', label = 'Free Vortices')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

        self.update_vortex_plot()

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        plt.show()

    def plot_orientation(self):
        """Set the background to the angle of each dof with respect to the wolffplane."""
        phis = np.arctan2(self.dofs[:,:,0], self.dofs[:,:,1])
        phiwn = np.arctan2(self.wn[0], self.wn[1])
        phis = -np.mod(phiwn-phis+np.pi/2, 2*np.pi)+np.pi

        self.clplot.set_array(phis)
        self.clplot.set_clim(-np.pi,np.pi)
        self.clplot.set_cmap('twilight')

    def plot_isinc(self):
        """Set the background to a qualitative cluster visualization"""
        clcmap = cluster_random_cmap(self.isinc, mode='largecolored1')
        self.clplot.set_clim(0,np.max(self.isinc))
        self.clplot.set_array(self.isinc)
        self.clplot.set_cmap(clcmap)

    def update_cloplot(self):
        """Set the background to the mean cluster orientation with respect to the wolff normal"""
        self.clplot.set_array(self.clo)
        self.clplot.set_clim(-1.,1.)
        self.clplot.set_cmap('coolwarm')
     
    def update_background(self ):
        """Updates whichever background is activated."""
        self.bgplotcmds[self.background]()


    def mc_update(self, ntimes =1):
        """performs ntimes multicluster updates and updates the mean cluster orientation"""
        pw.mc_update(self.dofs, self.beta, ntimes =ntimes)
        self.wn = pw.random_unit_vector()
        self.isinc=pw.assign_clusters(self.dofs,self.beta, self.wn)
        self.clo= self.clo_from_dofs()
        
        
    def flip_cluster_at(self,x,y):
        """ Flip the cluster to which the site at (x,y) belongs to and update the mean cluster orientation"""
        self.pvprev = o2v.plaquettevorticity_vec(self.dofs)
        
        clid = self.isinc[int(round(y)),int(round(x))]
        self.dofs[self.isinc==clid] = pw.flip_dofs_1d(self.dofs[self.isinc==clid],self.wn)
        self.clo[self.isinc==clid] = - self.clo[self.isinc==clid]

    def plot_vort_changes(self):
        pv = o2v.plaquettevorticity_vec(self.dofs)
        pvd = pv - self.pvprev
        sx,sy= pv.shape[0:2]
        x, y = np.meshgrid(range(sx),range(sy))
        vlocx =x[np.where(pvd==1)]
        vlocy =y[np.where(pvd==1)]
        avlocx=x[np.where(pvd==-1)]
        avlocy=y[np.where(pvd==-1)]

        self.pfreevs[0].set_data([],[])
        self.pvs[0].set_data(vlocx-0.5, vlocy-0.5)
        self.pavs[0].set_data(avlocx-0.5, avlocy-0.5)
    
    def update_vortex_plot(self):
        """"""
        if self.show_vortices:
            pv = o2v.plaquettevorticity_vec(self.dofs)
            sx,sy= pv.shape[0:2]
            x, y = np.meshgrid(range(sx),range(sy))
            vlocx =x[np.where(pv==1)]
            vlocy =y[np.where(pv==1)]
            avlocx=x[np.where(pv==-1)]
            avlocy=y[np.where(pv==-1)]

            freevs=o2v.completely_free_vortex_locations(self.dofs,self.isinc,self.wn)
            if len(freevs)>0:
                freex = np.array(freevs)[:,1]
                freey = np.array(freevs)[:,0]
                self.pfreevs[0].set_data(freex-0.5, freey-0.5)
            else:
                self.pfreevs[0].set_data([],[])

            self.pvs[0].set_data(vlocx-0.5, vlocy-0.5)
            self.pavs[0].set_data(avlocx-0.5, avlocy-0.5)
            
        else:
            self.pvs[0].set_data([],[])
            self.pavs[0].set_data([],[])
            self.pfreevs[0].set_data([],[])    

    def clo_from_dofs(self):
        """ returns the mean cluster orientaion"""
        return get_clo(self.dofs, self.isinc,self.wn)
        

    def flip_to_refconv(self):
        refconf = get_reference_configuration(self.dofs, self.isinc, self.wn)
        self.dofs = refconf
        self.clo= self.clo_from_dofs()

       
    def set_title(self):
        """updates the plot title to show the number of currently present vortices"""
        nvorts = np.sum( o2v.plaquettevorticity_vec(self.dofs)!=0)/np.prod(self.isinc.shape)
        nfreevorts = len(o2v.completely_free_vortex_locations(self.dofs,self.isinc,self.wn))/np.prod(self.isinc.shape)
        self.axis.set_title('$\\beta = {:.2f}, \\rho_v = {}, \\rho_v^{{(f)}} = {}$'.format(self.beta, nvorts, nfreevorts))

    def remove_all_contours(self):
        for contplot in self.clustercontour.values():
            for coll in contplot.collections:
                try:
                    coll.remove()
                except:
                    pass
        self.clustercontour = {}

    def onclick(self, event):
        if event.button ==3:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self.flip_cluster_at(x,y)
            self.update_background()
            self.update_vortex_plot()
            self.set_title()
            
    def onkey(self, event):
        if event.key in self.bgplotcmds.keys():
            self.background = event.key
            self.update_background()
        
        if event.key =='u':
            self.mc_update()
            self.update_background()
            self.update_vortex_plot()
            self.set_title()
        if event.key =='r':
            self.mc_update(ntimes = 20)
            self.update_background()
            self.update_vortex_plot()
            self.set_title()
        if event.key =='R':
            self.flip_to_refconv()
            self.update_cloplot()
            self.update_vortex_plot()
            self.set_title()
        if event.key =='v':
            self.show_vortices = not self.show_vortices
            self.update_vortex_plot()
            self.set_title()
        if event.key == 'c':
            self.plot_vort_changes()
        if event.key=='up':
            self.beta +=0.1
            self.set_title()
        if event.key=='down':
            self.beta-=0.1
            self.set_title()
        if event.key == 'm':
            pw.metropolis_update(self.dofs, self.beta, nsteps = np.prod(self.isinc.shape)*100)
            self.update_background()
            self.update_vortex_plot()

        if event.key == 'B':
            self.remove_all_contours()

        if event.key == 'b':
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            clid = self.isinc[y,x]
            if clid not in self.clustercontour.keys():
                self.clustercontour[clid]=plt.contour(self.isinc==clid, [1], colors =['yellow'], linewidths=1)
            else:
                for coll in self.clustercontour[clid].collections:
                    coll.remove()
                del self.clustercontour[clid]


def get_reference_configuration(dofs, isinc, wolffnormal):
    """Flip everything to the reference configuration"""
    nclusters= int(np.max(isinc))
    clusterids = np.arange(nclusters+1)
    refconf = dofs.copy()
    for cluster in clusterids:
        #get a representative:
        x,y = np.argwhere(isinc==cluster)[0]
        if dofs[x,y,:].dot(wolffnormal) < 0:
            refconf[isinc==cluster] = pw.flip_dofs_1d(refconf[isinc==cluster],wolffnormal)
    return refconf

def cluster_random_cmap(isinc, mode ='eyecancer'):
    """returns a custom random colormap to visualize isinc with plt.imshow or similar."""
    unique, counts = np.unique(isinc, return_counts=True)
    clweights =dict(zip(unique, counts))
    mycmapl=[]

    cfuns ={
        'soft': lambda clidx: random.choice(cm.tab20c.colors),
        'eyecancer': lambda clidx: cm.hsv(np.random.uniform()),
        'eyecancer2': lambda clidx: cm.hsv(np.random.uniform()),
        'largecolored1': lambda clidx: random.choice(cm.tab20b.colors) if clweights[clidx]>10 else np.ones(3)* np.random.uniform(0.6,1),
        'largecolored2': lambda clidx: random.choice(cm.tab20b.colors) if clweights[clidx]>50 else np.ones(3)* np.random.uniform(0.6,1)
    }

    cmap = ListedColormap([cfuns[mode](clidx) for clidx in range(int(isinc.max())+1)])
    return cmap

