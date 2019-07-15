""" An interactive visualization of the 2d O(2) or XY model with matplotlib"""


import numpy as np
import random
from numba import njit
try:
    from licpy.lic import runlic
except:
    print('line integral convolution not installed')
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import O2py.wolffcompiled as pw
import O2py.vortices as o2v

MARKERSIZE = 5

class interactiveo2plot:
    """A class for the interactive visualization of the 2d O(2) model.

    Attributes:
        bgplotcmds: a dict of keybinding:function to update the background
        background: the keybinding of the active background
        axis: a handle to the axis of the plot
        clplot: a handle to the background plot
        layers: dict of all layers (active and inactive) with key bindings as keys
    

    
        dofs([sx,sy,2] float array): the spin state of the model.
        beta(float): the inverse temperature
        wn([2] float array): the current wolff plane
        isinc([sx,sy] int): the current cluster assignment

        clo ([sx,ys] int): the mean orientation of the assigned cluster for each site
        pv ([sx,sy] int): the local vorticity
        freevs: vortices, for which both associated clusters are only associated
                  to one vortex currently visible
    """

    manual ="""
    right click: flips the cluster
    keys:

        Backgrounds
        -----------
        alt+o : show mean cluster orientation as background
        alt+i : show the clusters with qualitative colors
        alt+s : show the spin orientaitons with respect to the wolff plane
        alt+l : show the spin orientaitons as line integral convolution

        Layers
        ------
        alt+v : toggle vortices

        alt+b : toggle boundary orientaion of cluster under cursor
        alt+B : remove all boundary orientations from plot

        alt+c : toggle boundary contour of cluster under cursor
        alt+C : remove all boundary contours from plot

        alt+q : toggle a quiver plot of the spins

        alt+d : toggle vorticity changes du to last cluster flip (a bit experimental)



        Model Controls
        --------------
        u : perform a multicluster update
        r : perform 20 multicluster updates
        m : perform L^2 * 100 metropolis updates
        R : flip everything to the reference configuration

        up: increase the inverse temperature by 0.1
        down: decrease the inverse temperature by 0.1

    """

    def __init__(self, beta=1., l = 40, dofs=None, wn=pw.random_unit_vector()):

        if dofs  is None:
            dofs = pw.random_dofs(l,l)
        print(self.manual)
        self.bgplotcmds = {'alt+o':self.update_cloplot, 
                           'alt+i':self.plot_isinc,
                           'alt+s':self.plot_orientation,
                           'alt+l':self.plot_lic}

        self.background = list(self.bgplotcmds.keys())[0]


        self.dofs=dofs
        self.beta=beta
        self.wn = wn
        self.isinc=pw.assign_clusters(dofs,beta, wn)

        self.clo = self.clo_from_dofs()
        self.pv = o2v.plaquettevorticity_vec(self.dofs)
        self.freevs=o2v.completely_free_vortex_locations(self.dofs,self.isinc,self.wn)

        #self.show_dof_quiver = False

        plt.ion()
        fig, ax = plt.subplots()
        self.axis = ax


        self.set_title()
        self.clplot = ax.imshow(self.clo, cmap = 'coolwarm')


        self.layers = {'alt+v':VortexLayer(self),
                       'alt+q':QuiverLayer(self),
                       'alt+b':BoundaryLayer(self),
                       'alt+c':BoundaryContourLayer(self),
                       'alt+d':VortexChangeLayer(self)}


        self.clplot.set_clim(-1,1)
        plt.subplots_adjust(right=0.7)



        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.show()


    def plot_lic(self):

        dofsi1 = RectBivariateSpline(np.arange(self.dofs.shape[0]), np.arange(self.dofs.shape[1]), self.dofs[:,:,0])
        dofsi2 = RectBivariateSpline(np.arange(self.dofs.shape[0]), np.arange(self.dofs.shape[1]), self.dofs[:,:,1])
        xi1 = np.arange(0, self.dofs.shape[0], 0.1)
        xi2 = np.arange(0, self.dofs.shape[1], 0.1)
        xxi1, xxi2 = np.meshgrid(xi1,xi2)

        try:
            licimage = runlic(dofsi1(xi1,xi2),dofsi2(xi1,xi2),50)
        except:
            print("Line integral convolution depends on licpy, which itself depends on tensorflow: pip install O2py[lic]")
            self.background='alt+s'
            return

        self.clplot.set_array(licimage)
        self.clplot.set_clim(0,1)
        self.clplot.set_cmap('Blues')

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
        #self.update_dof_quiver()

    def update_layers(self):
        for layer in self.layers.values():
            layer.update_plot()

    def mc_update(self, ntimes =1):
        """performs ntimes multicluster updates and updates the mean cluster orientation"""
        pw.mc_update(self.dofs, self.beta, ntimes =ntimes)
        self.wn = pw.random_unit_vector()
        self.isinc=pw.assign_clusters(self.dofs,self.beta, self.wn)
        self.clo= self.clo_from_dofs()
        self.pv = o2v.plaquettevorticity_vec(self.dofs)
        self.freevs=o2v.completely_free_vortex_locations(self.dofs,self.isinc,self.wn)

    def flip_cluster_at(self,x,y):
        """ Flip the cluster to which the site at (x,y) belongs to and update the mean cluster orientation"""
        self.pvprev = o2v.plaquettevorticity_vec(self.dofs)
        
        clid = self.isinc[int(round(y)),int(round(x))]
        self.dofs[self.isinc==clid] = pw.flip_dofs_1d(self.dofs[self.isinc==clid],self.wn)
        self.clo[self.isinc==clid] = - self.clo[self.isinc==clid]
        self.pv = o2v.plaquettevorticity_vec(self.dofs)
        self.freevs=o2v.completely_free_vortex_locations(self.dofs,self.isinc,self.wn)


    def clo_from_dofs(self):
        """ returns the mean cluster orientaion"""
        return get_clo(self.dofs, self.isinc,self.wn)

    def flip_to_refconv(self):
        refconf = get_reference_configuration(self.dofs, self.isinc, self.wn)
        self.dofs = refconf
        self.clo= self.clo_from_dofs()
        self.pv = o2v.plaquettevorticity_vec(self.dofs)

    def set_title(self):
        """updates the plot title to show the number of currently present vortices"""
        nvorts = np.sum( self.pv!=0)/np.prod(self.isinc.shape)
        nfreevorts = len(self.freevs)/np.prod(self.isinc.shape)
        self.axis.set_title('$\\beta = {:.2f}, \\rho_v = {:.4f}, \\rho_v^{{(f)}} = {:.4f}$'.format(self.beta,nvorts,nfreevorts))

    def onclick(self, event):
        if event.button ==3:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self.flip_cluster_at(x,y)
            self.update_background()
            self.update_layers()
            self.set_title()
            
    def onkey(self, event):
        if event.key in self.bgplotcmds.keys():
            self.background = event.key
            self.update_background()

        if event.key in self.layers.keys():
            self.layers[event.key].toggle(event)

        if event.key == 'alt+B':
            self.layers['alt+b'].remove_all()

        if event.key == 'alt+C':
            self.layers['alt+c'].remove_all()


        if event.key =='u':
            self.mc_update()
            self.update_background()
            self.update_layers()
            self.set_title()

        if event.key =='r':
            self.mc_update(ntimes = 20)
            self.update_background()
            self.update_layers()
            self.set_title()

        if event.key =='R':
            self.flip_to_refconv()
            self.update_background()
            self.update_layers()
            self.set_title()

        if event.key=='up':
            self.beta +=0.1
            self.set_title()
        if event.key=='down':
            self.beta-=0.1
            self.set_title()
        if event.key == 'm':
            pw.metropolis_update(self.dofs, self.beta, nsteps = np.prod(self.isinc.shape)*100)
            self.update_background()
            self.update_layers()





###########################################################################################
class QuiverLayer:
    def __init__(self, o2plot):
        self.o2plot = o2plot
        self.active = False
        self.handle = self.o2plot.axis.quiver(self.o2plot.dofs[:,:,0], self.o2plot.dofs[:,:,1], alpha = 0, scale_units='xy')
    def hide(self):
        self.handle.set_alpha(0)

    def update_data(self):
        self.handle.set_alpha(1.)
        self.handle.set_UVC(self.o2plot.dofs[:,:,0], self.o2plot.dofs[:,:,1])

    def update_plot(self):
        if self.active == True:
            self.update_data()

    def toggle(self, event):
        if self.active == False:
            self.active =True
            self.update_plot()
        else:
            self.active = False
            self.hide()

###########################################################################################
class BoundaryContourLayer:
    def __init__(self, o2plot):
        self.o2plot = o2plot
        self.clustercontour = {}

    def update_plot(self):
        return

    def remove_all(self):
        for contplot in self.clustercontour.values():
            for coll in contplot.collections:
                try:
                    coll.remove()
                except:
                    pass
        self.clustercontour = {}

    def toggle(self, event):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        clid = self.o2plot.isinc[y,x]
        if clid not in self.clustercontour.keys():
            self.clustercontour[clid]=plt.contour(self.o2plot.isinc==clid, [1], colors =['yellow'], linewidths=1)
        else:
            for coll in self.clustercontour[clid].collections:
                coll.remove()
            del self.clustercontour[clid]





###########################################################################################
class BoundaryLayer:
    def __init__(self, o2plot):
        self.o2plot = o2plot
        self.boundaryplots_p={}
        self.boundaryplots_m={}

    @staticmethod
    def shiftinalldirs(array):
        return [np.roll(array, -1 ,axis = 0),
                np.roll(array, 1, axis =0),
                np.roll(array, -1 ,axis = 1),
                np.roll(array, 1, axis =1)]

    @staticmethod
    def bdry_int_sig(v1,v2,wn):
        return np.sign(np.linalg.det([v1,wn])) if v1.dot(wn) < v2.dot(wn) else np.sign(np.linalg.det([v2,wn]))

    def update_plot(self):
        return

    def toggle(self, event):

        bdry_int_sig_vec = np.vectorize(self.bdry_int_sig, signature='(2),(2),(2)->()')

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        clid = self.o2plot.isinc[y,x]
        if clid not in self.boundaryplots_p.keys():
            refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,self.o2plot.wn )
            allorientations = [bdry_int_sig_vec(refconf, refconfshifted,self.o2plot.wn)
                               for refconfshifted in self.shiftinalldirs(refconf)]
            clbdrybools = [(self.o2plot.isinc ==clid) & (shiftedisinc != clid)
                           for shiftedisinc in self.shiftinalldirs(self.o2plot.isinc) ]
            x2plot = []
            y2plot = []
            os2plot= []

            for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.4,-0.4,0.,0.],[0.,0.,0.4,-0.4]):
                x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
                y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
                os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
            self.boundaryplots_p[clid] = self.o2plot.axis.plot(y2plot[os2plot==1], x2plot[os2plot==1], 'C0+' )
            self.boundaryplots_m[clid] = self.o2plot.axis.plot(y2plot[os2plot==-1],x2plot[os2plot==-1], 'C1_' )

        else:
            self.boundaryplots_m[clid][0].remove()
            self.boundaryplots_p[clid][0].remove()
            del self.boundaryplots_m[clid]
            del self.boundaryplots_p[clid]

    def remove_all(self):
        for clid in self.boundaryplots_m.keys():
            self.boundaryplots_m[clid][0].remove()
            self.boundaryplots_p[clid][0].remove()
        self.boundaryplots_m={}
        self.boundaryplots_p={}


###########################################################################################
class VortexChangeLayer:
    def __init__(self, o2plot):
        self.o2plot = o2plot
        self.active =False
        self.pvs = self.o2plot.axis.plot([0],[0],'^C4', markersize=MARKERSIZE)
        self.pavs = self.o2plot.axis.plot([1],[1],'vC6', markersize=MARKERSIZE )

    def update_data(self):
        sx,sy= self.o2plot.pv.shape[0:2]
        x, y = np.meshgrid(range(sx),range(sy))
        vlocx =x[np.where((self.o2plot.pv-self.o2plot.pvprev) ==1)]
        vlocy =y[np.where((self.o2plot.pv-self.o2plot.pvprev) ==1)]
        avlocx=x[np.where((self.o2plot.pv-self.o2plot.pvprev) ==-1)]
        avlocy=y[np.where((self.o2plot.pv-self.o2plot.pvprev) ==-1)]

        self.pvs[0].set_data(vlocx-0.5, vlocy-0.5)
        self.pavs[0].set_data(avlocx-0.5, avlocy-0.5)


    def update_plot(self):
        if self.active ==True:
            self.update_data()

    def hide(self):
        self.pvs[0].set_data([],[])
        self.pavs[0].set_data([],[])

    def toggle(self, event=None):
        if self.active == False:
            self.active =True
            self.update_plot()
        else:
            self.active = False
            self.hide()


###########################################################################################
class VortexLayer:
    def __init__(self, o2plot):
        self.o2plot = o2plot
        self.active =False
        self.pvs = self.o2plot.axis.plot([0],[0],'^C5', markersize=MARKERSIZE, label='Vortices')
        self.pavs=self.o2plot.axis.plot([1],[1],'vC8', markersize=MARKERSIZE, label='Anti Vortices')
        self.pfreevs = self.o2plot.axis.plot([1],[1],'oC2',markersize=MARKERSIZE , fillstyle='none', label = 'Free Vortices')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    def hide(self):
        self.pvs[0].set_data([],[])
        self.pavs[0].set_data([],[])
        self.pfreevs[0].set_data([],[])

    def update_data(self):
        sx,sy= self.o2plot.pv.shape[0:2]
        x, y = np.meshgrid(range(sx),range(sy))
        vlocx =x[np.where(self.o2plot.pv==1)]
        vlocy =y[np.where(self.o2plot.pv==1)]
        avlocx=x[np.where(self.o2plot.pv==-1)]
        avlocy=y[np.where(self.o2plot.pv==-1)]

        self.pvs[0].set_data(vlocx-0.5, vlocy-0.5)
        self.pavs[0].set_data(avlocx-0.5, avlocy-0.5)

        if len(self.o2plot.freevs)>0:
            freex = np.array(self.o2plot.freevs)[:,1]
            freey = np.array(self.o2plot.freevs)[:,0]
            self.pfreevs[0].set_data(freex-0.5, freey-0.5)
        else:
            self.pfreevs[0].set_data([],[])

    def update_plot(self):
        if self.active ==True:
            self.update_data()

    def toggle(self, event=None):
        if self.active == False:
            self.active =True
            self.update_plot()
        else:
            self.active = False
            self.hide()

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



