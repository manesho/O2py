""" An interactive visualization of the 2d O(2) or XY model with matplotlib"""


import numpy as np
import random
from numba import njit
#try:
#    from licpy.lic import runlic
#except:
#    print('line integral convolution not installed')
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import O2py.wolffcompiled as pw
import O2py.vortices as o2v
import O2py.hvgraph as o2hvg
from O2py.halfvortices import get_halfvortex_locations

MARKERSIZE = 10

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
        {}

        Special
        -------
        alt+g : show the halfvortex graph of the current configuration in a new window - experimental -

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
        self.clplot = ax.imshow(self.clo.transpose(), cmap = 'coolwarm', origin ='lower')


        self.layers = {'alt+q':QuiverLayer(self),
                       'alt+b':BoundaryLayer(self),
                       'alt+1':BondOrientationsLayer(self),
                       'alt+2':AllBoundariesLayer(self),
                       'alt+3':HalfVortexLayer(self),
                       'alt+4':OrientedCoverHLayer(self),
                       'alt+5':WindingNumbersLayer(self),
                       'alt+6':CrossingBondOrientaitonsLayer(self),
                       'alt+7':RectCrossingBondOrientaitonsLayer(self),
                       'alt+8':CoordinateCrossingLayer(self),
                       'alt+c':BoundaryContourLayer(self),
                       'alt+v':VortexLayer(self),
                       'alt+d':VortexChangeLayer(self)}
        
        layersdocstr = "\n".join(['\t {} : {}'.format(key, layer.description) for key,layer in self.layers.items()])
        print(self.manual.format(layersdocstr))

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

        self.clplot.set_array(licimage.transpose())
        self.clplot.set_clim(0,1)
        self.clplot.set_cmap('Blues')

    def plot_orientation(self):
        """Set the background to the angle of each dof with respect to the wolffplane."""
        phis = np.arctan2(self.dofs[:,:,0], self.dofs[:,:,1])
        phiwn = np.arctan2(self.wn[0], self.wn[1])
        phis = -np.mod(phiwn-phis+np.pi/2, 2*np.pi)+np.pi

        self.clplot.set_array(phis.transpose())
        self.clplot.set_clim(-np.pi,np.pi)
        self.clplot.set_cmap('twilight')

    def plot_isinc(self):
        """Set the background to a qualitative cluster visualization"""
        clcmap = cluster_random_cmap(self.isinc, mode='largecolored1')
        self.clplot.set_clim(0,np.max(self.isinc))
        self.clplot.set_array(self.isinc.transpose())
        self.clplot.set_cmap(clcmap)

    def update_cloplot(self):
        """Set the background to the mean cluster orientation with respect to the wolff normal"""
        self.clplot.set_array(self.clo.transpose())
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
            x = int(round(event.ydata))
            y = int(round(event.xdata))
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
            self.pv = o2v.plaquettevorticity_vec(self.dofs)
            self.update_background()
            self.update_layers()

        if event.key =='alt+g':
            o2hvg.show_hvbdrygraph(self.dofs, self.isinc, self.wn)

        if event.key =='alt+h':
            o2hvg.show_geometricvbdrygraph(self.dofs, self.isinc, self.wn)


###########################################################################################


###########################################################################################
class QuiverLayer:
    def __init__(self, o2plot):
        self.description = "Quiver Plot of all spins "
        self.o2plot = o2plot
        self.active = False
        self.handle = self.o2plot.axis.quiver(self.o2plot.dofs[:,:,0].transpose(),
                                              self.o2plot.dofs[:,:,1].transpose(),
                                              alpha = 0, scale_units='xy')
    def hide(self):
        self.handle.set_alpha(0)

    def update_data(self):
        self.handle.set_alpha(1.)
        self.handle.set_UVC(self.o2plot.dofs[:,:,0].transpose(),
                            self.o2plot.dofs[:,:,1].transpose())

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
        self.description = "Cluster Boundary for cluster under cursor  (alt+C to remove all boundaries)"
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
        sx,sy = self.o2plot.isinc.shape
        x = int(round(event.ydata))
        y = int(round(event.xdata))
        clid = self.o2plot.isinc[y,x]
        if clid not in self.clustercontour.keys():
            isincbig = np.repeat(np.repeat(self.o2plot.isinc, 20, axis=0), 20 , axis =1)
            self.clustercontour[clid]=plt.contour(isincbig.transpose()==clid, [1],
                                                  colors =['yellow'], linewidths=1,
                                                  extent=[-0.5, sx-0.5, -0.5,sy-0.5])
        else:
            for coll in self.clustercontour[clid].collections:
                coll.remove()
            del self.clustercontour[clid]

###########################################################################################
class HalfVortexLayer:
    def __init__(self, o2plot):
        self.description = "All half vortices(x) and their associated spins(up and down) "
        self.o2plot = o2plot
        self.active = False
        self.hvplot = self.o2plot.axis.plot([],[], 'C7x')
        self.hvpplot = self.o2plot.axis.plot([],[], 'C8^')
        self.hvmplot = self.o2plot.axis.plot([],[], 'C9v')
        self.textbox = self.o2plot.axis.text(0.05, 0.95, '',  transform=self.o2plot.axis.transAxes)

    def update_data(self):
        hvloc, hvploc, hvmloc = get_halfvortex_locations(self.o2plot.dofs, self.o2plot.wn, self.o2plot.isinc)
        self.hvplot[0].set_data(hvloc[:,0],hvloc[:,1])
        self.hvpplot[0].set_data(hvploc[:,0],hvploc[:,1])
        self.hvmplot[0].set_data(hvmloc[:,0],hvmloc[:,1])
        self.textbox.set_text('# half vortices  = {}'.format(hvloc.shape[0]))

            


    def hide(self):
        self.hvplot[0].set_data([],[])
        self.hvpplot[0].set_data([],[])
        self.hvmplot[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()


#############################################################################################
class WindingNumbersLayer:

    def __init__(self, o2plot):
        self.description = "List the spin winding numbers of each slice"
        self.o2plot = o2plot
        self.active= False
        self.handles =[self.o2plot.axis.text(self.o2plot.dofs.shape[1]+0.4,ci,'',ha="center", va="center") for ci in range(self.o2plot.dofs.shape[0])]

    def update_data(self):
        for i, w in enumerate(self.handles):
            wn =o2v.plaq_vort(self.o2plot.dofs[:,i])
            w.set_text(int(wn))

    def hide(self):
        for w in self.handles:
            w.set_text('')

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()

#############################################################################################
class OrientedCoverHLayer:
    def __init__(self, o2plot):
        self.description = "Oriented coverings of the east or west pole"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_p = self.o2plot.axis.plot([],[], 'C8',marker='3', ls='')
        #self.bondsplot_0 = self.o2plot.axis.plot([],[], 'C6o ')
        self.bondsplot_m = self.o2plot.axis.plot([],[], 'C9',marker='4', ls='')

    def update_data(self): 

        wn = self.o2plot.wn
        eastpole = np.array([wn[1], -wn[0]])
        ocsh = o2v.orientedcover_vec(self.o2plot.dofs,
                                     np.roll(self.o2plot.dofs, -1, axis=0),
                                     eastpole)
        #ocsv = o2v.orientedcover_vec(self.o2plot.dofs,
        #                             np.roll(self.o2plot.dofs, -1, axis=1),
        #                             self.o2plot.wn)

        #xs2plot=np.append( np.where(ocsh==1.)[0] + 0.5,)
        #                   np.where(ocsv==1.)[0] )
        #ys2plot=np.append( np.where(ocsh==1.)[1] ,)
        #                   np.where(ocsv==1.)[1] + 0.5)

        xs2plot = np.where(ocsh==1.)[0] + 0.5
        ys2plot =  np.where(ocsh==1.)[1] 
        self.bondsplot_p[0].set_data(xs2plot,ys2plot)

        #xs2plot=np.append( np.where(ocsh==-1)[0] + 0.5,)
        #                   np.where(ocsv==-1)[0] )
        #ys2plot=np.append( np.where(ocsh==-1)[1] ,
        #                   np.where(ocsv==-1)[1] + 0.5)

        xs2plot = np.where(ocsh==-1.)[0] + 0.5
        ys2plot =  np.where(ocsh==-1.)[1] 
 
        self.bondsplot_m[0].set_data(xs2plot,ys2plot )

        #xs2plot=np.append( np.where(ocsh==0)[0] + 0.5,
        #                   np.where(ocsv==0)[0] )
        #ys2plot=np.append( np.where(ocsh==0)[1] ,
        #                   np.where(ocsv==0)[1] + 0.5)

        #self.bondsplot_0[0].set_data(xs2plot,ys2plot )




    def hide(self):
            self.bondsplot_m[0].set_data([],[])
            self.bondsplot_0[0].set_data([],[])
            self.bondsplot_p[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()

##########################################################################################
class CrossingBondOrientaitonsLayer:
    def __init__(self, o2plot):
        self.description = "Orientation of all wolffplane crossing Links"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_p = self.o2plot.axis.plot([],[], 'C4+')
        self.bondsplot_m = self.o2plot.axis.plot([],[], 'C1_')

    def update_data(self):
        sx,sy = self.o2plot.isinc.shape
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,self.o2plot.wn )
        allorientations = [bdry_int_sig_vec(refconf, refconfshifted,self.o2plot.wn)
                               for refconfshifted in shiftinalldirs(refconf)]

        spinorientations = self.o2plot.dofs.dot(self.o2plot.wn) > 0 


        clbdrybools = [(spinorientations != shiftedspinorientations )
                           for shiftedspinorientations in  shiftinalldirs(spinorientations) ]
        x2plot = []
        y2plot = []
        os2plot= []

        for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
            x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
            y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
            os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
        self.bondsplot_p[0].set_data(x2plot[os2plot==1], y2plot[os2plot==1] )
        self.bondsplot_m[0].set_data(x2plot[os2plot==-1], y2plot[os2plot==-1] )


    def hide(self):
            self.bondsplot_m[0].set_data([],[])
            self.bondsplot_p[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()


##########################################################################################

class CoordinateCrossingLayer:
    def __init__(self, o2plot):
        self.description = "Orientation of all Links where the plane perpendicular to the wolffplane is crossed"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_xp = self.o2plot.axis.plot([],[], 'C5<')
        self.bondsplot_xm = self.o2plot.axis.plot([],[], 'C5>')
        self.bondsplot_yp = self.o2plot.axis.plot([],[], 'C5^')
        self.bondsplot_ym = self.o2plot.axis.plot([],[], 'C5v')

    def update_data(self):
        sx,sy = self.o2plot.isinc.shape
        thiswn = np.array([1,0])
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,thiswn)
        allorientations = [bdry_int_sig_vec(refconf, refconfshifted,thiswn)
                               for refconfshifted in shiftinalldirs(refconf)]

        spinorientations = self.o2plot.dofs.dot(thiswn) > 0 


        clbdrybools = [(spinorientations != shiftedspinorientations )
                           for shiftedspinorientations in  shiftinalldirs(spinorientations) ]
        x2plot = []
        y2plot = []
        os2plot= []

        for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
            x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
            y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
            os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
        self.bondsplot_yp[0].set_data(x2plot[os2plot==1], y2plot[os2plot==1] )
        self.bondsplot_ym[0].set_data(x2plot[os2plot==-1], y2plot[os2plot==-1] )

        thiswn = np.array([0,1])
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,thiswn)
        allorientations = [bdry_int_sig_vec(refconf, refconfshifted,thiswn)
                               for refconfshifted in shiftinalldirs(refconf)]

        spinorientations = self.o2plot.dofs.dot(thiswn) > 0 


        clbdrybools = [(spinorientations != shiftedspinorientations )
                           for shiftedspinorientations in  shiftinalldirs(spinorientations) ]
        x2plot = []
        y2plot = []
        os2plot= []

        for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
            x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
            y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
            os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
        self.bondsplot_xp[0].set_data(x2plot[os2plot==1], y2plot[os2plot==1] )
        self.bondsplot_xm[0].set_data(x2plot[os2plot==-1], y2plot[os2plot==-1] )



    def hide(self):
            self.bondsplot_xm[0].set_data([],[])
            self.bondsplot_xp[0].set_data([],[])
            self.bondsplot_ym[0].set_data([],[])
            self.bondsplot_yp[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()




##########################################################################################
class RectCrossingBondOrientaitonsLayer:
    def __init__(self, o2plot):
        self.description = "Orientation of all Links where the plane perpendicular to the wolffplane is crossed"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_p = self.o2plot.axis.plot([],[], 'C4+')
        self.bondsplot_m = self.o2plot.axis.plot([],[], 'C1_')

    def update_data(self):
        sx,sy = self.o2plot.isinc.shape
        thiswn = np.array([self.o2plot.wn[1], - self.o2plot.wn[0]])
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,thiswn)
        allorientations = [bdry_int_sig_vec(refconf, refconfshifted,thiswn)
                               for refconfshifted in shiftinalldirs(refconf)]

        spinorientations = self.o2plot.dofs.dot(thiswn) > 0 


        clbdrybools = [(spinorientations != shiftedspinorientations )
                           for shiftedspinorientations in  shiftinalldirs(spinorientations) ]
        x2plot = []
        y2plot = []
        os2plot= []

        for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
            x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
            y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
            os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
        self.bondsplot_p[0].set_data(x2plot[os2plot==1], y2plot[os2plot==1] )
        self.bondsplot_m[0].set_data(x2plot[os2plot==-1], y2plot[os2plot==-1] )


    def hide(self):
            self.bondsplot_m[0].set_data([],[])
            self.bondsplot_p[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()




###########################################################################################
class BondOrientationsLayer:
    def __init__(self, o2plot):
        self.description = "Orientation of all Links"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_p = self.o2plot.axis.plot([],[], 'C4+')
        self.bondsplot_m = self.o2plot.axis.plot([],[], 'C1_')

    def update_data(self):
        sx,sy = self.o2plot.isinc.shape
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,self.o2plot.wn )

        orientationsh = bdry_int_sig_vec(refconf, np.roll(refconf, -1, axis=0),self.o2plot.wn)
        orientationsv = bdry_int_sig_vec(refconf, np.roll(refconf, -1, axis=1),self.o2plot.wn)

        xs2plot=np.append( np.where(orientationsh==1.)[0] + 0.5,
                           np.where(orientationsv==1.)[0] )
        ys2plot=np.append( np.where(orientationsh==1.)[1] ,
                           np.where(orientationsv==1.)[1] + 0.5)
        self.bondsplot_p[0].set_data(xs2plot,ys2plot)
        xs2plot=np.append( np.where(orientationsh==-1)[0] + 0.5,
                           np.where(orientationsv==-1)[0] )
        ys2plot=np.append( np.where(orientationsh==-1)[1] ,
                           np.where(orientationsv==-1)[1] + 0.5)

        self.bondsplot_m[0].set_data(xs2plot,ys2plot )


    def hide(self):
            self.bondsplot_m[0].set_data([],[])
            self.bondsplot_p[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()


###########################################################################################
class AllBoundariesLayer:
    def __init__(self, o2plot):
        self.description = "Orientation of all cluster boundary Links"
        self.o2plot = o2plot
        self.active = False
        self.bondsplot_p = self.o2plot.axis.plot([],[], 'C4+')
        self.bondsplot_m = self.o2plot.axis.plot([],[], 'C1_')

    def update_data(self):
        sx,sy = self.o2plot.isinc.shape
        refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,self.o2plot.wn )
        allorientations = [bdry_int_sig_vec(refconf, refconfshifted,self.o2plot.wn)
                               for refconfshifted in shiftinalldirs(refconf)]
        clbdrybools = [(self.o2plot.isinc != shiftedisinc )
                           for shiftedisinc in shiftinalldirs(self.o2plot.isinc) ]
        x2plot = []
        y2plot = []
        os2plot= []

        for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
            x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
            y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
            os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
        self.bondsplot_p[0].set_data(x2plot[os2plot==1], y2plot[os2plot==1] )
        self.bondsplot_m[0].set_data(x2plot[os2plot==-1], y2plot[os2plot==-1] )


    def hide(self):
            self.bondsplot_m[0].set_data([],[])
            self.bondsplot_p[0].set_data([],[])

    def update_plot(self):
        if self.active:
            self.update_data()

    def toggle(self, event=None):
        if self.active == True:
            self.active = False
            self.hide()
        else:
            self.active = True 
            self.update_data()


###########################################################################################

class BoundaryLayer:
    def __init__(self, o2plot):
        self.description = "Boundary Layer"
        self.o2plot = o2plot
        self.boundaryplots_p={}
        self.boundaryplots_m={}

    def update_plot(self):
        return

    def toggle(self, event):


        x = int(round(event.ydata))
        y = int(round(event.xdata))
        clid = self.o2plot.isinc[y,x]

        if clid not in self.boundaryplots_p.keys():
            refconf = get_reference_configuration(self.o2plot.dofs, self.o2plot.isinc,self.o2plot.wn )
            allorientations = [bdry_int_sig_vec(refconf, refconfshifted,self.o2plot.wn)
                               for refconfshifted in shiftinalldirs(refconf)]


            clbdrybools = [(self.o2plot.isinc ==clid) & (shiftedisinc != clid)
                           for shiftedisinc in shiftinalldirs(self.o2plot.isinc) ]
            x2plot = []
            y2plot = []
            os2plot= []

            for bdrybool, orientations, xshift, yshift in zip( clbdrybools,allorientations,
                                                               [0.5,-0.5,0.,0.],[0.,0.,0.5,-0.5]):
                x2plot = np.append(x2plot,np.where(bdrybool)[0]+xshift)
                y2plot = np.append(y2plot,np.where(bdrybool)[1]+yshift)
                os2plot= np.append(os2plot, orientations[bdrybool])

            #coordinates have to be exchanged due to the funny coordinate system of imshow
            self.boundaryplots_p[clid] = self.o2plot.axis.plot(x2plot[os2plot==1], y2plot[os2plot==1], 'C4+' )
            self.boundaryplots_m[clid] = self.o2plot.axis.plot(x2plot[os2plot==-1],y2plot[os2plot==-1], 'C1_' )

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
        self.description = "Change in Vorticety due to last cluster flip"
        self.o2plot = o2plot
        self.active =False
        self.pvs = self.o2plot.axis.plot([],[],'^C4', markersize=MARKERSIZE)
        self.pavs = self.o2plot.axis.plot([],[],'vC6', markersize=MARKERSIZE )

    def update_data(self):
        sx,sy= self.o2plot.pv.shape[0:2]
        x, y = np.meshgrid(range(sx),range(sy))
        vlocx =x[np.where((self.o2plot.pv-self.o2plot.pvprev) ==1)]
        vlocy =y[np.where((self.o2plot.pv-self.o2plot.pvprev) ==1)]
        avlocx=x[np.where((self.o2plot.pv-self.o2plot.pvprev) ==-1)]
        avlocy=y[np.where((self.o2plot.pv-self.o2plot.pvprev) ==-1)]

        self.pvs[0].set_data(vlocy-0.5, vlocx-0.5)
        self.pavs[0].set_data(avlocy-0.5, avlocx-0.5)


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
        self.description = "Vortices"
        self.o2plot = o2plot
        self.active =False
        self.pvs = self.o2plot.axis.plot([],[],'^C5', markersize=MARKERSIZE, label='Vortices')
        self.pavs=self.o2plot.axis.plot([],[],'vC8', markersize=MARKERSIZE, label='Anti Vortices')
        self.pfreevs = self.o2plot.axis.plot([],[],'oC2',markersize=MARKERSIZE , fillstyle='none', label = 'Free Vortices')
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

        self.pvs[0].set_data(vlocy-0.5, vlocx-0.5)
        self.pavs[0].set_data(avlocy-0.5, avlocx-0.5)

        if len(self.o2plot.freevs)>0:
            freex = np.array(self.o2plot.freevs)[:,0]
            freey = np.array(self.o2plot.freevs)[:,1]
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




def shiftinalldirs(array):
    """returns a list of array shifted by +-1 in x an d y"""
    return [np.roll(array, -1 ,axis = 0),
                np.roll(array, 1, axis =0),
                np.roll(array, -1 ,axis = 1),
                np.roll(array, 1, axis =1)]

def bdry_int_sig(v1,v2,wn):
    """Returns the boundary interpolation signature between v1 and v2, wn is the wolff normal"""
    return np.sign(np.linalg.det([wn,v1])) if v1.dot(wn)**2 < v2.dot(wn)**2 else np.sign(np.linalg.det([wn,v2]))




bdry_int_sig_vec = np.vectorize(bdry_int_sig, signature='(2),(2),(2)->()')
