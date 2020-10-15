import graph_tool.all as gt
import numpy as np
    

from O2py.halfvortices import get_halfvortex_clusters
import O2py.bdrytrace as bdryt

def show_hvgraph(dofs, isinc, wn):
    clp, clm = get_halfvortex_clusters(dofs, wn, isinc)
    edgelist = list(zip(clp,clm))
    g = gt.Graph()
    names = g.add_edge_list(edgelist, hashed=True)
    ebet =gt.betweenness(g)[1]
    deg = g.degree_property_map("in")
    deg2 = g.degree_property_map("in")
    deg.a = 8*deg.a**0.4
    pos = gt.sfdp_layout(g)
    gt.graph_draw(g, pos=pos, edge_pen_width=1, vertex_size=deg, edge_end_marker="none",
                   edge_mid_marker="arrow", edge_marker_size = 6 , vertex_fill_color=deg2,
                       update_layout=True,sync=False, 
                  display_props=[names], display_props_size=20)



def show_hvbdrygraph(dofs, isinc, wn):
    loc, c1, c2 = bdryt.find_hv_fake_included(dofs, isinc, wn)
    elist = bdryt.construct_boundary_elist(dofs,isinc,wn)
    g = gt.Graph()

    vid ={pos:idx for idx,pos in enumerate(loc)}
    elistint = [(vid[e[0]], vid[e[1]]) for e in elist]

    g.add_edge_list(edge_list=elistint, hashed=False,)

    pos = g.new_vertex_property("vector<double>")
    for i,v in enumerate(g.vertices()):
        pos[v] = (loc[i][0], -loc[i][1])
 
    
    esigs = g.new_edge_property('int')
    esigs.a = np.array([-1 if (i%2) == 0 else +1 for i in range(len(elistint))])    


    ecids = g.new_edge_property('int')
    ecids.a = np.array([c1[i//2] if i %2 ==0 else c2[i//2] for i in range(len(elistint))])

    
    gt.graph_draw(g,
                  edge_color=ecids,
                  edge_text = esigs,
                  edge_pen_width=1, edge_end_marker="none",
                   edge_mid_marker="arrow", edge_marker_size = 6 ,
                       update_layout=True,sync=False, 
                  display_props=[pos], display_props_size=20)

def show_geometricvbdrygraph(dofs, isinc, wn):
    isincg = dofs.dot(wn) >0
    loc, c1, c2 = bdryt.find_hv_fake_included(dofs, isincg, wn)
    elist = bdryt.construct_boundary_elist(dofs,isincg,wn)
    g = gt.Graph()

    vid ={pos:idx for idx,pos in enumerate(loc)}
    elistint = [(vid[e[0]], vid[e[1]]) for e in elist]

    g.add_edge_list(edge_list=elistint, hashed=False,)

    pos = g.new_vertex_property("vector<double>")
    for i,v in enumerate(g.vertices()):
        pos[v] = (loc[i][0], -loc[i][1])


    ecids = g.new_edge_property('int')
    for i,e in enumerate(elistint):
        v1 = e[0]
        v2 = e[1]
        if i%2 ==0:
            ecids[g.edge(v1,v2)] = c1[i//2]
        else:
            ecids[g.edge(v1,v2)] = c2[i//2]
    

    gt.graph_draw(g,
                  pos=pos,
                  edge_color=ecids,
                  edge_pen_width=1, edge_end_marker="none",
                   edge_mid_marker="arrow", edge_marker_size = 6 ,
                       update_layout=True,sync=False, 
                  display_props=[pos], display_props_size=20)


