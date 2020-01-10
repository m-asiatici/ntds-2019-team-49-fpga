import os
import itertools
import networkx as nx
import json
from networkx.algorithms import bipartite

###########################################################################
#NOTES: 1) It could be nice to test the hypothesis under different channel
#          widths, and observe how the problems start to appear when it
#          approaches Wmin.
#       2) We may be forced to do some sparsification of the bipartite
#          graph, as it could be too large (even for small benchamrks).
#          A reasonable option could be to select kU heaviest edges leading
#          from each net node, where U is the number of tracks used in the
#          final routing, and k a constant from [1, inf). For the final
#          clustering, etc, the size should not be a problem, as we will
#          be operating on the projection that depends solely (although
#          not when it comes to sparsity and SVD engines might depend on it)
#          on the number of nets in the circuit, and not the number of tracks
#          trapped within the bounding boxes. WARNING: No, this is not so simple,
#          as if k is too small, we may be selecting mutually exclusive tracks that
#          would never be able to form a path. It is a reasonable starting point,
#          though. Setting k to the channel width used for routing is fairly conservative.
##########################################################################

CHAN_W = 42
CONGESTION_THR = 0.75
#If less than CHAN_W * CONGESTION_THR nets are contending for use of a particular
#track, we do not have to consider it in the conflicting weight computation.

##########################################################################
def preprocess_net_for_tg_reading(lines):
    #Eliminates the irregularities in the net file that would cause
    #"read_placement_timing_graph_full" to crash (it is whitespace sensitive).

    new_lines = []

    #Make the port maps one-line.
    catch = 0
    lcnt = -1
    for line in lines:
        lcnt += 1
        if not catch:
            new_lines.append(line)
        else:
            new_lines[-1] += line
        if "<port" in line:
            catch = 1
        if "</port>" in line:
            catch = 0

    return new_lines
##########################################################################

##########################################################################
def read_placement_timing_graph_full(tg_file, net_file, strip = True):
    #Reads the complete post-placement timing graph, along with all edges
    #and their delays. If "strip" is true, all nonblif nodes will be removed
    #and the missing paths modeled by edges of the appropriate delay.

    net = open(net_file, "r")
    lines = net.readlines()
    net.close()
    lines = preprocess_net_for_tg_reading(lines)

    net_indices = {"ff" : {}, "lut" : {}, "in" : {}, "out" : {}}

    port_inits = ["<inputs>", "<outputs>", "<clocks>"]
    port_ends = ["</inputs>", "</outputs>", "</clocks>"]
    state = "idle"
    pins = []
    open_blocks = 0
    block_name_stack = []
    lcnt = 0
    blif_name = ""
    for line in lines[1:]:
        lcnt += 1
        if state == "idle":
            if "<block" in line:
                state = "port_init"
                block_name = line.split("instance=\"", 1)[1].split("\"")[0]
                block_ind = int(block_name.split('[', 1)[1].split(']', 1)[0])
                net_indices["ff"].update({block_ind : []})
                net_indices["lut"].update({block_ind : []})
                net_indices["in"].update({block_ind : []})
                net_indices["out"].update({block_ind : []})
                block_name_stack.append(block_name)
                open_blocks += 1
        elif state == "port_init":
            if line.strip() in port_inits:
                state = "port"
            elif "<block" in line:
                if not "/>" in line:
                    blif_name = line.split("name=\"", 1)[1].split("\"")[0]
                    block_name += "___" + line.split("instance=\"", 1)[1].split("\"")[0]
                    block_name_stack.append(block_name)
                    open_blocks += 1
            elif "</block" in line:
                open_blocks -= 1
                block_name_stack.pop(-1)
                if open_blocks == 0:
                    state = "idle"
                    continue
                block_name = block_name_stack[-1]
        elif state == "port":
            if line.strip() in port_ends:
                state = "port_init"
                continue
            port_name = line.split("name=\"", 1)[1].split("\"")[0]
            assignments = line.split('>', 1)[1].split('<', 1)[0].split()
            added = 0
            for i in xrange(0, len(assignments)):
                if assignments[i] != "open":
                    pins.append(block_name + '.' + port_name + '[' + str(i) + ']')
                    added = 1
            if added:
                if pins[-1].rsplit("___", 1)[-1] == "lut[0].out[0]":
                    #FIXME: Make this more architecture independent.
                    net_indices["lut"][block_ind].append(blif_name)
                elif "name=\"Q" in line:
                    net_indices["ff"][block_ind].append(blif_name)
                elif "name=\"out\">repeater_" in line:
                    #We add repeaters to the same cathegory as LUTs because they
                    #have the same annotation in the timing graph, a nd we want to
                    #maintain the order.
                    net_indices["lut"][block_ind].append(blif_name)
                elif "name=\"outpad\">io" in line:
                    net_indices["out"][block_ind].append(blif_name)
                elif "name=\"inpad" in line and not "inpad\">inpad" in line:
                    net_indices["in"][block_ind].append(blif_name)

    blif_nodes = {}
    tg = open(tg_file, "r")
    lines = tg.readlines()
    tg.close()

    nodes = []
    edges = []
    coords = []
    reading = 0
    lcnt = -1
    types = set([])
    for line in lines:
        lcnt += 1
        if line.isspace():
            continue
        if "TN" in line:
            u = int(line.split()[0].split('(')[0])
            nodes.append(u)
            try:
                x = int(line.split()[0].split('(')[1].split(',')[0])
                y = int(line.split()[0].split('(')[1].split(',')[1][:-1])
                coords.append((x, y))
            except:
                pass
            block_ind = int(line.split()[3])
            if "PRIMITIVE_OP" in line or "CONSTANT_GEN" in line :
                if net_indices["lut"].get(block_ind, None):
                    blif_nodes.update({net_indices["lut"][block_ind].pop(0) : u})
            elif "INPAD_SOURCE" in line:
                if net_indices["in"].get(block_ind, None):
                    blif_nodes.update({net_indices["in"][block_ind].pop(0) : u})
            elif "OUTPAD_SINK" in line:
                if net_indices["out"].get(block_ind, None):
                    blif_nodes.update({net_indices["out"][block_ind].pop(0) : u})
            elif "FF_SINK" in line:
                if net_indices["ff"].get(block_ind, None):
                    blif_nodes.update({net_indices["ff"][block_ind][0] + ".d" : u})
            elif "FF_SOURCE" in line:
                if net_indices["ff"].get(block_ind, None):
                    blif_nodes.update({net_indices["ff"][block_ind].pop(0) + ".q" : u})
                
            if "FF_SINK" in  line or "FF_CLOCK" in line or "OUTPAD_SINK" in  line:
                #These pins have no outgoing edges
                #(in fact, the SINK/SOURCE pair opens the FFs)
                reading = 0
                continue
            v = int(line.split()[-2])
            td = float(line.split()[-1])
            if abs(td) > 1e1:
                #This is a constant generator. Set its delay to zero.
                td = 0
            reading = 1
            edges.append((u, v, {"td" : td}))
        elif "num_tnode_levels" in line:
            break
        elif reading:
            v = int(line.split()[0])
            td = float(line.split()[-1])
            edges.append((u, v, {"td" : td}))

    tg = nx.DiGraph()
    tg.add_edges_from(edges) 

    cp_delay = annotate_timing(tg)

    if coords:
        #Annotate the nodes with placement coordinates.
        for i in xrange(0, len(nodes)):
            tg.node[nodes[i]]["coords"] = coords[i]
    
    relabeling_dict = {}
    for blif_node in blif_nodes:
        u = nodes[blif_nodes[blif_node]]
        tg.node[u]["is_blif"] = True
        relabeling_dict.update({u : blif_node})

    tg = nx.relabel_nodes(tg, relabeling_dict)
    if strip:
        tg = remove_non_blif_nodes(tg, blif_nodes, cp_delay)
 
    return tg, cp_delay
##########################################################################

##########################################################################
def remove_non_blif_nodes(tg, blif_nodes, cpd):
    #Removes the non-blif nodes from the timing graph and adds direct
    #edges between the remaining ones connected by a path originally
    #passing only through non-blif nodes.

    new_tg = nx.DiGraph()
    new_tg.add_nodes_from(blif_nodes.keys())

    visited = set([])
    for blif_node in blif_nodes:
        new_tg.node[blif_node]["tarr"] = tg.node[blif_node]["tarr"]
        new_tg.node[blif_node]["treq"] = tg.node[blif_node]["treq"]
        new_tg.node[blif_node]["coords"] = tg.node[blif_node]["coords"]
        stack = [blif_node]
        td_stack = [0]
        slack_stack = [-1]
        while stack:
            u = stack.pop(-1)
            visited.add(u)
            td = td_stack.pop(-1)
            slack = slack_stack.pop(-1)
            for v in tg[u]:
                edge_td = tg[u][v]["td"]
                edge_slack = tg[u][v]["slack"]
                if blif_nodes.get(v, None):
                    td += edge_td
                    slack = max(slack, edge_slack)
                    new_tg.add_edge(blif_node, v, td = td, slack = slack,\
                                    crit = 1 - slack / cpd)
                else:
                    stack.append(v)
                    td_stack.append(td + edge_td)
                    slack_stack.append(max(slack, edge_slack))

    add_non_blif_periphery(tg, new_tg, blif_nodes, visited)

    return new_tg    
##########################################################################

##########################################################################
def add_non_blif_periphery(tg, new_tg, blif_nodes, visited):
    #Adds the periphery so that the nodes which lost the most critical
    #but not the last remaining child can be properly constrained by the dummy
    #outputs, and likewise for parents.

    for u in blif_nodes:
        for child in tg[u]:
            if blif_nodes.get(child, None) or child in visited:
                continue
            new_tg.add_edge(u, "nonblif_" + str(child), td = tg[u][child]["td"],\
                            slack = tg[u][child]["slack"])
            new_tg.node["nonblif_" + str(child)]["tarr"] = tg.node[child]["tarr"]
            new_tg.node["nonblif_" + str(child)]["treq"] = tg.node[child]["treq"]
        for parent in tg.pred[u]:
            if blif_nodes.get(parent, None) or parent in visited:
                continue
            new_tg.add_edge("nonblif_" + str(parent), u, td = tg[parent][u]["td"],\
                            slack = tg[parent][u]["slack"])
            new_tg.node["nonblif_" + str(parent)]["tarr"] = tg.node[parent]["tarr"]
            new_tg.node["nonblif_" + str(parent)]["treq"] = tg.node[parent]["treq"]
##########################################################################

##########################################################################
def annotate_timing(tg):
    #Annotates the graph nodes with arrival and required times and assigns
    #slacks to edges.

    cp_delay = nx.algorithms.dag.dag_longest_path_length(tg, weight = "td")
    nodes = list(nx.algorithms.dag.topological_sort(tg))

    for v in nodes:
        tg.node[v]["tarr"] = max([tg.node[p]["tarr"] + tg[p][v]["td"] for p in tg.pred[v]]\
                                 + [0])

    for v in reversed(nodes):
        tg.node[v]["treq"] = min([tg.node[c]["treq"] - tg[v][c]["td"] for c in tg[v]]\
                                 + [cp_delay])
        for p in tg.pred[v]:
            tg[p][v]["slack"] = tg.node[v]["treq"] - tg.node[p]["tarr"] - tg[p][v]["td"]
            #Clip the slack to a range between 1e-15 and 1e1. Anything less than
            #1fs is basically rounding noise, and anything above 1s belongs to
            #constant generators.
            if abs(tg[p][v]["slack"]) < 1e-15:
                tg[p][v]["slack"] = 0
            if abs(tg[p][v]["slack"]) > 1e1:
                tg[p][v]["slack"] = cp_delay
            tg[p][v]["crit"] = 1 - tg[p][v]["slack"] / cp_delay

    return cp_delay
##########################################################################

##########################################################################
def conv_rr_to_nx(rr_file):
    #Parses a routing-resource graph file and generates a networkx graph.

    rr_txt = open(rr_file, "r")
    lines = rr_txt.readlines()
    rr_txt.close()

    rr = nx.DiGraph()
    tracks = set([])
    for line in lines:
        if "Node:" in line:
            u = int(line.split()[1])
            t = line.split()[2]
            if "CHAN" in t:
                tracks.add(u)
            xmin = int(line.split()[3][1:-1])
            ymin = int(line.split()[4][:-1])

            if "CHAN" in t and line.split()[5] == "to":
                #Channels at the chip boundary may have
                #tracks shorter than L, due to staggering.
                xmax = int(line.split()[6][1:-1])
                ymax = int(line.split()[7][:-1])
            else:
                xmax = xmin
                ymax = ymin
            if not rr.has_node(u):
                rr.add_node(u)
            rr.node[u]["type"] = t
            rr.node[u]["xmin"] = xmin
            rr.node[u]["xmax"] = xmax
            rr.node[u]["ymin"] = ymin
            rr.node[u]["ymax"] = ymax
        elif "edge(s):" in line:
            if line[0] == '0':
                continue
            targets = line.split()[2:]
            for v in targets:
                rr.add_edge(u, int(v))

    return rr, tracks
##########################################################################

##########################################################################
def get_nets(route_file, tg, rr, coords_file):
    #Dictionary indexed by net names, holding tuples of net-pin rr-nodes,
    #with the source being the first one.

    nets = {}
    route = open(route_file, "r")
    lines = route.readlines()
    route.close()

    for line in lines:
        if "Net" in line:
            name = line.split()[2][1:-1]
            no = int(line.split()[1])
            nets.update({name : {"no" : no, "pins" : [], "crit" : [],\
                                 "wl" : 0, "coords" : []}})
        elif "SOURCE" in line or "SINK" in line:
            u = int(line.split()[1])
            pinx = rr.node[u]["xmin"]
            piny = rr.node[u]["ymin"]
            if "SOURCE" in line:
                nets[name]["coords"] = (pinx, piny)
            if not nets[name]["pins"]:
                fanout = tg[name]
                #Benchmarks that we use are purely combinational, so no FF
                #naming issues will arise.
                nets[name]["crit"].append(0)
            else:
                found = 0
                for v in fanout:
                    if (pinx, piny) == tg.node[v]["coords"]:
                        found = 1
                        crit = tg[name][v]["crit"]
                        break
                if not found:
                    print "Criticality of", name, "can not be determined."
                    exit(-1)
                    continue
                nets[name]["crit"].append(crit)
            nets[name]["pins"].append(u)
                        
        elif "CHAN" in line:
            nets[name]["wl"] += 1

    txt = ""
    for net in nets:
        txt += net + ' ' + str(nets[net]["coords"][0])\
             + ' ' + str(nets[net]["coords"][1]) + "\n"

    coords_out = open(coords_file, "w")
    coords_out.write(txt)
    coords_out.close()

    return nets
##########################################################################

##########################################################################
def get_bounding_box(pins, rr):
    #Computes the bounding box of a net represented by its pins (rr-nodes).

    xmin = float('inf')
    ymin = float('inf')
    xmax = -1
    ymax = -1
    for p in pins:
        #Pins have a single-point location.
        x = rr.node[p]["xmin"]
        y = rr.node[p]["ymin"]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    return xmin, xmax, ymin, ymax
##########################################################################

##########################################################################
def extract_bb_subgraph(bb, rr, bb_factor = 3):
    #Extracts the subgraph of the routing-resource graph that overlaps with
    #the given bounding box. "bb_factor" is a VPR's parameter telling how
    #muxh a route can go outside the bounding box. The default value in VPR
    #is 3, and this is the one that we will use.

    xmin, xmax, ymin, ymax = bb
    xmin -= bb_factor
    xmax += bb_factor
    ymin -= bb_factor
    ymax += bb_factor

    overlapped = []
    overlapped_tracks = []

    for u, attrs in rr.nodes(data = True):
        uxmin = attrs["xmin"]
        uxmax = attrs["xmax"]
        uymin = attrs["ymin"]
        uymax = attrs["ymax"]
        if xmin <= uxmin and xmax >= uxmax and ymin <= uymin and ymax >= uymax:
            overlapped.append(u)
            if "CHAN" in attrs["type"]:
                overlapped_tracks.append(u)

    subg = rr.subgraph(overlapped)

    return subg, overlapped_tracks
##########################################################################

##########################################################################
def compute_adjacency(net, rr, nets):
    #Returns a list of neighbors of "net", along with the corresponding 
    #edge weights.

    pins = nets[net]["pins"]
    net_crit = max(nets[net]["crit"])
    bb = get_bounding_box(pins, rr)
    subg, overlapped_tracks = extract_bb_subgraph(bb, rr)

    src = pins[0]
    trgts = pins[1:]
    edges = []

    #We must exclude the source and its immediate children (OPINs) when searching 
    #for target paths, as we do not want the path to simply traverse back all the
    #way to the source.

    subg_cp = subg.copy()
    opins = list(subg_cp[src])
    for o in opins:
        subg_cp.remove_node(o)
    subg_cp.remove_node(src)

    for track in overlapped_tracks:
        if not nx.has_path(subg, src, track):
            continue
        sp_beg = nx.shortest_path(subg, src, track)
        sp_beg_len = 0
        for u in sp_beg:
            if "CHAN" in subg.node[u]["type"]:
                sp_beg_len += 1
        connected = True
        total_len = 0
        tcnt = 0
        for t in trgts:
            tcnt += 1
            if not nx.has_path(subg_cp, track, t):
                connected  = False
                break
            sp_end = nx.shortest_path(subg_cp, track, t)[1:]
            sp_end_len = 0
            for u in sp_end:
                if "CHAN" in subg_cp.node[u]["type"]:
                    sp_end_len += 1
            crit = nets[net]["crit"][tcnt] / net_crit if net_crit else 0
            total_len += crit * (sp_beg_len + sp_end_len)
        if not connected or not total_len:
            continue
        edges.append((track, 1.0 / total_len))

    return sorted(edges, key = lambda e : e[1], reverse = True)
##########################################################################

##########################################################################
def sparsify_bipartite(net, nets, edges, k = -1):
    #Keeps only a subset of edges incident to "net".
    #FIXME: A dummy function for now.

    if k >= 0:
        return edges[:k * nets[net]["wl"]]
    return edges
##########################################################################

##########################################################################
def build_bipartite_graph(rr, tracks, nets):
    #Constructs the bipartite graph modeling the net-track preference.

    bp = nx.Graph()
    bp.add_nodes_from(nets.keys(), bipartite = 0)
    bp.add_nodes_from(tracks, bipartite = 1)
    for net in nets:
        bp.node[net]["crit"] = max(nets[net]["crit"])
        edges = sparsify_bipartite(net, nets, compute_adjacency(net, rr, nets), CHAN_W)
        #print net, len(edges)
        for e in edges:
            bp.add_edge(net, e[0], w = e[1])

    return bp
##########################################################################

##########################################################################
def generate_projection(net_track_graph, nets, skeleton = None):
    #Generates the projection of the net-representing node-subset of the
    #bipartite graph representing a net's preference for chosing a particular
    #track. If "skeleton" is passed, only edge weights are computed, while
    #the structure is assumed to be given by the skeleton.

    if skeleton:
        projection = skeleton.copy()
    else:
        projection = bipartite.projected_graph(net_track_graph, nets.keys())

    edge_removal_list = []
    for u, v in projection.edges():
        u_adj = net_track_graph[u]
        v_adj = net_track_graph[v]
        cong = [t for t in u_adj if t in v_adj\
               if net_track_graph.degree(t) > CHAN_W * CONGESTION_THR]
        if not cong:
            edge_removal_list.append((u, v))
            continue

        u_uncong = [t for t in u_adj if not t in cong]
        v_uncong = [t for t in v_adj if not t in cong]

        u_cong_avg_w = sum([net_track_graph[u][t]['w'] for t in cong])\
                       / len(cong)
        u_uncong_avg_w = sum([net_track_graph[u][t]['w'] for t in u_uncong])\
                       / len(u_uncong) if u_uncong else 0
        delta_u_w = u_cong_avg_w - u_uncong_avg_w
        
        v_cong_avg_w = sum([net_track_graph[v][t]['w'] for t in cong])\
                       / len(cong)
        v_uncong_avg_w = sum([net_track_graph[v][t]['w'] for t in v_uncong])\
                       / len(v_uncong) if v_uncong else 0
        delta_v_w = v_cong_avg_w - v_uncong_avg_w

        #w = max(delta_u_w, delta_v_w)
        #NOTE: We probably need "min" here in fact, because as soon as one of
        #the nets is willing to give up on its resources, the problem is resolved.
        w = min(delta_u_w, delta_v_w)

        if w < 0:
            edge_removal_list.append((u, v))
        else:
            projection[u][v]['w'] = w

    for u, v in edge_removal_list:
        projection.remove_edge(u, v)

    return projection
##########################################################################

##########################################################################
def calc_cluster_weight(projection, cls, with_nodes = False):
    #Computes the weight of the given cluster, either as the average of its
    #edge weights, or the average of products of the edge weigths and their
    #endpoint weights.

    weight = 0
    card = 0
    for u in cls:
        for v in cls:
            if projection.has_edge(u, v):
                e_w = projection[u][v]['w']
                if with_nodes: 
                    u_w = projection.node[u]["crit"]
                    v_w = projection.node[v]["crit"]
                    #print cluster, u_w, v_w, e_w
                    weight += u_w * v_w * e_w
                else:
                    weight += e_w
                card += 1

    return weight / card if card else 0
##########################################################################

##########################################################################
def brute_force_heaviest_cluster(projection, min_crit = 0.9, size = 2):
    #Performs brute-force enumeration of node subsets of size "size" and
    #picks the heaviset one, where the weight is the average of the edge-endpoint
    #weight products, for all eadges.

    nodes = list(projection.nodes())

    max_weight = 0
    heaviest_cluster = None

    for cluster in itertools.combinations(nodes, size):
        found = 0
        for u in cluster:
            if projection.node[u]["crit"] < min_crit:
                found = 1
                break
        if found:
            continue
        weight = calc_cluster_weight(projection, cluster, False)
        if weight > max_weight:
            max_weight = weight
            heaviest_cluster = cluster

    final_weight = calc_cluster_weight(projection, heaviest_cluster, True)\
                   if heaviest_cluster else 0
    return heaviest_cluster, final_weight
##########################################################################

##########################################################################
def dump_projection(graph, out_file, gephi = True):
    #Dumps a projection graph as a simple list of weighted edges. The first
    #line contains the node and the edge set cardinality. Then there are |V|
    #lines with the node counts and their criticalities. Finally the edges
    #are dumped using the counts.
   
    txt = str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + "\n"
    nodes = {}
    for u in sorted(graph.nodes()):
        txt += str(len(nodes)) + ' ' + u + " %.4g"%graph.node[u].get("crit", -1) + "\n"
        nodes.update({u : len(nodes)})

    for u, v, attrs in graph.edges(data = True):
        txt += str(nodes[u]) + ' ' + str(nodes[v]) + " %.4g"%attrs.get('w', -1) + "\n"
    
    dump = open(out_file, "w")
    dump.write(txt[:-1])
    dump.close()

    if gephi:
        #Also dump it in the gephi format.
        for u, v, attrs in graph.edges(data = True):
            graph[u][v]["Weight"] = attrs.get('w', -1)
        nx.write_gexf(graph, out_file.rsplit('.', 1)[0] + ".gexf")
##########################################################################

##########################################################################
def load_projection(in_file):
    #Loads a previously dumped projection graph.

    dump = open(in_file, "r")
    lines = dump.readlines()
    dump.close()

    node_no = int(lines[0].split()[0])
    edge_no = int(lines[0].split()[1])

    nodes = {}
    graph = nx.Graph()
    for lptr in xrange(1, node_no + 1):
        num_id = int(lines[lptr].split()[0])
        str_id = lines[lptr].split()[1]
        crit = float(lines[lptr].split()[2])
        nodes.update({num_id : str_id})
        graph.add_node(str_id, crit = crit)

    lptr += 1
    for lptr in xrange(lptr, node_no + 1 + edge_no):
        u = int(lines[lptr].split()[0])
        v = int(lines[lptr].split()[1])
        w = float(lines[lptr].split()[2])
        graph.add_edge(nodes[u], nodes[v], w = w)

    return graph
##########################################################################

##########################################################################
def dump_net_track(graph, out_file, gephi = True):
    #Dumps a net-track graph as a simple list of weighted edges. The first
    #line contains the two node subset and the edge set cardinality.
    #Then there are |Vn| lines with the net subset node counts and their
    #criticalities, followed by the |Vt| lines with the nodes of the track
    #subset. Finally the edges are dumped using the counts.
   
    nets = [u for u, attrs in graph.nodes(data = True) if attrs["bipartite"] == 0]
    tracks = [u for u, attrs in graph.nodes(data = True) if attrs["bipartite"] == 1]

    txt = str(len(nets)) + ' ' + str(len(tracks)) + ' '\
        + str(graph.number_of_edges()) + "\n"

    net_rlb = {}
    for u in sorted(nets):
        txt += str(len(net_rlb)) + ' ' + u + " %.4g"%graph.node[u].get("crit", -1) + "\n"
        net_rlb.update({u : len(net_rlb)})

    track_rlb = {}
    for u in sorted(tracks):
        txt += str(len(track_rlb)) + ' ' + str(u) + "\n"
        track_rlb.update({u : len(track_rlb)})

    for u in sorted(nets):
        for v in graph[u]:
            txt += str(net_rlb[u]) + ' ' + str(track_rlb[v])\
                 + " %.4g"%graph[u][v].get('w', -1) + "\n"
    
    dump = open(out_file, "w")
    dump.write(txt[:-1])
    dump.close()

    if gephi:
        #Also dump it in the gephi format.
        for u, v, attrs in graph.edges(data = True):
            graph[u][v]["Weight"] = attrs.get('w', -1)
        nx.write_gexf(graph, out_file.rsplit('.', 1)[0] + ".gexf")
##########################################################################

##########################################################################
def load_net_track(in_file):
    #Loads a previously dumped net-track graph.

    dump = open(in_file, "r")
    lines = dump.readlines()
    dump.close()

    net_no = int(lines[0].split()[0])
    track_no = int(lines[0].split()[1])
    edge_no = int(lines[0].split()[2])

    nets = {}
    graph = nx.Graph()
    for lptr in xrange(1, net_no + 1):
        num_id = int(lines[lptr].split()[0])
        str_id = lines[lptr].split()[1]
        crit = float(lines[lptr].split()[2])
        nets.update({num_id : str_id})
        graph.add_node(str_id, crit = crit, bipartite = 0)

    tracks = {}
    for lptr in xrange(lptr + 1, net_no + 1 + track_no):
        num_id = int(lines[lptr].split()[0])
        str_id = lines[lptr].split()[1]
        tracks.update({num_id : str_id})
        graph.add_node(str_id, bipartite = 1)

    lptr += 1
    for lptr in xrange(lptr, net_no + 1 + track_no + edge_no):
        u = int(lines[lptr].split()[0])
        v = int(lines[lptr].split()[1])
        w = float(lines[lptr].split()[2])
        graph.add_edge(nets[u], tracks[v], w = w)

    return graph
##########################################################################

##########################################################################
def normalize_net_track(G):
    #Normalizes the edge weights of the net-track preference graph, on a 
    #per-net basis. If this is not done, nets with shorter wirelength 
    #(or/and smaller fanout) will be favored (i.e., have heavier incident
    #edges).

    for u, attrs in G.nodes(data = True):
        if attrs["bipartite"] == 0:
            max_w = max([G[u][v]['w'] for v in G[u]])
            for v in G[u]:
                G[u][v]['w'] /= max_w 
##########################################################################

##########################################################################
def generate_graphs(vpr_file_dir):
    #Generates all the graphs and dumps them in the same directory where the
    #VPR files are stored. It is assumed that the VPR directory is named as
    #benchmark.channel_width.

    benchmark_name = vpr_file_dir.rsplit('.', 1)[0]
    chan_w = int(vpr_file_dir.rsplit('.', 1)[1])
    global CHAN_W
    CHAN_W = chan_w

    #Input files:
    tg_file = vpr_file_dir + "/final_placement_timing_graph.echo"
    net_file = vpr_file_dir + '/' + benchmark_name + ".net"
    rr_file = vpr_file_dir + "/rr_graph.echo"
    route_file = vpr_file_dir + '/' + benchmark_name + ".route"

    #Output files:
    net_track_file = vpr_file_dir + "/net_track.dump"
    projection_file = vpr_file_dir + "/projection.dump"
    coords_file = vpr_file_dir + "/coords.dump"

    tg, cpd = read_placement_timing_graph_full(tg_file, net_file)
    rr, tracks = conv_rr_to_nx(rr_file)
    nets = get_nets(route_file, tg, rr, coords_file)
    bp = build_bipartite_graph(rr, tracks, nets)
    dump_net_track(bp, net_track_file)

    proj = generate_projection(bp, nets)
    dump_projection(proj, projection_file)

    return bp, proj
##########################################################################

##########################################################################
def load_graphs(vpr_file_dir):
    #Loads the graphs dumped into the VPR file directory.

    net_track_file = vpr_file_dir + "/net_track.dump"
    projection_file = vpr_file_dir + "/projection.dump"

    bp = load_net_track(net_track_file)
    proj = load_projection(projection_file)

    return bp, proj
##########################################################################

##########################################################################
def run_vpr(benchmark, chan_w = -1, widening = 0):
    #Runs VPR on the specified benchmark with the specified channel width.
    #If the channel width is negative, the minimum is determined and increased
    #by "widening" percent for the final routing.

    arc = "k6_N10_40nm.xml"
    tail = " --pres_fac_mult 1.1 --max_criticality 0.999 --max_router_iterations 150"
    os.system("mkdir run")
    os.system("cp " + arc  + " run/")
    os.system("cp " + benchmark  + " run/")

    os.chdir("run/")
    benchmark = benchmark.rsplit('/', 1)[-1]
    rtc_w = " --route_chan_width " + str(chan_w) if chan_w > 0 else ""
    os.system("vtr7-devel " + arc + ' ' + benchmark\
              + " --echo_file on --nodisp" + rtc_w + tail)
    log = open("vpr_stdout.log", "r")
    lines = log.readlines()
    log.close()
    rd_first = 0
    for line in lines:
        if rd_first == 1:
            rd_first = 2
            continue
        elif rd_first == 2:
            fi_td = float(line.split()[-2])
            rd_first = 0
        if "Placement estimated critical path delay" in line:
            pp_td = float(line.split()[-2])
        elif "Final critical path" in line:
            rt_td = float(line.split()[-2]) 
        elif line.strip() == "Routing iteration: 1":
            rd_first = 1

    if chan_w < 0:
        log = open("vpr_stdout.log", "r")
        lines = log.readlines()
        log.close()
        for line in lines:
            if "Best routing used a channel width factor of" in line:
                chan_w = int(line.split()[-1][:-1])

        chan_w += float(chan_w) * widening / 100
        if int(chan_w) < chan_w:
            chan_w += 1
        chan_w = int(chan_w)
        if chan_w % 2:
            chan_w += 1
        rtc_w = " --route_chan_width " + str(chan_w)
        os.system("vtr7-devel " + arc + ' ' + benchmark\
                  + " --route --echo_file on --nodisp" + rtc_w + tail)
 
        log = open("vpr_stdout.log", "r")
        lines = log.readlines()
        log.close()
        rd_first = 0
        for line in lines:
            if rd_first == 1:
                rd_first = 2
                continue
            elif rd_first == 2:
                fi_td = float(line.split()[-2])
                rd_first = 0
            if "Final critical path" in line:
                rt_td = float(line.split()[-2])
            elif line.strip() == "Routing iteration: 1":
                rd_first = 1
 
    #Export files:
    benchmark_name = benchmark.rsplit('.', 1)[0]
    tg_file = "run/final_placement_timing_graph.echo"
    net_file = "run/" + benchmark_name + ".net"
    rr_file = "run/rr_graph.echo"
    route_file = "run/" + benchmark_name + ".route"
    cp_file = "run/" + benchmark_name + ".critical_path.out"
    export_files = [tg_file, net_file, rr_file, route_file, cp_file]

    export_dir = benchmark_name + '.' + str(chan_w)
    os.chdir("../")
    os.system("mkdir " + export_dir)
    for f in export_files:
        os.system("cp " + f + ' ' + export_dir)
    os.system("rm -rf run/")

    log = open(export_dir + "/log.txt", "w")
    log.write("pp_td = %.4g"%pp_td + "; fi_td = %.4g"%fi_td\
              + " (%.4g%%)"%((fi_td - pp_td) / pp_td * 100)\
              + "; rt_td = %.4g"%rt_td\
              + " (%.4g%%)"%((rt_td - pp_td) / pp_td * 100))
    log.close()
##########################################################################

##########################################################################
def sweep_channel_widths(benchmark, minw, maxw):
    #Sweeps the channel widths by increments of 10% and runs VPR.

    benchmark_name = benchmark.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    os.system("mkdir " + benchmark_name)
    prev = os.listdir('.')
    for inc in xrange(minw, maxw + 10, 10):
        run_vpr(benchmark, -1, inc)
    for f in os.listdir('.'):
        if not f in prev:
            os.system("mv " + f + ' ' + benchmark_name)
##########################################################################

##########################################################################
def generate_batch(resdir):
    #Generates graphs for a batch of VPR files, produced by sweeping route
    #channel widths on a particular benchmark.

    wd = os.getcwd()
    os.chdir(resdir)
    for d in sorted(os.listdir('.')):
        bp, proj = generate_graphs(d)
        #print brute_force_heaviest_cluster(proj)

    os.chdir(wd)
##########################################################################

##########################################################################
def read_logs(resdir):
    #Reads the logs generated in the channel sweep.
    
    for d in sorted(os.listdir(resdir)):
        os.system("cat " + resdir + '/' + d + "/log.txt")
        print 
########################################################################## 

##########################################################################
def read_critical_path(cp_dir):
    #Reads the critical path and dumps it as json.

    cp_file = cp_dir + '/' + [f for f in os.listdir(cp_dir)\
                              if f.endswith(".critical_path.out")][0]

    cp = open(cp_file, "r")
    lines = cp.readlines()
    cp.close()

    cp = []
    for line in lines:
        if "Internal Net" in line or "External-to_Block Net" in line:
            u = line.split('(', 1)[1].split(')', 1)[0]
            if not cp or cp[-1] != u:
                cp.append(u)

    cp_out = open(cp_dir + "/cp.json", "w")
    json.dump(cp, cp_out)
    cp_out.close()
##########################################################################
