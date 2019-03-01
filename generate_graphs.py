import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm


def generate_SBM_graphs():
    """Generate SBM Graphs"""

    #Parameters for Stochastic block model graph
    # Total of 1000 nodes
    node_num           = 3000
    # Test with two communities
    community_num      = 2
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num    = 10
    # Length of total time steps the graph will dynamically change
    length             = 30
    # output directory for result
    outdir = './output'

    testDataType = 'sbm_cd'

    # community ID to perturb
    com_id = 1 

    #Generate the dynamic graph
    dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num, community_num, length, com_id, node_change_num))
    graphs = [g[0] for g in dynamic_sbm_series]

    # Save the graphs to disk as scipy spare matrices
    save_location = "./"
    for i in range(len(graphs)):
        adj = nx.linalg.adj_matrix(graphs[i])
        sp.save_npz(f'{save_location}sbm_{node_num}_t{i}.npz', adj)

    return graphs

if __name__=="__main__":

    generate_SBM_graphs()
