import argparse
import matplotlib.pyplot as plt
from time import time
import networkx as nx
import pickle
import numpy as np
import os

import scipy.sparse as sp

#import helper libraries
from dynamicgem.utils      import graph_util, plot_util, dataprep_util, third_party_utils
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.evaluation.evaluate_link_prediction import evaluateDynamicLinkPrediction as LP
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm

#import the methods
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
from dynamicgem.embedding.dynAERNN     import DynAERNN

# Parameters for Stochastic block model graph
# Todal of 1000 nodes
node_num           = 1000
# Test with two communities
community_num      = 2
# At each iteration migrate 10 nodes from one community to the another
node_change_num    = 10
# Length of total time steps the graph will dynamically change
length             = 7
# output directory for result
outdir = './output'
intr='./intermediate'
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(intr):
    os.mkdir(intr)  
testDataType = 'sbm_cd'
#Generate the dynamic graph

dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num, community_num, length, 
                                                          1, #comminity ID to perturb
                                                          node_change_num))
graphs = [g[0] for g in dynamic_sbm_series]
print("Graph Generated!")

print(type(graphs[0]))
# parameters for the dynamic embedding
# dimension of the embedding
dim_emb  = 128
lookback = 2

def main(args):

    # Set the number of timesteps in the sequence
    num_timesteps = args.seq_len - 1 # one timestep per pair of consecutive graphs
    num_training_loops = num_timesteps - 1 # Num training loops to actually do (keep last graph for test/validation)

    # Preload the training graphs into memory...not very scaleable but helps with CPU load
    # Preload all but the last graph as this is use for val/test
    graphs = []
    for i in range(num_timesteps):
        adj_train, features = third_party_utils.load_adj_graph(f'data/{args.dataset}_t{i}.npz') # Load the input graph 
        graphs.append(nx.from_scipy_sparse_matrix(adj_train, create_using=nx.DiGraph()))
        print(f'data/{args.dataset}_t{i} Loaded')
    assert len(graphs) == num_timesteps #Should be the length of the time series as the index will start from zero
    print("Training graphs loaded into memory")

    # Extract the val/test graph which is the final one in the sequence
    val_test_graph, _ = third_party_utils.load_adj_graph(f'data/{args.dataset}_t{args.seq_len-1}.npz')
    val_test_graph_adj, _, val_edges, val_edges_false, test_edges, test_edges_false = third_party_utils.mask_test_edges(val_test_graph)
    val_test_graph = nx.from_scipy_sparse_matrix(val_test_graph_adj, create_using=nx.DiGraph())
    print("Validation and Test edges capture from last graph in the sequence")

    # Chose the model to run
    #AE Static ----------------------------------------------------------------------------
    if args.model == "AE":

        embedding = AE(d            = dim_emb, 
                        beta       = 5, 
                        nu1        = 1e-6, 
                        nu2        = 1e-6,
                        K          = 3, 
                        n_units    = [500, 300, ],
                        n_iter     = 100, 
                        xeta       = 1e-4,
                        n_batch    = 100,
                        modelfile  = ['./intermediate/enc_modelsbm.json',
                                    './intermediate/dec_modelsbm.json'],
                        weightfile = ['./intermediate/enc_weightssbm.hdf5',
                                    './intermediate/dec_weightssbm.hdf5'])
        embs  = []
        t1 = time()
        #ae static
        # Loop through each of the graphs in the time series
        print("Starting training AE")
        for temp_var in range(num_training_loops):
            emb, _= embedding.learn_embeddings(graphs[temp_var])
            embs.append(emb)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))

        #viz.plot_static_sbm_embedding(embs[-4:], dynamic_sbm_series[-4:])   
        MAP, prec_curv = LP(graphs[-1], embedding)
        print(f"MAP Value: {MAP}")

    #dynAE ------------------------------------------------------------------------------
    elif args.model == "DynAE":

        embedding= DynAE(d           = dim_emb,
                        beta           = 5,
                        n_prev_graphs  = lookback,
                        nu1            = 1e-6,
                        nu2            = 1e-6,
                        n_units        = [500, 300,],
                        rho            = 0.3,
                        n_iter         = 250,
                        xeta           = 1e-4,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynAE.json', 
                                        './intermediate/dec_model_dynAE.json'],
                        weightfile     = ['./intermediate/enc_weights_dynAE.hdf5', 
                                        './intermediate/dec_weights_dynAE.hdf5'],
                        savefilesuffix = "testing" )
        embs = []
        t1 = time()
        for temp_var in range(lookback+1, length+1):
                        emb, _ = embedding.learn_embeddings(graphs[:temp_var])
                        embs.append(emb)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        plt.figure()
        plt.clf()    
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])    
        plt.show()

    #dynRNN ------------------------------------------------------------------------------
    elif args.model == "DynRNN":

        embedding= DynRNN(d        = dim_emb,
                        beta           = 5,
                        n_prev_graphs  = lookback,
                        nu1            = 1e-6,
                        nu2            = 1e-6,
                        n_enc_units    = [500,300],
                        n_dec_units    = [500,300],
                        rho            = 0.3,
                        n_iter         = 250,
                        xeta           = 1e-3,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynRNN.json', 
                                        './intermediate/dec_model_dynRNN.json'],
                        weightfile     = ['./intermediate/enc_weights_dynRNN.hdf5', 
                                        './intermediate/dec_weights_dynRNN.hdf5'],
                        savefilesuffix = "testing"  )
        embs = []
        t1 = time()
        for temp_var in range(lookback+1, length+1):
                        emb, _ = embedding.learn_embeddings(graphs[:temp_var])
                        embs.append(emb)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        plt.figure()
        plt.clf()    
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])    
        plt.show()

    #dynAERNN ------------------------------------------------------------------------------
    elif args.model == "DynAERNN":

        embedding = DynAERNN(d   = dim_emb,
                    beta           = 5,
                    n_prev_graphs  = lookback,
                    nu1            = 1e-6,
                    nu2            = 1e-6,
                    n_aeunits      = [500, 300],
                    n_lstmunits    = [500,dim_emb],
                    rho            = 0.3,
                    n_iter         = 250,
                    xeta           = 1e-3,
                    n_batch        = 100,
                    modelfile      = ['./intermediate/enc_model_dynAERNN.json', 
                                    './intermediate/dec_model_dynAERNN.json'],
                    weightfile     = ['./intermediate/enc_weights_dynAERNN.hdf5', 
                                    './intermediate/dec_weights_dynAERNN.hdf5'],
                    savefilesuffix = "testing")

        embs = []
        t1 = time()
        for temp_var in range(lookback+1, length+1):
                        emb, _ = embedding.learn_embeddings(graphs[:temp_var])
                        embs.append(emb)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        plt.figure()
        plt.clf()    
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])    
        plt.show()


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--model', type=str, default='AE', help='Which model to train.')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset string.')  
    parser.add_argument('--seq_len', type=int, default=6, help='Length of the sequence to load.')  

    args = parser.parse_args()

    main(args)

