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
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm

#import the methods
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
from dynamicgem.embedding.dynAERNN     import DynAERNN


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
    print(f"Validation and Test edges capture from graph {args.dataset}_t{args.seq_len-1} in the sequence")

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
                        xeta       = 1e-5,
                        n_batch    = 100,
                        modelfile  = ['./intermediate/enc_modelsbm.json',
                                    './intermediate/dec_modelsbm.json'],
                        weightfile = ['./intermediate/enc_weightssbm.hdf5',
                                    './intermediate/dec_weightssbm.hdf5'])
        t1 = time()
        #ae static

        # Loop through each of the graphs in the time series and train model 
        print("Starting training AE")
        for temp_var in range(num_training_loops):
            emb, _= embedding.learn_embeddings(graphs[temp_var])
            
        print(embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding))

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
                        xeta           = 1e-5,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynAE.json', 
                                        './intermediate/dec_model_dynAE.json'],
                        weightfile     = ['./intermediate/enc_weights_dynAE.hdf5', 
                                        './intermediate/dec_weights_dynAE.hdf5'],
                        savefilesuffix = "testing" )
        t1 = time()
        for temp_var in range(lookback+1, num_training_loops+1):
            print(temp_var)
            print(graphs[:temp_var])
            emb, _ = embedding.learn_embeddings(graphs[:temp_var])
            
        print(embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding))

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
                        xeta           = 1e-4,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynRNN.json', 
                                        './intermediate/dec_model_dynRNN.json'],
                        weightfile     = ['./intermediate/enc_weights_dynRNN.hdf5', 
                                        './intermediate/dec_weights_dynRNN.hdf5'],
                        savefilesuffix = "testing"  )

        t1 = time()
        for temp_var in range(lookback+1, num_training_loops+1):
            emb, _ = embedding.learn_embeddings(graphs[:temp_var])

        print(embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding))

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

        t1 = time()
        for temp_var in range(lookback+1, num_training_loops+1):
                        emb, _ = embedding.learn_embeddings(graphs[:temp_var])

        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding))


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--model', type=str, default='AE', help='Which model to train.')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset string.')  
    parser.add_argument('--seq_len', type=int, default=6, help='Length of the sequence to load.')  

    args = parser.parse_args()

    main(args)

