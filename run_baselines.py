import argparse
import os
import pickle
from collections import defaultdict
from time import time

import networkx as nx
import numpy as np

#import the methods
from dynamicgem.embedding.ae_static import AE
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.dynRNN import DynRNN

from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.evaluation import evaluate_link_prediction as lp
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
#import helper libraries
from dynamicgem.utils import (dataprep_util, graph_util, plot_util,
                              third_party_utils)
from dynamicgem.visualization import plot_dynamic_sbm_embedding

# parameters for the dynamic embedding
# dimension of the embedding
dim_emb  = 128
lookback = 2

def main(args):

    # Set seeds
    np.random.seed(args.seed)
    from tensorflow import set_random_seed
    set_random_seed(args.seed)

    # Set the number of timesteps in the sequence
    num_timesteps = args.seq_len - 1 # one timestep per pair of consecutive graphs
    num_training_loops = num_timesteps - 1 # Num training loops to actually do (keep last graph for test/validation)

    data_loc = os.path.join(args.data_loc, args.dataset)

    # Preload the training graphs into memory...not very scaleable but helps with CPU load
    # Preload all but the last graph as this is use for val/test
    graphs = []
    for i in range(num_timesteps):
        adj_train, features = third_party_utils.load_adj_graph(f'{data_loc}_t{i}.npz') # Load the input graph 
        graphs.append(nx.from_scipy_sparse_matrix(adj_train, create_using=nx.DiGraph()))
        print(f'{args.dataset}_t{i} Loaded')
    assert len(graphs) == num_timesteps #Should be the length of the time series as the index will start from zero
    print("Training graphs loaded into memory")

    # Extract the val/test graph which is the final one in the sequence
    val_test_graph_previous, _ = third_party_utils.load_adj_graph(f'{data_loc}_t{num_timesteps-1}.npz')
    val_test_graph, _ = third_party_utils.load_adj_graph(f'{data_loc}_t{num_timesteps}.npz')
    val_test_graph_adj, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = third_party_utils.mask_test_edges(val_test_graph)
    val_test_graph_adj, train_edges_pre, val_edges_pre, val_edges_false, test_edges_pre, test_edges_false = third_party_utils.mask_test_edges(val_test_graph_previous)

    pos_edges = np.concatenate((val_edges, test_edges, train_edges)).tolist()
    pos_edges = set(map(tuple, pos_edges))
    pos_edges_pre = np.concatenate((val_edges_pre, test_edges_pre, train_edges_pre)).tolist()
    pos_edges_pre = set(map(tuple, pos_edges_pre))
    new_edges = np.array(list(pos_edges - pos_edges_pre))

    num_edges = len(new_edges)
    new_edges_false = test_edges[:num_edges]

    print(f"Validation and Test edges capture from graph {args.dataset}_t{args.seq_len-1} in the sequence")

    # Chose the model to run
    #AE Static ----------------------------------------------------------------------------
    # None offset auto encoder seems to be
    if args.model == "AE":

        embedding = AE(d            = dim_emb, 
                        beta       = 5, 
                        nu1        = 1e-6, 
                        nu2        = 1e-6,
                        K          = 3, 
                        n_units    = [500, 300, ],
                        n_iter     = 100, 
                        xeta       = 1e-6,
                        n_batch    = 100,
                        modelfile  = ['./intermediate/enc_modelsbm.json',
                                    './intermediate/dec_modelsbm.json'],
                        weightfile = ['./intermediate/enc_weightssbm.hdf5',
                                    './intermediate/dec_weightssbm.hdf5'])
        t1 = time()
        #ae static

        # Loop through each of the graphs in the time series and train model 
        print("Starting training AE")
        # for temp_var in range(num_training_loops):
        #     emb, _= embedding.learn_embeddings(graphs[temp_var])

        emb, _ = embedding.learn_embeddings(graphs[:num_training_loops])
        print(embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding))

        accuracy, roc_score, ap_score, tn, fp, fn, tp = third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False)
        ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp = third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False)

    #dynAE ------------------------------------------------------------------------------
    # As proposed in dyngraph2vec paper. Seems to just be an offset dense auto encoder trained to predict next graph. 
    elif args.model == "DynAE":

        embedding= DynAE(d           = dim_emb,
                        beta           = 5,
                        n_prev_graphs  = lookback,
                        nu1            = 1e-6,
                        nu2            = 1e-6,
                        n_units        = [500, 300,],
                        rho            = 0.3,
                        n_iter         = 150,
                        xeta           = 1e-5,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynAE.json', 
                                        './intermediate/dec_model_dynAE.json'],
                        weightfile     = ['./intermediate/enc_weights_dynAE.hdf5', 
                                        './intermediate/dec_weights_dynAE.hdf5'],
                        savefilesuffix = "testing" )
        t1 = time()
        # for temp_var in range(lookback+1, num_training_loops+1):
        #     print(temp_var)
        #     print(graphs[:temp_var])
        #     emb, _ = embedding.learn_embeddings(graphs[:temp_var])

        emb, _ = embedding.learn_embeddings(graphs[:num_training_loops]) 

        if new_edges.size != 0:
            print("Here yo")
            accuracy, roc_score, ap_score, tn, fp, fn, tp = third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False)
            print(third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False))
        else:
            accuracy, roc_score, ap_score, tn, fp, fn, tp = 0,0,0,0,0,0,0 

        ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp = third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False)
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False))

    #dynRNN ------------------------------------------------------------------------------
    # As proposed in dyngraph2vec paper. Only seems to use LSTM cells with no compression beforehand.
    elif args.model == "DynRNN":

        embedding= DynRNN(d        = dim_emb,
                        beta           = 5,
                        n_prev_graphs  = lookback,
                        nu1            = 1e-6,
                        nu2            = 1e-6,
                        n_enc_units    = [500,200],
                        n_dec_units    = [200,500],
                        rho            = 0.3,
                        n_iter         = 150,
                        xeta           = 1e-4,
                        n_batch        = 100,
                        modelfile      = ['./intermediate/enc_model_dynRNN.json', 
                                        './intermediate/dec_model_dynRNN.json'],
                        weightfile     = ['./intermediate/enc_weights_dynRNN.hdf5', 
                                        './intermediate/dec_weights_dynRNN.hdf5'],
                        savefilesuffix = "testing"  )

        t1 = time()
        # for temp_var in range(lookback+1, num_training_loops+1):
        #     emb, _ = embedding.learn_embeddings(graphs[:temp_var])
        
        emb, _ = embedding.learn_embeddings(graphs[:num_training_loops])

        if new_edges.size != 0:
            accuracy, roc_score, ap_score, tn, fp, fn, tp = third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False)
            print(third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False))
        else:
            accuracy, roc_score, ap_score, tn, fp, fn, tp = 0,0,0,0,0,0,0 

        ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp = third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False)
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False))

    #dynAERNN ------------------------------------------------------------------------------
    # As proposed in dyngraph2vec paper. Use auto encoder before passing to an LSTM cell.
    elif args.model == "DynAERNN":
        
        embedding = DynAERNN(d   = dim_emb,
                    beta           = 5,
                    n_prev_graphs  = lookback,
                    nu1            = 1e-6,
                    nu2            = 1e-6,
                    n_aeunits      = [500, 300],
                    n_lstmunits    = [300, dim_emb],
                    rho            = 0.3,
                    n_iter         = 150,
                    xeta           = 1e-3,
                    n_batch        = 100,
                    modelfile      = ['./intermediate/enc_model_dynAERNN.json', 
                                    './intermediate/dec_model_dynAERNN.json'],
                    weightfile     = ['./intermediate/enc_weights_dynAERNN.hdf5', 
                                    './intermediate/dec_weights_dynAERNN.hdf5'],
                    savefilesuffix = "testing")

        t1 = time()
        # for temp_var in range(lookback+1, num_training_loops+1):
        #                 emb, _ = embedding.learn_embeddings(graphs[:temp_var])

        #lp.expLP(graphs, embedding, 2, 0, 0)

        emb, _ = embedding.learn_embeddings(graphs[:num_training_loops])

        if new_edges.size != 0:
            accuracy, roc_score, ap_score, tn, fp, fn, tp = third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False)
            print(third_party_utils.eval_gae(new_edges, new_edges_false, embedding, use_embeddings=False))
        else:
            accuracy, roc_score, ap_score, tn, fp, fn, tp = 0,0,0,0,0,0,0 

        ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp = third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False)
        print(third_party_utils.eval_gae(test_edges, test_edges_false, embedding, use_embeddings=False))

    return accuracy, roc_score, ap_score, tn, fp, fn, tp, ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--model', type=str, default='AE', help='Which model to train.')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset string.')
    parser.add_argument('--data_loc', type=str, default='/data/Temporal-Graph-Data/proc-data/', help='Dataset location string.')
    parser.add_argument('--seq_len', type=int, default=6, help='Length of the sequence to load.')  

    args = parser.parse_args()

    results = defaultdict(list)

    print(f'Dataset = {args.dataset}')

    # Here we loop over all combinates of 0...t
    time_range = args.seq_len
    for i in range(8, time_range):
        
        # Set the maximum sequence length
        args.seq_len = i

        # run the model
        acc_list, roc_list, ap_list, tn_list, fp_list, fn_list, tp_list, ae_acc_list, ae_roc_list, ae_ap_list, ae_tn_list, ae_fp_list, ae_fn_list, ae_tp_list = main(args)

        # Store the results!
        # This will be a list of lists, where first dimension is the time
        results['acc_list'].append(acc_list)
        results['roc_list'].append(roc_list)
        results['ap_list'].append(ap_list)
        results['tn_list'].append(tn_list)
        results['fp_list'].append(fp_list)
        results['fn_list'].append(fn_list)
        results['tp_list'].append(tp_list)

        # Store the results for new and old edges
        results['ae_acc_list'].append(ae_acc_list)
        results['ae_roc_list'].append(ae_roc_list)
        results['ae_ap_list'].append(ae_ap_list)
        results['ae_tn_list'].append(ae_tn_list)
        results['ae_fp_list'].append(ae_fp_list)
        results['ae_fn_list'].append(ae_fn_list)
        results['ae_tp_list'].append(ae_tp_list)

        pickle.dump(results, open(f'{args.dataset}_{args.model}_results.pickle', 'wb'))

    print("RUN COMPLETE!")