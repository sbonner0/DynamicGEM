import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score, confusion_matrix)

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gcn
# ------------------------------------


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_adj_graph(data_location):
    """Load a given graph and return the adjacency matrix"""

    adj = sp.load_npz(data_location)
    return adj, None


def mask_test_edges(adj, test_percent=30., val_percent=20.):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0
 
    edges_positive, _, _ = sparse_to_tuple(adj)
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:] # filtering out edges from lower triangle of adjacency matrix
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx] # positive test edges
    val_edges = edges_positive[val_edge_idx] # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # positive train edges

    # the above strategy for sampling without replacement will not work for sampling negative edges on large graphs, because the pool of negative edges is much much larger due to sparsity
    # therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT (without replacement is super slow). sample more than we need so we'll probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough

    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices

    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test-len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx,positive_idx,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_test_edges_false,assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords,axis=0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0]!=coords[:,1]]
        # step 7:
        coords = coords[:min(num_test,len(idx))]
        test_edges_false = np.append(test_edges_false,coords,axis=0)
        idx = idx[:min(num_test,len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)


    val_edges_false = np.empty((0,2),dtype='int64')
    idx_val_edges_false = np.empty((0,),dtype='int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val-len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx,positive_idx,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_test_edges_false,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_val_edges_false,assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords,axis=0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0]!=coords[:,1]]
        # step 7:
        coords = coords[:min(num_val,len(idx))]
        val_edges_false = np.append(val_edges_false,coords,axis=0)
        idx = idx[:min(num_val,len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], test_edges_linear))

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def eval_gae(edges_pos, edges_neg, model, use_embeddings=False):
    """Evaluate the GAE model via link prediction"""

    if use_embeddings:
        emb = model.get_embeddings()
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
    else:
        adj_rec = model.predict_next_adj()
    preds = []
    
    # Loop over the positive test edges
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
    
    preds_neg = []

    # Loop over the negative test edges
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    accuracy = accuracy_score(labels_all, (preds_all > 0.5).astype(float))
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    tn, fp, fn, tp = confusion_matrix(labels_all, (preds_all > 0.5).astype(float)).ravel()

    return accuracy, roc_score, ap_score, tn, fp, fn, tp
