'''
'''
from __future__ import print_function, division

import pickle
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import node2vec
import operator as op
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Default parameters from node2vec paper (and for DeepWalk)
default_params = {
    'log2p': 0,                     # Parameter p, p = 2**log2p
    'log2q': 0,                     # Parameter q, q = 2**log2q
    'log2d': 7,                     # Feature size, dimensions = 2**log2d
    'num_walks': 10,                # Number of walks from each node
    'walk_length': 80,              # Walk length
    'window_size': 10,              # Context size for word2vec
    'edge_function': "hadamard",    # Default edge function to use
    "prop_pos": 0.5,                # Proportion of edges to remove nad use as positive samples
    "prop_neg": 0.5,                # Number of non-edges to use as negative samples
                                    #  (as a proportion of existing edges, same as prop_pos)
}

parameter_searches = {
    'log2p': (np.arange(-2, 3), '$\log_2 p$'),
    'log2q': (np.arange(-2, 3), '$\log_2 q$'),
    'log2d': (np.arange(4, 9), '$\log_2 d$'),
    'num_walks': (np.arange(6, 21, 2), 'Number of walks, r'),
    'walk_length': (np.arange(40, 101, 10), 'Walk length, l'),
    'window_size': (np.arange(8, 21, 2), 'Context size, k'),
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('task', type=str,
                        help="Task to run, one of 'gridsearch', 'edgeencoding', and 'sensitivity'")

    parser.add_argument('--input', nargs='?', default='karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Regenerate random positive/negative links')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--num_experiments', type=int, default=5,
                        help='Number of experiments to average. Default is 5.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


class GraphN2V(node2vec.Graph):
    def __init__(self,
                 nx_G=None, is_directed=False,
                 prop_pos=0.5, prop_neg=0.5,
                 random_seed=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_neg
        self.prop_neg = prop_pos
        self.wvecs = None
        self._rnd = np.random.RandomState(seed=random_seed)

    def read_graph(self, input, enforce_connectivity=True, weighted=False, directed=False):
        '''
        Reads the input network in networkx.
        '''
        if weighted:
            G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G.edge[edge[0]][edge[1]]['weight'] = 1

        if not directed:
            G = G.to_undirected()

        # Take largest connected subgraph
        if enforce_connectivity and not nx.is_connected(G):
            G = max(nx.connected_component_subgraphs(G), key=len)
            print("Input graph not connected: using largest connected subgraph")

        # I'm not going to consider self-edges right now
        # There aren't that many for AstroPh.
        for se in G.nodes_with_selfloops():
            G.remove_edge(se, se)

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    def learn_embeddings(self, walks, dimensions, window_size=10, workers=4, niter=5):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        # TODO: Python27 only
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks,
                         size=dimensions,
                         window=window_size,
                         min_count=0,
                         sg=1,
                         workers=workers,
                         iter=niter)
        self.wvecs = model.wv

    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.
        """
        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        n_nodes = self.G.number_of_nodes()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        if not nx.is_connected(self.G):
            raise RuntimeError("Input graph is not connected")

        n_neighbors = [len(self.G.neighbors(v)) for v in self.G.nodes_iter()]
        n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        non_edges = [e for e in nx.non_edges(self.G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "Only %d negative edges found" % (len(neg_edge_list))
            )

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        edges = self.G.edges()
        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        rnd_inx = self._rnd.permutation(n_edges)
        for eii in rnd_inx:
            edge = edges[eii]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            # Check if graph is still connected
            #TODO: We shouldn't be using a private function for bfs
            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                print("Pos Edges: %d" % n_count, end="\r")
                n_count += 1

            # Exit if we've found npos nodes or we have gone through the whole list
            if n_count >= npos:
                break

        if len(pos_edge_list) < npos:
            raise RuntimeWarning("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def train_embeddings(self, p, q, dimensions, num_walks,
                         walk_length, window_size, workers=1):
        """
        Calculate nodde embedding with specified parameters
        :param p:
        :param q:
        :param dimensions:
        :param num_walks:
        :param walk_length:
        :param window_size:
        :param workers:
        :return:
        """
        self.p = p
        self.q = q
        self.preprocess_transition_probs()
        walks = self.simulate_walks(num_walks, walk_length)
        self.learn_embeddings(
            walks, dimensions, window_size, workers=workers
        )

    def edges_to_features(self, edge_list, edge_function, dimensions):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list

        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(self.wvecs[str(v1)])
            emb2 = np.asarray(self.wvecs[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


def create_train_test_graphs(args):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.

    :param args:
    :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # Create random training and test graphs with different random edge selections
    cached_fn = "%s.graph" % (os.path.basename(args.input))
    if os.path.exists(cached_fn) and not args.regen:
        print("Loading link prediction graphs from %s" % cached_fn)
        with open(cached_fn, 'rb') as f:
            cache_data = pickle.load(f)
        Gtrain = cache_data['g_train']
        Gtest = cache_data['g_test']

    else:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = GraphN2V(is_directed=False,
                          prop_pos=prop_pos, prop_neg=prop_neg,
                          random_seed=0x12A283)
        Gtrain.read_graph(args.input, weighted=args.weighted, directed=args.directed)
        Gtrain.generate_pos_neg_links()

        # Generate a different random graph for testing
        Gtest = GraphN2V(is_directed=False,
                         prop_pos=prop_pos, prop_neg=prop_neg,
                         random_seed=0x223C4D2)
        Gtest.read_graph(args.input, weighted=args.weighted, directed=args.directed)
        Gtest.generate_pos_neg_links()

        # Cache generated  graph
        cache_data = {'g_train': Gtrain, 'g_test': Gtest}
        with open(cached_fn, 'wb') as f:
            pickle.dump(cache_data, f)

    return Gtrain, Gtest


def test_edge_functions(args):
    Gtrain, Gtest = create_train_test_graphs(args)

    p = 2.0**default_params['log2p']
    q = 2.0**default_params['log2q']
    dimensions = 2**default_params['log2d']
    num_walks = default_params['num_walks']
    walk_length = default_params['walk_length']
    window_size = default_params['window_size']

    # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()

    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    # TODO: It would be nice if the walks had a settable random seed
    aucs = {name: [] for name in edge_functions}
    for iter in range(args.num_experiments):
        print("Iteration %d of %d" % (iter, args.num_experiments))

        # Learn embeddings with current parameter values
        Gtrain.train_embeddings(p, q, dimensions, num_walks, walk_length, window_size)
        Gtest.train_embeddings(p, q, dimensions, num_walks, walk_length, window_size)

        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, dimensions)
            edge_features_test = Gtest.edges_to_features(edges_test, edge_fn, dimensions)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(edge_features_train, labels_train)
            auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

            # Test classifier
            auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)
            aucs[edge_fn_name].append(auc_test)

    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))

    return aucs


def plot_parameter_sensitivity(args):
    # Train and test graphs, with different edges
    Gtrain, Gtest = create_train_test_graphs(args)
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()

    # Setup plot
    fig, axes = plt.subplots(2, int(np.ceil(len(parameter_searches)/2)))
    axes = axes.ravel()

    # Explore different parameters
    for ii, param in enumerate(parameter_searches):
        cparams = default_params.copy()
        param_values, xlabel = parameter_searches[param]
        param_aucs = []
        for pv in param_values:
            # Update current parameters and get values for experiment
            cparams[param] = pv
            p = 2.0**cparams['log2p']
            q = 2.0**cparams['log2q']
            dimensions = 2**cparams['log2d']
            edge_fn = edge_functions[default_params['edge_function']]
            num_walks = cparams['num_walks']
            walk_length = cparams['walk_length']
            window_size = cparams['window_size']

            # With fixed test & train graphs (these are expensive to generate)
            # we perform num_experiments iterations of the algorithm, using
            # all positive & negative links in both graphs
            # TODO: It would be nice if the walks had a settable random seed
            cv_aucs = []
            for iter in range(args.num_experiments):
                print("Iteration %d of %d" % (iter, args.num_experiments))
                # Learn embeddings with current parameter values
                Gtrain.train_embeddings(p, q, dimensions, num_walks, walk_length, window_size)
                Gtest.train_embeddings(p, q, dimensions, num_walks, walk_length, window_size)

                # Calculate edge embeddings using binary function
                edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, dimensions)
                edge_features_test = Gtest.edges_to_features(edges_test, edge_fn, dimensions)

                # Linear classifier
                scaler = StandardScaler()
                lin_clf = LogisticRegression(C=1)
                clf = pipeline.make_pipeline(scaler, lin_clf)

                # Train classifier
                clf.fit(edge_features_train, labels_train)
                auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

                # Test classifier
                auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)

                cv_aucs.append(auc_test)

                print("%s = %.3f; AUC train: %.4g AUC test: %.4g"
                      % (param, pv, auc_train, auc_test))

            # Add mean of partitoned scores
            param_aucs.append(np.mean(cv_aucs))

        # Plot figure
        ax = axes[ii]
        ax.plot(param_values, param_aucs, 'r-', marker='s', ms=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('AUC')

    plt.savefig()
    plt.show()


def grid_search(args):
    Gtrain, Gtest = create_train_test_graphs(args)
    num_partitions = args.num_experiments

    # Parameter grid
    grid_parameters = ['log2p', 'log2q']
    grid_values = [np.arange(-2,3), np.arange(-2,3)]
    grid_shape = [len(p) for p in grid_values]

    # Store values in tensor
    grid_aucs = np.zeros(grid_shape + [num_partitions])

    # Explore different parameters
    cparams = default_params.copy()
    for grid_inx in np.ndindex(*grid_shape):
        for ii, param in enumerate(grid_parameters):
            cparams[param] = grid_values[ii][grid_inx[ii]]

        # I'm not sure about this, but it makes plotting things easier
        p = 2.0**cparams['log2p']
        q = 2.0**cparams['log2q']
        dimensions = 2**cparams['log2d']
        edge_fn = edge_functions[cparams['edge_function']]
        num_walks = cparams['num_walks']
        walk_length = cparams['walk_length']
        window_size = cparams['window_size']

        # With fixed test & train graphs (these are expensive to generate)
        # we perform num_experiments iterations of the algorithm, using
        # different sets of links to train & test the linear classifier.
        # This really isn't k-fold CV as the embeddings are learned without
        # holdout of any data, but it will average over the random walks and
        # estimate how the linear classifier generalizes, at least.
        partitioner = model_selection.StratifiedKFold(num_partitions, shuffle=True)
        edges_all, edge_labels_all = Gtrain.get_selected_edges()

        # Iterate over folds
        cv_aucs = []
        iter = 0
        for train_inx, test_inx in partitioner.split(edges_all, edge_labels_all):
            edges_train = [edges_all[jj] for jj in train_inx]
            labels_train = [edge_labels_all[jj] for jj in train_inx]
            edges_test = [edges_all[jj] for jj in test_inx]
            labels_test = [edge_labels_all[jj] for jj in test_inx]

            # Learn embeddings with current parameter values
            Gtrain.train_embeddings(p, q, dimensions, num_walks, walk_length, window_size)

            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, dimensions)
            edge_features_test = Gtrain.edges_to_features(edges_test, edge_fn, dimensions)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train & validate classifier
            clf.fit(edge_features_train, labels_train)
            auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

            # Test classifier
            auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)

            print("%s; AUC train: %.4g AUC test: %.4g"
                  % (grid_inx, auc_train, auc_test))

            # Add to grid scores
            grid_aucs[grid_inx + (iter,)] = auc_test
            iter += 1

    # Now find the best:
    mean_aucs = grid_aucs.mean(axis=-1)

    print("AUC mean:")
    print(mean_aucs)

    print("AUC std dev:")
    print(grid_aucs.std(axis=-1))

    if len(grid_values) == 2:
        plt.figure()
        plt.pcolormesh(grid_values[0], grid_values[1], mean_aucs)
        plt.colorbar()
        plt.xlabel(grid_parameters[0])
        plt.ylabel(grid_parameters[1])
        plt.show()

    return grid_aucs


if __name__ == "__main__":
    args = parse_args()

    if args.task is None:
        print("Specify task to run: edgeembedding, sensitivity, gridsearch")
        exit()

    if args.task.startswith("grid"):
        grid_search(args)

    elif args.task.startswith("edge"):
        test_edge_functions(args)

    elif args.task.startswith("sens"):
        plot_parameter_sensitivity(args)

