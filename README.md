# node2vec_linkprediction
Testing link prediction using Node2Vec

Installation
============

Requirements:
-------------
* git
* Python 2.7
* gensim
* networkx
* numpy
* matplotlib
* scikit-learn
* node2vec

To install using Anaconda:
--------------------------

1) To install on Mac OS or Linux, download and install Anaconda (2 or 3) from the following website:
https://www.continuum.io/downloads

2) At a command prompt, create a python 2.7 environment and install required packages:

    conda create -n py27 python=2.7 numpy ipython matplotlib seaborn networkx gensim scikit-learn

3) Switch to this environment:

    source activate py27

3) Get node2vec python code:

    git clone https://github.com/aditya-grover/node2vec.git

4) Copy node2vec.py to link prediction code directory:

    cp node2vec/src/node2vec.py <node2vec_linkprediction path>

Usage
=====

To use the link_prediction code, we assume the graph data is saved in the form of an edgelist of node pairs on a seperate line:

    Example edgelist:
    1 2
    3 4
    4 2

A task must be specified, which is one of:

* *edgeencoding*: Test the node2vec embedding using different edge functions, and analyse their performance.

* *sensitivity*: Run a parameter sensitivity test on the node2vec parameters of q, p, r, l, d, and k.

* *gridsearch*: Run a grid search on the node2vec parameters of q, p.

For example, to test the edge encodings for the graph AstroPh.edgelist, with averaging over five random walk samplings in node2vec:

    python link_prediction.py edgeembedding --input AstroPh.edgelist  --num_experiments 5

For help on the options, use:

    python link_prediction.py --help

The default values for the experiments and parameter search settings are in the code link_prediction.py.
