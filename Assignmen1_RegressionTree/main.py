import numpy as np
import pandas as pd

def preprocess_data(data) :
    #return data as numpy array and a seaprate column for countries and a list of distinct countries
    pass

def read_data(PATH) :
    pass

def split_data(data, split_ratio=[0.8,0.2], random_seed=0) :
    assert(sum(split_ratio)==1)
    #return splitted datasets
    pass

def get_variance(data) :
    #return variance of vector data
    pass

def get_variance_gain(attr_values) :
    #return variance gain after splitting across mean
    pass

def get_max_variance_gain(data) :
    #return column attr and mean of that
    pass

class DecisionTree():
    def __init__(self, metadata, max_depth):
        self.id = None
        self.metadata = metadata #metadata is countries list
        self.max_depth = max_depth
        self.children = []
        # initialise tree and split by countries
        pass

    def train(self, data):
        # train by recusrsively creating nodes
        pass

    def show(self):
        # use graphviz to add nodes an edges recursively
        pass

    def predict(self, data):
        # get predcitions
        pass
    
    def test(self, data, target) :
        #test on test data and return r2 value
        pass

    def prune_tree(self) :
        #prune the tree
        pass


class Node():
    def __init__(self, data, level, max_level, id,  parent_id):
        self.id = id
        self.parent_id = parent_id
        self.data = data
        self.level = level
        self.max_level = max_level
        self.left_child = None
        self.right_child = None
        # initialise node
        pass

    def set_children(self):
        # create children by finding max variance gain
        pass

    def show(self):
        # use graphviz to add nodes an edges recursively
        pass

    def predict(self):
        # get predcitions
        pass


if __name__ == "__main__" :
    #add argparse
    '''
    Options:    1. train to max depth
                2. train to given depth
                3. find best depth and plot
                3. train with pruning i.e. variance gain thresholding("check if hypothesis testing to be used")
    '''
    pass