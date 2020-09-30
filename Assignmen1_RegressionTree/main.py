import numpy as np
import pandas as pd

def preprocess_data() :
    pass

def read_data() :
    pass

def get_variance_gain() :
    pass

def get_max_variance_gain() :
    pass



class DecisionTree():
    def __init__():
        # initialise tree and split by countries
        pass

    def train():
        # train by recusrsively creating nodes
        pass

    def show():
        # use graphviz to add nodes an edges recursively
        pass

    def predict():
        # get predcitions
        pass
    
    def test() :
        #test on test data and return r2 value
        pass


class Node():
    def __init__():
        # initialise node
        pass

    def set_children():
        # create children by finding max variance gain
        pass

    def show():
        # use graphviz to add nodes an edges recursively
        pass

    def predict():
        # get predcitions
        pass

class Leaf():
    def __init__():
        # initialise final leaf and set the value 
        pass

    def show():
        # use graphviz to add nodes an edges recursively
        pass

    def predict():
        # return value
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