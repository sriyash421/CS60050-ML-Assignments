import numpy as np
import pandas as pd
PATH = "AggregatedCountriesCOVIDStats.csv"
def preprocess_data(df) :
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m%d%Y").astype(int)
    return df

def read_data(PATH) :
    df = pd.read_csv(PATH);
    df = preprocess_data(df)
    return df

def split_data(data, split_ratio=[0.8,0.2], random_seed = 0) :
    assert(sum(split_ratio)==1)
    if(len(split_ratio)  == 2):
        train = df.sample(frac=split_ratio[0],random_state=random_seed) #random state is a seed value
        test = df.drop(train.index)
        train_country = np.array(train["Country"])
        test_country = np.array(test["Country"])
        train = np.asarray(train.drop(columns = ["Country"]))
        test = np.asarray(test.drop(columns = ["Country"]))
        return (train, test, train_country, test_country)
    else :
        train = df.sample(frac=split_ratio[0],random_state=random_seed) #random state is a seed value
        test = df.drop(train.index)
        cross = test.sample(frac = (split_ratio[1]) / (1 - split_ratio[0]),random_state=random_seed)
        test = df.drop(cross.index)
        train_country = np.array(train["Country"])
        test_country = np.array(test["Country"])
        cross_country = np.array(cross["Country"])
        train = np.asarray(train.drop(columns = ["Country"]))
        test = np.asarray(test.drop(columns = ["Country"]))
        cross = np.asarray(cross.drop(columns = ["Country"]))
        return (train, cross, test, train_country, cross_country, test_country)        


def get_variance(data) :
    #return variance of vector data
    data = np.array(data)
    return np.var(data)

def get_variance_gain(A, B, C):
    #return variance gain of 3 lists
    len_B = len(B)
    len_C = len(C)
    return get_variance(A) - get_variance(B)*(len_B/(len_C+len_B)) - get_variance(B)*(len_C/(len_C+len_B))

def get_max_variance_gain(data) :
    # return colname , split point and gain of a given dataframe
    Y = data[:, [3]]
    X = data[:, 0:3]
    max_col = None
    max_gain = 0
    splice_point = -1
    for col in range(3):
        X = X[X[:,col].argsort()]
        current_col = list(X[:, [col]])
        for j in range(X.shape[0] - 1):
            pref = Y[:j]
            suf = Y[j+1:]
            current_gain = get_variance_gain(Y, pref, suf)
            if(current_gain >= min_gain):
                max_gain = current_gain
                max_col = col
                splice_point = (current_col[j] + current_col[j+1])/2
         
    return (max_col, splice_point, max_gain)


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