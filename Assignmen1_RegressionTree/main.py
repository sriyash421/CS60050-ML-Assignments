import pickle
import argparse
import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt

CURR_ID = 0

def get_col_label(i) :
    temp = ["Date","Confirmed","Recovery","Deaths"]
    return temp[i]

def preprocess_data(df) :
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m%d%Y").astype(int)
    return df

def read_data(PATH) :
    df = pd.read_csv(PATH)
    df = preprocess_data(df)
    return df, list(df["Country"].unique())

def split_data(data, split_ratio=[0.8,0.2], random_seed = 0) :
    assert(sum(split_ratio)==1)
    if(len(split_ratio)  == 2):
        train = data.sample(frac=split_ratio[0],random_state=random_seed) #random state is a seed value
        test = data.drop(train.index)
        train_country = np.array(train["Country"])
        test_country = np.array(test["Country"])
        train = np.asarray(train.drop(columns = ["Country"]))
        test = np.asarray(test.drop(columns = ["Country"]))
        return (train, test, train_country, test_country)
    else :
        train = data.sample(frac=split_ratio[0],random_state=random_seed) #random state is a seed value
        test = data.drop(train.index)
        cross = test.sample(frac = (split_ratio[1]) / (1 - split_ratio[0]),random_state=random_seed)
        test = data.drop(cross.index)
        train_country = np.array(train["Country"])
        test_country = np.array(test["Country"])
        cross_country = np.array(cross["Country"])
        train = np.asarray(train.drop(columns = ["Country"]))
        test = np.asarray(test.drop(columns = ["Country"]))
        cross = np.asarray(cross.drop(columns = ["Country"]))
        return (train, cross, test, train_country, cross_country, test_country)        


def get_variance(data) :
    #return variance of vector data
    return np.var(data)

def get_variance_gain(A, B, C):
    #return variance gain of 3 lists
    return get_variance(A) - get_variance(B)*(len(B)/(len(C)+len(B))) - get_variance(B)*(len(C)/(len(C)+len(B)))

def get_max_variance_gain(data) :
    # return colname , split point and gain of a given dataframe
    Y = data[:, [3]]
    X = data[:, 0:3]
    max_col = None
    max_gain = 0
    slice_point = -1
    for col in range(3):
        X = X[X[:,col].argsort()]
        current_col = list(X[:, [col]])
        for j in range(X.shape[0] - 1):
            pref = Y[:j]
            suf = Y[j+1:]
            current_gain = get_variance_gain(Y, pref, suf)
            if(current_gain >= max_gain):
                max_gain = current_gain
                max_col = col
                slice_point = (current_col[j] + current_col[j+1])/2
         
    return max_col, slice_point, max_gain


class DecisionTree():
    def __init__(self, metadata, max_level=30):
        self.id = CURR_ID
        self.metadata = metadata #metadata is countries list
        self.level = 0
        self.max_level = max_level
        self.children = []
        # initialise tree and split by countries
        pass

    def train(self, data, country_data):
        global CURR_ID
        for i in self.metadata :
            child_data = data[np.array([j for j, x in enumerate(country_data) if x==i])]
            CURR_ID += 1
            self.children.append(Node(child_data, self.level+1, self.max_level, CURR_ID, self.id))
        for i in self.children :
            if(self.level < self.max_level) :
                i.set_children()

    def show(self, PATH):
        graph = Digraph(filename=PATH)
        graph.node(name=self.id, label="Countries")
        with graph.subgraph() as subgraph :
            subgraph.attr(rank="same")
            for (i,child) in enumerate(self.children) :
                child.show(subgraph,graph,self.metadata[i])
        return

    def predict(self, data, country_data):
        preds = []
        for (i,v) in enumerate(country_data) :
            child = self.children[self.metadata.index(v)]
            preds.append(child.predict(data[i]))
        return np.array(preds)
    
    def test(self, data, country_data) :
        #test on test data and return mse loss and r2 value
        target = data[:,-1]
        preds = self.predict(data, country_data)
        mse = ((preds-target)**2).mean()
        preds = preds-preds.mean()
        target = target-target.mean()
        return np.dot(preds, target)/(np.sqrt((preds**2).sum())*np.sqrt((target**2).sum())+1e-6), mse

    def prune_tree(self) :
        #prune the tree
        pass
    
    def save(self, PATH) :
        with open(PATH, "w") as fout :
            pickle.dump(self, fout)
        return
        


class Node():
    def __init__(self, data, level, max_level, id,  parent_id):
        self.id = id
        self.parent_id = parent_id
        self.data = data
        self.level = level
        self.max_level = max_level
        self.left_child = None
        self.right_child = None
        pass

    def set_children(self):
        global CURR_ID
        self.attr, self.value, _ = get_max_variance_gain(self.data)
        if self.level < self.max_level :
            child_data = data[np.where(self.data[:,self.attr]<=self.value)]
            CURR_ID+=1
            self.left_child = Node(child_data, self.level+1, self.max_level, CURR_ID, self.id)
            child_data = data[np.where(self.data[:,self.attr]>self.value)]
            CURR_ID+=1
            self.right_child = Node(child_data, self.level+1, self.max_level, CURR_ID, self.id)
            self.left_child.set_children()
            self.right_child.set_children()
        else :
            self.attr = 3
            self.value = self.data[:,3].mean()
        return

    def show(self, graph, master_graph, edge_attr):
        graph.node(name=self.id, label=f"{get_col_label(self.attr)}:{self.value}")
        master_graph.edge(self.id, self.parent_id, edge_attr=edge_attr)
        with graph.subgraph() as subgraph :
            subgraph.attr(rank="same")
            if self.left_child :
                self.left_child.show(subgraph, master_graph, "<")
            if self.right_child :
                self.left_child.show(subgraph, master_graph, ">")
        return

    def predict(self, data):
        if self.left_child == None and self.right_child == None :
            return self.value
        elif data[self.attr] <= self.value :
            return self.left_child.predict(data)
        else :
            return self.right_child.predict(data)

def train_across_splits(data, metadata, MAX_DEPTH) :
    print("Building the tree")
    mse_loss = []
    r2_value = []
    for i in range(10) :
        train_data, test_data, train_country, test_country = split_data(data, random_seed=i)
        tree = DecisionTree(metadata, MAX_DEPTH)
        tree.train(train_data, train_country)
        r2, mse = tree.test(test_data, test_country)
        mse_loss.append(mse)
        r2_value.append(r2)
        print(f"Split:{i+1} MSE:{mse} R2 score:{r2}")
    print(f"Best tree on the basis of mse loss at split = {range(10)[mse_loss.index(min(mse_loss))]}")
    print(f"Best tree on the basis of r2 score at split = {range(10)[r2_value.index(max(r2_value))]}")

def get_best_depth(data, metadata, METRIC) :
    print("Finding best depth...")
    data, metadata = read_data(PATH)
    train_data, test_data, train_country, test_country = split_data(data)
    mse_loss = []
    r2_value = []
    for depth in range(5,50,2) :
        tree = DecisionTree(metadata, depth)
        tree.train(train_data, train_country)
        r2, mse = tree.test(test_data, test_country)
        mse_loss.append(mse)
        r2_value.append(r2)
    plt.subplot(2,1,1)
    
    plt.plot(range(5,50,2), mse_loss)
    plt.title('Mean Squared Error vs Max Depth')
    plt.xlabel('depth')
    plt.ylabel('mse loss')
    
    plt.plot(range(5,50,2), r2_value)
    plt.title("Pearson's Correlation coefficent vs Max Depth")
    plt.xlabel('depth')
    plt.ylabel('r2 score')
    
    print(f"Best tree on the basis of mse loss at depth = {range(5,50,2)[mse_loss.index(min(mse_loss))]}")
    print(f"Best tree on the basis of r2 score at depth = {range(5,50,2)[r2_value.index(max(r2_value))]}")
    
    if METRIC == "mse" :
        return range(5,50,2)[mse_loss.index(min(mse_loss))]
    else :
        return range(5,50,2)[r2_value.index(max(r2_value))]

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--metric",type=str, default="mse")
    '''
    Options:    1. train to max depth
                2. train to given depth
                3. find best depth and plot
                3. train with pruning i.e. variance gain thresholding("check if hypothesis testing to be used")
    '''
    args = parser.parse_args()
    MAX_DEPTH = args.max_depth
    METRIC = args.metric
    PATH = "AggregatedCountriesCOVIDStats.csv"
    
    data, metadata = read_data(PATH)
    train_across_splits(data, metadata, MAX_DEPTH)
    best_depth = get_best_depth(data, metadata, METRIC)
    #prune