import pickle
import argparse
import numpy as np
import pandas as pd
from graphviz import Digraph
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
# plt.ion()

CURR_ID = 0

def get_col_label(i) :
    """Function to get column label
    
    Args:
        i (int): index of feature column

    Returns:
        str: name of the column
    """

    temp = ["Date","Confirmed","Recovery","Deaths"]
    
    return temp[i]

def preprocess_data(df) :
    """Function to preprocess date to a continuous variable

    Args:
        df (pandas.DataFrame): data read from file

    Returns:
        int: data, which has date replaced by integer value
    """
    
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m%d%Y").astype(int)
    
    return df

def read_data(PATH) :
    """Function to read data from file

    Args:
        PATH (str): PATH to file containing data

    Returns:
        df: data frame containing data
        country_list: metadata from the read data
    """
    df = pd.read_csv(PATH)
    df = preprocess_data(df)
    return df, list(df["Country"].unique())

def split_data(data, split_ratio=[0.8,0.2], random_seed = 0) :
    """Function to generate splits for training, testing and/or pruning

    Args:
        data (pandas.DataFrame): data frame containing input data
        split_ratio (list, optional): [description]. Defaults to [0.8,0.2].
        random_seed (int, optional): [description]. Defaults to 0.

    Returns:
        [arrays]: features for train, test, val depending on inputs
        [list]: country list for train, test, val depending on inputs
    """
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
    """Function to get variance of vector
    Args:
        data (np.array): column vector

    Returns:
        float: variance of input data
    """
    return np.var(data)

def get_variance_gain(A, B, C):
    """Function to get variance gain between the split A -> B and C

    Args:
        A (np.array): target values for parent
        B (np.array): target values for child
        C (np.array): target values for child

    Returns:
        float: variance gain due to the split
    """
    assert(len(A) == len(B) + len(C))
    return get_variance(A) - (get_variance(B)*(len(B)/(len(C)+len(B))) + get_variance(C)*(len(C)/(len(C)+len(B))))


def get_max_variance_gain(data) :
    """Function to get max variance gain from given data

    Args:
        data (np.array): data stored in a node

    Returns:
        max_col: attr to be used to split
        slice_point: value of the attr to be split
        mean: mean of target
    """

    X = data
    max_col = 0
    max_gain = 0
    slice_point = 0
    if(X.shape[0] == 1):
        return 0,0,np.mean(X[:,0])
    for col in range(3):
        X = X[X[:,col].argsort()]
        Y = X[:, 3]
        current_col = list(X[:, col])
        if np.all(current_col==current_col[0]) :
            continue
        for j in range(0,X.shape[0]-1):
            pref = Y[:j+1]
            suf = Y[j+1:]
            current_gain = get_variance_gain(Y, pref, suf)
            if(current_gain >= max_gain):
                max_gain = current_gain
                max_col = col
                slice_point = (current_col[j] + current_col[j+1])/2
    
    mean = np.mean(data[:,3])
    return max_col, slice_point, mean

def check_confidence(error_child, error_parent) :
    """Function to find 95% interval of error difference

    Args:
        error_child ([float): error of child nodes
        error_parent ([float): error of parent
        n (int): size of sample from data

    Returns:
        bool: decides to prune or not
    """
    error_diff = np.mean(error_child-error_parent)
    variance =  np.var(error_child)+np.var(error_parent)
    left_limit = error_diff - 1.96*variance
    right_limit = error_diff + 1.96*variance
    return (left_limit >=0 and right_limit>=0)



class DecisionTree():
    """Class to create a Decision Tree
    """
    def __init__(self, metadata, max_level=30):
        """

        Args:
            metadata (list(str)): list of countries
            max_level (int, optional): max depth of tree allowed. Defaults to 30.
        """
        self.id = CURR_ID
        self.metadata = metadata
        self.level = 0
        self.max_level = max_level
        self.children = []

    def train(self, data, country_data):
        """Function to train the tree
        Args:
            data (np.array): training data
            country_data (list(str)): country value for each training data
        """
        global CURR_ID
        self.height = 0
        for i in self.metadata :
            child_data = data[np.array([j for j, x in enumerate(country_data) if x==i])]
            CURR_ID += 1
            self.children.append(Node(child_data, self.level+1, self.max_level, CURR_ID, self.id))
        for i in self.children :
            if(self.level < self.max_level) :
                self.height = max(self.height, i.set_children())

    def show(self, PATH):
        """Function to print the tree

        Args:
            PATH (str): path to store the image
        """
        graph = Digraph(filename=PATH, format='png')
        graph.node(name=str(self.id), label="Countries")
        for (i,child) in enumerate(self.children) :
            if i >2 : break #displaying only 2 because anything bigger leads to extremely low resolution
            child.show(graph,self.metadata[i])
        graph.view()
        return


    def predict(self, data, country_data):
        """Function to predict the deaths for given data

        Args:
            data (np.array): data to be used for predictions
            country_data (list(str)): corresponding country values

        Returns:
            np.array: predicted  values for the input data
        """
        preds = []
        for (i,v) in enumerate(country_data) :
            child = self.children[self.metadata.index(v)]
            preds.append(child.predict(data[i]))
        return np.array(preds)
    
    def test(self, data, country_data) :
        """Function to test the data

        Args:
            data (np.array): data to be used for predictions
            country_data (list(str)): corresponding country values

        Returns:
            float: mean squared error of the tree on the given test data
        """
        target = data[:,-1]
        preds = self.predict(data, country_data)
        return np.mean(np.power(preds-target,2))

    def prune_tree(self, data, country_data):
        """Function to prune tree

        Args:
            data (np.array): data to be used for pruning
            country_data (list(str)): corresponding country values
        """
        nodes_pruned = 0
        total_nodes = 0
        for v in self.metadata:
            child = self.children[self.metadata.index(v)]        
            current_data = data[country_data == v,:]
            nodes_pruned += child.subtree
            total_nodes += child.subtree
            child.prune_node(current_data)
            nodes_pruned -= child.subtree
        return total_nodes, nodes_pruned
    
    def save(self, PATH) :
        """Function to save tree

        Args:
            PATH (str): path to save tree
        """
        with open(PATH, "w") as fout :
            pickle.dump(self, fout)
        return
        


class Node():
    """Class to store a node in the decision tree
    """
    def __init__(self, data, level, max_level, id,  parent_id):
        self.id = id
        self.parent_id = parent_id
        self.data = data
        self.level = level
        self.max_level = max_level
        self.left_child = None
        self.right_child = None
        self.subtree = 1
        pass

    def set_children(self):
        """function to generate subtree for a node
        Returns:
            int: Height of node after creating subtrees
        """
        global CURR_ID
        self.attr, self.value, self.mean = get_max_variance_gain(self.data)
        if np.all(self.data[:,3]==self.data[0,3]) or self.value == np.max(self.data[:,self.attr]) or self.value == np.min(self.data[:,self.attr]) or self.level == self.max_level:
            self.attr = 3
            self.value = np.mean(self.data[:,3])
            self.height = 0
            self.subtree = 1
        elif self.level < self.max_level :
            child_data = self.data[np.where(self.data[:,self.attr]<=self.value)]
            CURR_ID+=1
            self.left_child = Node(child_data, self.level+1, self.max_level, CURR_ID, self.id)
            child_data = self.data[np.where(self.data[:,self.attr]>self.value)]
            CURR_ID+=1
            self.right_child = Node(child_data, self.level+1, self.max_level, CURR_ID, self.id)
            self.height = max(self.left_child.set_children(),self.right_child.set_children())
            self.subtree = 1 + self.left_child.subtree + self.right_child.subtree
        return 1 + self.height

    def show(self, graph, edge_attr):
        """Function to add node to the graph of the tree

        Args:
            graph (Digraph object): graph object of the decision tree
            edge_attr (str): label of edge to parent
        """

        graph.node(name=str(self.id), label=f"{get_col_label(self.attr)}:{self.value}")
        graph.edge(str(self.parent_id), str(self.id), label=edge_attr)
        if self.left_child :
            self.left_child.show(graph, "<")
        if self.right_child :
            self.right_child.show(graph, ">")
        return

    def predict(self, data):
        """Function to predict the deaths for given data
        Args:
            data (np.array): single data point to be used for predictions

        Returns:
            float: predicted  value for the input data
        """
        if self.left_child == None and self.right_child == None :
            return self.value
        elif data[self.attr] <= self.value :
            return self.left_child.predict(data)
        else :
            return self.right_child.predict(data)
    
    def prune_node(self, data):
        """Function to prune a node

        Args:
            data (np.array): data to be used for pruning
        """
        if (self.left_child == None or self.right_child == None):
            return 
        Y_left = data[data[:, self.attr]<=self.value,3]
        Y_right = data[data[:, self.attr]>self.value,3]
        self.left_child.prune_node(data[data[:, self.attr]<=self.value,:])
        self.right_child.prune_node(data[data[:, self.attr]>self.value,:])
        children_error = np.concatenate((np.power(np.subtract(Y_left , self.left_child.mean), 2), np.power(np.subtract(Y_right , self.right_child.mean), 2)),axis=0)
        parent_error = np.power(np.subtract(data[:,3] , self.mean), 2)
        if(check_confidence(children_error,parent_error)):
            self.left_child = None
            self.right_child = None
            self.value = np.mean(self.data[:,3])
            self.attr = 3
            self.subtree  = 1
        else:
            self.subtree = 1 + self.left_child.subtree + self.right_child.subtree
        return 
        

def train_across_splits(data, metadata, MAX_DEPTH) :
    """Function to train trees according different splits

    Args:
        data (pandas.DataFrame): data read from file
        metadata (list(str)): metadata of the given data
        MAX_DEPTH (int): max depth for the tree
    """
    print("Building trees across splits")
    mse_loss = []
    for i in trange(10) :
        train_data, test_data, train_country, test_country = split_data(data, random_seed=i)
        tree = DecisionTree(metadata, MAX_DEPTH)
        tree.train(train_data, train_country)
        mse = tree.test(test_data, test_country)
        mse_loss.append(mse)
        print("Split:{} MSE:{:8e} Height of tree: {}".format(i, mse, tree.height))
    print("Best tree on the basis of mse loss at split = {}".format(range(10)[mse_loss.index(min(mse_loss))]))
    print(f"Best mse: {min(mse_loss)}")
    return range(10)[mse_loss.index(min(mse_loss))]

def get_best_depth(data, metadata, random_seed) :
    """Function to plot depth vs loss and find best tree

    Args:
        data (pandas.DataFrame): data read from file
        metadata (list(str)): metadata of the given data

    Returns:
        int: best depth for a tree on the given data
    """
    print("Finding best depth...")
    train_data, test_data, train_country, test_country = split_data(data,random_seed=random_seed)
    mse_loss = []
    depth_list = list(range(1,20))
    for depth in tqdm(depth_list):
        tree = DecisionTree(metadata, depth)
        tree.train(train_data, train_country)
        mse = tree.test(test_data, test_country)
        mse_loss.append(mse)
    plt.subplot(1,1,1)
    plt.plot(depth_list, list(np.array(mse_loss)/10e4))
    plt.title("Mean Squared Error vs Max Depth")
    plt.xlabel("depth"), plt.ylabel("mse loss (x 10e4)")
    plt.savefig("plot.png")
    
    print("Best tree on the basis of mse loss at depth = {}".format(depth_list[mse_loss.index(min(mse_loss))]))
    print(f"Least mse: {min(mse_loss)}")
    
    return depth_list[mse_loss.index(min(mse_loss))]
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--data_path",type=str, default="AggregatedCountriesCOVIDStats.csv")
    args = parser.parse_args()
    MAX_DEPTH = args.max_depth
    if(MAX_DEPTH <  0):
        MAX_DEPTH = 15
    PATH = args.data_path
    
    data, metadata = read_data(PATH)
    best_split = train_across_splits(data, metadata, MAX_DEPTH)
    best_depth = get_best_depth(data, metadata, random_seed=best_split)
    (train, cross, test, train_country, cross_country, test_country) = split_data(data,split_ratio=[0.6,0.2,0.2],random_seed=best_split)
    tree = DecisionTree(metadata, best_depth)
    tree.train(train, train_country)
    tree.show("original_tree")
    mse_before = tree.test(test, test_country)
    total_nodes, nodes = tree.prune_tree(cross,cross_country)
    mse = tree.test(test, test_country)
    print(f"Before pruning the mse loss = {mse_before}")
    print("Total nodes before pruning= ", total_nodes)
    print("nodes pruned = ", nodes)
    print(f"After pruning the mse loss = {mse}")
    tree.show("pruned_tree")
