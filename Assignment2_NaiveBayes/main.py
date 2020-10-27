import pickle
import argparse
import numpy as np
import pandas as pd
from graphviz import Digraph
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA



# plt.ion()

def fill_missing_values(df):

    """Function to read data from file
    Args:
        df (Dataframe) : input dataframe 

    Returns:
        df: data frame with missing values replaced with MODE of that column
        
    """

    for col_name in df.columns:
        df[col_name].fillna(df[col_name].mode()[0], inplace=True) 
    return df

def encode(df):
    """Function to read data from file

    Args:
        df (Dataframe) : input dataframe 

    Returns:
        df: data frame with encoded labels
        
    """

    Encoder = preprocessing.LabelEncoder()
    for col_name in ["Gender", "Ever_Married", "Graduated","Profession", "Spending_Score", "Var_1", "Segmentation"]:
        df[col_name] = Encoder.fit_transform(df[col_name])
    return df


def read_data(PATH) :
    """Function to read data from file

    Args:
        PATH (str): PATH to file containing data

    Returns:
        df: data frame with mssing values filled and encoded labels 
    """

    df = pd.read_csv(PATH)
    df = fill_missing_values(df)
    df = encode(df)
    return df


class NaiveBayes():
    def __init__(self, X):
        self.d = X.shape[1]
        self.Y = X[:,-1]
        self.X = X[:,:self.d-1]
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=42)
        self.labels_train = np.array(self.labels_train).reshape(-1,1)
        self.labels_test = np.array(self.labels_test).reshape(-1,1)
        self.values = []
        for i in range(self.d-1):
            self.values.append(list(set(list(X[:,i]))))
        
    def learn(self):
        kf = KFold(n_splits=5)
        i =  0
        accuracies = []
        for train_index, test_index in kf.split(self.data_train):
            X_train, X_test = self.data_train[train_index], self.data_train[test_index]
            y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]
            self.train(X_train, y_train)
            accuracy = self.test(X_test, y_test)
            i = i + 1
            print(f"Accuracy at the {i} split  = ", accuracy)
            accuracies.append(accuracy)

        average_accuracy = sum(accuracies)/len(accuracies)
        print("Average train accuracy  = ", average_accuracy)
        
        self.train(self.data_train, self.labels_train)
        accuracy = self.test(self.data_test, self.labels_test)

        print("test accuracy = ", accuracy)


    def get_class_prob(self, data):
        num = [0,0,0,0]
        total = data.shape[0]
        for i in range(4):
            num[i] = len(np.where(data[:,0]==i)[0]) / total

        return num

    def classify(self, instance):
        probs = []
        for i in range(4):
            p = self.class_prob[i]
            for j in range(instance.shape[1]):
                p = p * self.P[i][j][self.values[j].index(instance[0][j])]
            probs.append(p)
        return np.argmax(np.array(probs))

    def get_prob_matrix(self, X_train, Y_train):
        P = [[0 for j in range(X_train.shape[1])] for i in range(4)]
        
        for i in range(4):
            X_ = X_train[np.where(Y_train[:,0] == i)]
            for j in range(X_train.shape[1]):
                P[i][j] = []
                
                for k in range(len(self.values[j])):
                    p = len(np.where(X_[:, j] == self.values[j][k])[0])/X_.shape[0]
                    P[i][j].append(p)
                P[i][j] = np.array(P[i][j])

        P = np.array(P)
        return P 

    def train(self, X_train, Y_train):
        self.class_prob = self.get_class_prob(Y_train)
        self.P = self.get_prob_matrix(X_train, Y_train)

    def test(self, X_test, Y_test):
        count = 0
        
        for i in range(X_test.shape[0]):
            if(self.classify(np.array([X_test[i]])) == Y_test[i][0]):
                count += 1
        
        accuracy = count / X_test.shape[0]

        return accuracy

    

def PCA_analysis(X):
    """Function to perform PCA preserving 95% variance 

    Args:
        X (np array) : input data 

    """
    Y = X[:,9]
    X = X[:, :9]
    pca = PCA()
    pca.fit(X)
    variances = pca.explained_variance_ratio_
    variance_kept = 0.0
    i = 0
    while(variance_kept < 0.95):
        variance_kept += variances[i]
        i = i + 1
    pca = PCA(n_components = i)
    X = pca.fit_transform(X)
    num_components = [i+1 for i in range(9)]
    plt.plot(num_components,np.cumsum(variances))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.savefig('PCA.png')
    Y = Y.reshape((X.shape[0],1))
    X = np.hstack((X, Y))
    NB = NaiveBayes(X)
    NB.learn()


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="Train_A.csv")
    args = parser.parse_args()
    PATH = args.data_path
    data = read_data(PATH)
    NB = NaiveBayes(np.array(data, dtype = int)[:,1:])
    NB.learn()
    PCA_analysis(np.array(data, dtype = int)[:,1:])
