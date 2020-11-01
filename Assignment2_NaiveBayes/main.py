import math
import argparse
import numpy as np
import pandas as pd
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
    """Class for a NaiveBayes Classifier
    """
    def __init__(self, X, continuous_vars = [], Y=None):
        """Constructor

        Args:
            X (np.array): features or features+labels if Y is None
            continuous_vars (list, optional): [description]. Defaults to [].
            Y (np.array, optional): labels. Defaults to None.
        """
        self.d = X.shape[1] if Y is None else X.shape[1]+1
        self.Y = X[:,-1] if Y is None else Y
        self.X = X[:,:self.d-1] if Y is None else X
        self.continuous_vars = continuous_vars
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=42)
        self.labels_train = np.array(self.labels_train).reshape(-1,1)
        self.labels_test = np.array(self.labels_test).reshape(-1,1)
        self.values = []
        for i in range(self.d-1):
            self.values.append(list(set(list(X[:,i]))))
        
    def learn(self):
        """Function to train across k-folds and print average accuracy
        """
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
        print("\nAverage train accuracy  = ", average_accuracy)
        
        self.train(self.data_train, self.labels_train)
        test_accuracy = self.test(self.data_test, self.labels_test)

        print("test accuracy = ", test_accuracy)

    def learn_single_fold(self) :
        """Function to train clasifier across a single fold

        Returns:
            float: validation accuracy 
        """
        self.train(self.data_train, self.labels_train)
        val_accuracy = self.test(self.data_test, self.labels_test)
        return val_accuracy
        
    def get_gaussian_prob(self, x, mean, std):
        """Function to get probability from a  gaussian distribution

        Args:
            x (float)
            mean (float)
            std (float)

        Returns:
            float: probability
        """
        return (float)(1.0/(math.sqrt(2.0*math.pi) *std))* math.exp((-(x - mean)**2) / (2*(std**2)))


    def get_class_prob(self, data):
        """Function to get probability of classes
        Args:
            data (np.array)

        Returns:
            list: list of class probabilities i.e P(C_i)
        """
        num = [0,0,0,0]
        total = data.shape[0]
        for i in range(4):
            num[i] = len(np.where(data[:,0]==i)[0]) / total

        return num

    def classify(self, instance):
        """Function to classify a datapoint

        Args:
            instance (np.array): datapoint

        Returns:
            int: predicted class
        """
        probs = []
        for i in range(4):
            p = self.class_prob[i]
            for j in range(instance.shape[1]):
                if (j in self.continuous_vars):
                    p = p * self.get_gaussian_prob(instance[0][j], self.means[i][j], self.std[i][j])
                else:
                    p = p * self.P[i][j][self.values[j].index(instance[0][j])]
            probs.append(p)
        return np.argmax(np.array(probs))

    def get_prob_matrix(self, X_train, Y_train):
        """Function to get probability matrix of features given classes

        Args:
            X_train (np.array): features
            Y_train (np.array): labels

        Returns:
            np.array: probability matrix of features given classes i.e. P(X_i| C_j)
        """
        P = [[0 for j in range(X_train.shape[1])] for i in range(4)]
        
        for i in range(4):
            X_ = X_train[np.where(Y_train[:,0] == i)]
            for j in range(X_train.shape[1]):
                P[i][j] = []
                
                for k in range(len(self.values[j])):
                    p = len(np.where(X_[:, j] == self.values[j][k])[0])/X_.shape[0]
                    P[i][j].append(p)
                P[i][j] = np.array(P[i][j])

        P = np.array(P, dtype=object)
        return P 
    
    def get_mean_std(self, X_train, Y_train):
        """Function to get mean and std deviation for continuous features
        """
        means = [[np.mean(X_train[np.where(Y_train[:,0] == i)][:,j]) for j in range(X_train.shape[1])] for i in range(4)]
        std =  [[np.std(X_train[np.where(Y_train[:,0] == i)][:,j]) for j in range(X_train.shape[1])] for i in range(4)]
        
        return means, std

    def train(self, X_train, Y_train):
        """Function to train the classifier

        Args:
            X_train (np.array): training set features
            Y_train (np.array): training set labels
        """
        self.class_prob = self.get_class_prob(Y_train)
        self.P = self.get_prob_matrix(X_train, Y_train)
        self.means, self.std = self.get_mean_std(X_train, Y_train)
        
    def test(self, X_test, Y_test):
        """Function to test the classifier

        Args:
            X_test (np.array): testing set features
            Y_test (np.array): testing set labels

        Returns:
            float: test accuracy
        """
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
    Eigen_values = pca.singular_values_
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
    plt.ylabel('cumulative explained variance')
    plt.savefig('PCA_1.png')

    plt.clf()

    plt.plot(num_components,Eigen_values)
    plt.xlabel('number of components')
    plt.ylabel('Eigen_values')
    plt.savefig('PCA_2.png')

    Y = Y.reshape((X.shape[0],1))
    X = np.hstack((X, Y))
    NB = NaiveBayes(X,[0,1])

    NB.learn()
    
def remove_outliers(X,Y) :
    """Function to remove outliers

    Args:
        X (np.array): features
        Y (np.array): labels

    Returns:
        np.array: filtered features
        np.array: filtered labels
    """

    std_devs = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
    means = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
    
    X_ = (X-means)/std_devs
    X_ = (np.abs(X) > 3)
    sums = np.sum(X_, axis=1)
    sums = np.where(sums == np.max(sums))
    
    X = np.delete(X, sums, axis=0)
    Y = np.delete(Y, sums, axis=0)
    
    return X, Y

def sequential_backward_selection(X, Y) :

    """Function to perform sequential backward selection of features

    Args:
        X (np.array): features
        Y (np.array): labels

    Returns:
        np.array: filtered features
    """

    features = list(range(X.shape[1]))
    continuous_vars = [2]
    nb = NaiveBayes(X, continuous_vars, Y)
    curr_acc = nb.learn_single_fold()
    while True :
        accs = []
        for i in range(X.shape[1]) :
            temp_X = np.delete(X, i, axis=1)
            nb = NaiveBayes(temp_X, [(j if j<i else j-1) for j in continuous_vars if j!=i], Y)
            accs.append(nb.learn_single_fold())
        accs = np.array(accs)
        accs_improvement = accs-curr_acc
        remove_col = np.argmax(accs_improvement)
        
        if accs_improvement[remove_col] < 0 or X.shape[1]==1:
            break
        curr_acc = accs[remove_col]
        features.pop(remove_col)
        X = np.delete(X, remove_col, axis=1)
        continuous_vars = [(j if j<remove_col else j-1) for j in continuous_vars if j!=remove_col]
    return X, continuous_vars, features
    
def FeatureRemoval(X, features_names) :
    """Function to perform feature removal and retrain the classfier

    Args:
        X (np array) : input data 
        features_names (list) : column names
    """
    
    Y = X[:, 9]
    X = X[:, :9]
    
    print("Removing outliers i.e. samples with max features beyond 3*std_dev..")
    print("Samples before removal: {}".format(X.shape[0]))
    X, Y = remove_outliers(X, Y)
    print("Samples after removal: {}\n".format(X.shape[0]))
    
    print("Sequential backward selection...\n")
    print("Initial features: ", ", ".join([v for i,v in enumerate(features_names)]),"\n")
    
    X, continuous_vars, features = sequential_backward_selection(X, Y)
    
    print("Remaining features: ", ", ".join([v for i,v in enumerate(features_names) if i in features]),"\n")
    
    NB = NaiveBayes(X, continuous_vars, Y)
    NB.learn()


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="Train_A.csv")
    args = parser.parse_args()
    PATH = args.data_path
    data = read_data(PATH)
    print("-"*20)
    print("Training across K-Folds...")
    NB = NaiveBayes(np.array(data, dtype = int)[:,1:], [2, 5, 7])
    NB.learn()
    print("-"*20)
    print("PCA Analysis...")
    PCA_analysis(np.array(data, dtype = int)[:,1:])
    print("-"*20)
    print("Feature Removal...")
    FeatureRemoval(np.array(data, dtype = int)[:,1:], data.columns[1:])
