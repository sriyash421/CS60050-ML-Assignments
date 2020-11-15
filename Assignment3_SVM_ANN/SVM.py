import math
import argparse
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def read_data(PATH) :
    """Function to read data from file

    Args:
        PATH (str): PATH to file containing data

    Returns:
        df: numpy array after performing normalization
    """
    df = pd.read_csv(PATH)
    df = df.to_numpy()
    labels = df[:,-1]
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df[:,:-1], labels



def SVM(X, y):
    """Function to perform classification using SVM
    Args:
        X (numpy array) : input_feature vector
        Y (numpy array) : input labels
    
    """
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=11)
    g_max = 0.0
    k_accuracies = []
    print("Using SVM classifier ... ")
    for kernel in ['linear', 'poly', 'rbf']:
        max_accuracy = 0.0
        accuracies = []
        for i in trange(1, 41):
            C = 0.05*i
            clf = SVC(C = C, kernel= kernel, max_iter=2000, degree=2)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            max_accuracy = max(max_accuracy, accuracy)
            accuracies.append(accuracy)
        k_accuracies.append(accuracies)
        kernel_ = kernel if kernel != 'poly' else 'Quadratic'
        kernel_ = kernel_ if kernel_ != 'rbf' else 'Radial basis function'
        C_ = (accuracies.index(max_accuracy) + 1)*0.05
        
        clf = SVC(C = C_, kernel= kernel, max_iter=2000, degree=2)
        clf.fit(X_train, y_train)
        acc_train = clf.score(X_train, y_train)
        
        print("Max_Test_Accuracy for kernel =",kernel_, " : ", max_accuracy, "at C = ", C_)
        print("Train_Accuracy for best C =", acc_train)
        g_max = max(g_max, max_accuracy)
    print("Overall Max_Test_Accuracy: ", g_max)
    C_values = [0.05*i for i in range(1, 41)]

    plt.figure(figsize = (10, 10))
    plt.plot(C_values,k_accuracies[0], label = 'linear')
    plt.plot(C_values,k_accuracies[1], label = 'Quadratic')
    plt.plot(C_values,k_accuracies[2], label = 'Radial basis function')
    plt.xlabel('C values')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('SVM.png')





if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="spambase.data")
    args = parser.parse_args()
    PATH = args.data_path
    data, labels = read_data(PATH)
    SVM(data, labels)
