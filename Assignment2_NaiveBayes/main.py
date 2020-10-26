import pickle
import argparse
import numpy as np
import pandas as pd
from graphviz import Digraph
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
    def __init__(self):
        pass
    

    def train():
        pass
    def test():
        pass

    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="Train_A.csv")
    args = parser.parse_args()
    PATH = args.data_path
    data = read_data(PATH)
