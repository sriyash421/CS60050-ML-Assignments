import json
import math
import random
from time import time
import argparse
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from sklearn.preprocessing import StandardScaler
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.set_deterministic(True)

def read_data(PATH):
    """Function to read data from file

    Args:
        PATH (str): PATH to file containing data

    Returns:
        df: numpy array after performing normalization
    """
    df = pd.read_csv(PATH)
    df = df.to_numpy()
    labels = df[:, -1]
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df[:, :-1], labels


class SpamData(Dataset):
    # Dataset Class
    def __init__(self, PATH):
        super(SpamData, self).__init__()
        self.data, self.labels = read_data(PATH)
        self.data = torch.from_numpy(self.data).float()
        self.labels =  torch.from_numpy(self.labels).float()
    
    def input_size(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Net(nn.Module):
    # Class containing the AAN
    def __init__(self, input_size, layers=[]):
        super(Net, self).__init__()
        if len(layers) == 0:
            self.model = nn.Sequential(
                nn.Linear(input_size, 1)
            )
        else:
            self.model = [nn.Linear(input_size, layers[0]), nn.ReLU(True)]
            for i in range(1, len(layers)):
                self.model.extend([nn.Linear(layers[i-1], layers[i])])
            self.model.append(nn.Linear(layers[-1], 1))
            self.model = nn.Sequential(*self.model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.model(X)
        return self.sigmoid(X)


def get_loss(model, batch, loss_fn, device):
    # Function to run the model on the given batch
    input = batch[0].to(device)
    target = batch[1].to(device)
    output = model(input).squeeze()
    loss = loss_fn(output, target)
    accuracy = ((output >= 0.5).float() == target).float().mean()
    return loss, accuracy

def run(model, train_data, test_data, optim, device):
    # Function to train the model on the given data
    loss_fn = nn.BCELoss()
    stats = {"train_loss":[], "test_loss":[], "test_acc":[]}
    epoch = 0
    
    # Convergence criteria
    tolerance = 1e-4
    max_iter = 10
    counter = 0
    
    epoch = 0
    for epoch in range(500) :
        model.train()
        losses = []
        for _, batch in enumerate(train_data) :
            loss, _ = get_loss(model, batch, loss_fn, device)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        stats["train_loss"].append(sum(losses)/len(losses))
        
        if epoch > 2 and stats["train_loss"][-2]-stats["train_loss"][-1] < tolerance :
            counter+=1
        if counter == max_iter :
            break
    
    losses = []
    accs = []
    model.eval()
    for _, batch in enumerate(test_data) :
        with torch.no_grad() :
            loss, acc = get_loss(model, batch, loss_fn, device)
            losses.append(loss.item())
            accs.append(acc.item())
    stats["test_loss"].append(sum(losses)/len(losses))
    stats["test_acc"].append(sum(accs)/len(accs))
    
    return stats["test_loss"][-1], stats["test_acc"][-1], epoch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="spambase.data")
    args = parser.parse_args()

    dataset = SpamData(args.data_path)
    input_size = dataset.input_size()
    test_size = int(0.2*len(dataset))
    train_data, test_data = random_split(dataset, [len(dataset)-test_size, test_size])

    train_data = DataLoader(train_data, batch_size=128, shuffle=True)
    test_data = DataLoader(test_data, batch_size=128, shuffle=False)
    
    model_architectures = [[], [2], [6], [2,3], [3,2]]
    lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    results = []
    
    for arch in model_architectures :
        # Looping over given architectures
        model = Net(input_size, arch)
        model_info = {
            "architecture":arch,
            "trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "optim": "SGD",
            "loss_fn": "BCE Loss",
            "weight_decay":0.01,
            "activation":"ReLU",
            "output_activation": "Sigmoid",
            "batch_size": 32,
            "accuracies_lr":[],
            "losses_lr":[],
            "epochs":[],
            "best_accuracy":None,
            "gpu_time":None,
            "cpu_time":None
        }
        cpu_times = []
        gpu_times = []
        
        for lr in tqdm(lrs) :

            # Using GPU
            # print("Training model on gpu...")
            time_ = time()
            device = torch.device("cuda:0")
            model = Net(input_size, arch).to(device)
            opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
            loss, acc, e = run(model, train_data, test_data, opt, device)
            gpu_times.append(time()-time_)
            model_info["accuracies_lr"].append(acc)
            model_info["losses_lr"].append(loss)
            model_info["epochs"].append(e)
            
            # Using CPU
            # print("Training model on cpu...")
            time_ = time()
            device = torch.device("cpu")
            model = Net(input_size, arch).to(device)
            opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
            loss, acc, _ = run(model, train_data, test_data, opt, device)
            cpu_times.append(time()-time_)
        
        model_info["gpu_time"] = max(gpu_times)
        model_info["cpu_time"] = max(cpu_times)
        model_info["best_accuracy"] = max(model_info["accuracies_lr"])
        results.append(model_info)
        
        print(f"Trained on a model: \n{json.dumps(model_info, indent=4)}")
        
    
    # Plotting acc vs lr, for each model
    plt.figure(figsize = (10, 10))
    for i, info in enumerate(results) :
        plt.plot(lrs, info["accuracies_lr"], label = f'Model {i+1}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.savefig('ACC_LR.png')
    
    # Plotting acc vs model, for each learning rate
    fig, ax = plt.subplots()
    for i, lr in enumerate(lrs) :
        accs = [info["accuracies_lr"][i] for info in results]
        ax.plot(range(1, 1+len(model_architectures)), accs, label = f'lr = {lr}')
    ax.set_xlabel('Model')
    ax.set_xticks(range(1, 1+len(model_architectures)))
    ax.set_xticklabels([f"Model_{i+1}" for i in range(len(model_architectures))])
    ax.set_ylabel('Accuracy')
    plt.legend()
    plt.savefig('ACC_MODEL.png')
    
    # Plotting average gpu vs cpu time for each model
    fig, ax = plt.subplots()
    ax.plot(range(1, 1+len(model_architectures)), [info["gpu_time"] for info in results], label = f'GPU')   
    ax.plot(range(1, 1+len(model_architectures)), [info["cpu_time"] for info in results], label = f'CPU')
    ax.set_xlabel('Model')
    ax.set_xticks(range(1, 1+len(model_architectures)))
    ax.set_xticklabels([f"Model_{i+1}" for i in range(len(model_architectures))])
    ax.set_ylabel('Average Execution Time per step')
    plt.legend()
    plt.savefig('EXECUTION_TIME.png')
    
    
    #Best model
    max_acc = 0
    best_model = None
    for info in results :
        temp_max = max(info["accuracies_lr"])
        index_ = info["accuracies_lr"].index(temp_max)
        if(temp_max > max_acc) :
            max_acc = temp_max
            best_model = {
                "architecture":info["architecture"],
                "trainable_params":info["trainable_params"],
                "optim":info["optim"],
                "loss_fn":info["loss_fn"],
                "weight_decay":info["weight_decay"],
                "activation":info["activation"],
                "output_activation":info["output_activation"],
                "epochs":info["epochs"][index_]+1,
                "batch_size":info["batch_size"],
                "lr": lrs[index_],
                "loss": info["losses_lr"][index_],
                "accuracy":temp_max,
            }
    print("-"*10,"BEST MODEL","-"*10)
    print(json.dumps(best_model, indent=4))        
        
