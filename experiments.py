import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge, LinearRegression

from models import TIUReg
from sklearn.metrics import mean_squared_error

import sys

n_tries = int(sys.argv[1]) 
experiment = int(sys.argv[2])
scale = str(sys.argv[3])

true_sigma = int(sys.argv[4])

normalization = str(sys.argv[5])

def add_ones(X): return torch.cat([X,np.ones((X.shape[0],1))],dim=1)

def main():
    
    filename_train = f'data/exp{experiment}_data_train_{scale}.csv'
    filename_test = f'data/exp{experiment}_data_test_{scale}.csv'
    
    filename_train_s = f'data/exp{experiment}_data_train_s_{scale}.csv'
    filename_test_s = f'data/exp{experiment}_data_test_s_{scale}.csv'
        
    data = pd.read_csv(filename_train)
    data_test = pd.read_csv(filename_test)
    variable_names = ['one','two','three','four']

    X_train, X_test = data[variable_names].values, data_test[variable_names].values
    Y_train, Y_test = data.y.values, data_test.y.values

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).unsqueeze(-1).float()
    Y_test = torch.from_numpy(Y_test).unsqueeze(-1) .float()

        
    #S_train /= torch.abs(S_train).max()
    #S_test /= torch.abs(S_test).max()
    
    if normalization=='max':       
        X_test = X_test/X_train.max(0)[0]
        X_train = X_train/X_train.max(0)[0]
    elif normalization=='std':
        X_test = (X_test - X_train.mean(0))/X_train.std(0)
        X_train = (X_train - X_train.mean(0))/X_train.std(0)
    elif normalization=='minmax':
        X_test = (X_test - X_train.min(0)[0])/(X_train.max(0)[0] - X_train.min(0)[0])
        X_train = (X_train - X_train.min(0)[0])/(X_train.max(0)[0] - X_train.min(0)[0])
        
    if true_sigma==1:
        S_train = torch.diag_embed(torch.from_numpy(pd.read_csv(filename_train_s).values).float())**2
        S_test = torch.diag_embed(torch.from_numpy(pd.read_csv(filename_test_s).values).float())**2
    else:        
        S_ = torch.from_numpy(np.cov(X_train.T))
        print(S_)
        S_train = S_.unsqueeze(0).repeat([X_train.size(0),1,1])
        S_test = S_.unsqueeze(0).repeat([X_test.size(0),1,1])

    train_mse = []
    test_mse = []
    print("Training Linear Regression")
    for _ in range(n_tries):
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train,Y_train.squeeze())
        train_mse.append(mean_squared_error(model.predict(X_train),Y_train).item())
        test_mse.append(mean_squared_error(model.predict(X_test),Y_test).item())

    print(f"TRAIN performance MSE: {np.mean(train_mse)} +- {np.std(train_mse)}")
    print(f"TEST performance MSE: {np.mean(test_mse)} +- {np.std(test_mse)}")


    train_mse = []
    test_mse = []
    print("Training Bayesian Ridge Regression")
    for _ in range(n_tries):
        model = BayesianRidge(fit_intercept=False)
        model.fit(X_train,Y_train.squeeze())
        train_mse.append(mean_squared_error(model.predict(X_train),Y_train).item())
        test_mse.append(mean_squared_error(model.predict(X_test),Y_test).item())

    print(f"TRAIN performance MSE: {np.mean(train_mse)} +- {np.std(train_mse)}")
    print(f"TEST performance MSE: {np.mean(test_mse)} +- {np.std(test_mse)}")


    train_mse = []
    test_mse = []
    print("Training TIU Regression")
    for _ in range(n_tries):
        model = TIUReg()
        model.fit(X_train,Y_train,S_train.float(),X_test,Y_test,n_iter=500,lr=1e-2,viz=True,optimizer="adam",batch=True)#,init='closed form'
        train_mse.append(mean_squared_error(model.predict(X_train).detach(),Y_train).item())
        test_mse.append(mean_squared_error(model.predict(X_test).detach(),Y_test).item())

    print(f"TRAIN performance MSE: {np.mean(train_mse)} +- {np.std(train_mse)}")
    print(f"TEST performance MSE: {np.mean(test_mse)} +- {np.std(test_mse)}")
    
if __name__ == "__main__":
    main()
