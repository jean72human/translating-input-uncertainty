import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge, LinearRegression

from models import TIUReg
from sklearn.metrics import mean_squared_error

import wandb
api = wandb.Api()

import sys
import os

experiment = int(sys.argv[1])
scale = str(sys.argv[2])

def main():
    dirname = os.path.dirname(__file__)
    for init in ['random','closed form']:
        for opt in ['sgd','adam']:
            for lr in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-7,1e-9]:
                for norm in [True,False]:
                    for true_sigma in [True,False]:
                        for batch in [True,False]:
                            tags = [
                                "init "+init,
                                "opt "+opt,
                                "lr "+str(lr),
                                "norm "+str(norm),
                                "sigma "+str(true_sigma),
                                "batch "+str(batch)
                            ]
                            run = wandb.init(reinit=True, name=" ".join(tags), project="translating-input-uncertainty", tags=tags)

                            with run:

                                filename_train = os.path.join(dirname,f"data/exp{experiment}_data_train_{scale}.csv")
                                filename_test = os.path.join(dirname,f"data/exp{experiment}_data_test_{scale}.csv")

                                filename_train_s = os.path.join(dirname,f"data/exp{experiment}_data_train_s_{scale}.csv")
                                filename_test_s = os.path.join(dirname,f"data/exp{experiment}_data_test_s_{scale}.csv")

                                data = pd.read_csv(filename_train)
                                data_test = pd.read_csv(filename_test)
                                variable_names = ['one','two','three','four']

                                X_train, X_test = data[variable_names].values, data_test[variable_names].values
                                Y_train, Y_test = data.y.values, data_test.y.values

                                X_train = torch.from_numpy(X_train).float()
                                X_test = torch.from_numpy(X_test).float()
                                Y_train = torch.from_numpy(Y_train).unsqueeze(-1).float()
                                Y_test = torch.from_numpy(Y_test).unsqueeze(-1) .float()

                                if norm:
                                    X_test = (X_test - X_train.min(0)[0])/(X_train.max(0)[0] - X_train.min(0)[0])
                                    X_train = (X_train - X_train.min(0)[0])/(X_train.max(0)[0] - X_train.min(0)[0])

                                if true_sigma:
                                    S_train = torch.diag_embed(torch.from_numpy(pd.read_csv(filename_train_s).values).float())**2
                                    S_test = torch.diag_embed(torch.from_numpy(pd.read_csv(filename_test_s).values).float())**2
                                else:        
                                    S_ = torch.from_numpy(np.cov(X_train.T))
                                    S_train = S_.unsqueeze(0).repeat([X_train.size(0),1,1])
                                    S_test = S_.unsqueeze(0).repeat([X_test.size(0),1,1])

                                model = TIUReg()
                                params ={
                                    "X":X_train,
                                    "Y":Y_train,
                                    "S":S_train.float(),
                                    "X_test":X_test,
                                    "Y_test":Y_test,
                                    "n_iter":1500,
                                    "lr":lr,
                                    "init":init,
                                    "run":run,
                                    "optimizer":opt,
                                    "batch":batch
                                }
                                model.fit(**params)
                        
    
if __name__ == "__main__":
    main()
