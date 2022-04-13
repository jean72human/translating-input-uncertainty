import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize':(10,5)})

class TIUReg:
    """
    Basic linear regression implemetation using numpy
    """
    def __init__(self):
        """
        Initialization of theta
        """
        self.theta = None
  
    def fit(self,X,Y,S,X_test=None,Y_test=None,n_iter=100,lr=1e-2,init='random',viz=False,optimizer='adam',batch=False,run=None):
        """
        Fit function. Uses gradient descent
        """
        if init=='random':
            self.theta = torch.randn((X.size(1),1), requires_grad=True)
        elif init=='closed form':
            A = X.T @ X #+ 0.01*torch.eye(X.size(1)).float()
            B = X.T @ Y
            self.theta = torch.inverse(A) @ B
            self.theta.requires_grad = True
        if optimizer=='adam':
            optim = torch.optim.Adam([self.theta], lr=lr)
        elif optimizer=='sgd':
            optim = torch.optim.SGD([self.theta], lr=lr)
        
        losses = []
        mses_test = []
        mses = []
        mses.append(mean_squared_error(self.predict(X).detach(),Y))
        for _ in range(n_iter):
            if batch:
                negLogP = self.batch_criterion(X,Y,S)
                negLogP.backward()
                optim.step()
                optim.zero_grad()
                if X_test!=None and Y_test!=None: 
                    mses_test.append(mean_squared_error(self.predict(X_test).detach(),Y_test))
                mses.append(mean_squared_error(self.predict(X).detach(),Y))
                losses.append(negLogP.item())
                if run!=None:
                    run.log({"negative log likelihood":negLogP.item()})
                    run.log({"test MSE":mean_squared_error(self.predict(X_test).detach(),Y_test)})
                    run.log({"train MSE":mean_squared_error(self.predict(X).detach(),Y)})
            else:
                running_loss = 0
                for i in range(X.size(0)):
                    negLogP = self.criterion(X[i].unsqueeze(0),Y[i].unsqueeze(0),S[i].unsqueeze(0))
                    negLogP.backward()
                    optim.step()
                    optim.zero_grad()
                    running_loss += negLogP.item()
                if X_test!=None and Y_test!=None: 
                    mses_test.append(mean_squared_error(self.predict(X_test).detach(),Y_test))
                mses.append(mean_squared_error(self.predict(X).detach(),Y))
                losses.append(running_loss/X.size(0))
                if run!=None:
                    run.log({"negative log likelihood":running_loss/X.size(0)})
                    run.log({"test MSE":mean_squared_error(self.predict(X_test).detach(),Y_test)})
                    run.log({"train MSE":mean_squared_error(self.predict(X).detach(),Y)})
        if viz:
            print("Negative log likelihood:",losses[-1])
            plt.plot(losses)
            plt.xlabel("epochs")
            plt.ylabel("negative log likelihood")
            plt.show()
            plt.plot(mses, label='Train MSE')
            plt.plot(mses_test, label='Test MSE')
            plt.xlabel("epochs")
            plt.ylabel("MSE loss")
            plt.legend()
            plt.show()

    def predict(self,X):
        """
        prediction function
        """
        return X @ self.theta

    def criterion(self,M,Y,sigma):
        """
        implementing the custom criterion
        """
        term1 = torch.log(self.theta.T @ sigma @ self.theta)
        term2 = (1/(self.theta.T @ sigma @ self.theta))
        term3 = (M@self.theta - Y).T @ (M@self.theta - Y)
        return term2 @ term3 + term1 
    
    def batch_criterion(self,M,Y,Sigma):
        """
        implementing the custom criterion on a batch of data
        """
        V = torch.cat([
            self.theta.T @ Sigma[i].unsqueeze(0) @ self.theta for i in range(M.size(0))
        ],0)
        
        term1 = torch.log(V)
        term2 = 1/V
        term3 = (M@self.theta - Y).T @ (M@self.theta - Y)
        return (term2 @ term3 + term1).sum() * (1/M.size(0)) 
        