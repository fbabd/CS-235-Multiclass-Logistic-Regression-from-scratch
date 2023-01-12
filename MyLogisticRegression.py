import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
from scipy.special import softmax

def calc_loss(X, Y, W, mu, reg):
    """
    X: Input samples
    Y: Labels (onehot encoded)
    W: Weights 
    """
    import warnings
    warnings.filterwarnings('ignore')
    Z =  - X @  W
    N = X.shape[0]
    if reg=='l2':
        reg_term = mu * np.sum(np.square(W)) 
    elif reg=='l1':
        reg_term = mu * np.sum(W)

    loss = 1/N * ((np.trace(X @ W @ Y.T)) + np.sum(np.log(np.sum(np.exp(Z), axis=1)))) + reg_term
    #print(loss)
    return loss 
    
def calc_grad(X, Y, W, mu, reg):
    """
    X: Input samples
    Y: Labels (onehot encoded)
    W: Weights 
    mu: Regularization weight
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    if reg=='l2':
        grad = 1/N * (X.T @ (Y-P)) + 2 * mu * W 
    elif reg=='l1':
        grad = 1/N * (X.T @ (Y-P)) + mu
    return grad 

def gradient_descent(X, Y, regularization, max_iteration, Eta, Mu):
    """ 
    X: Input samples
    Y: Labels (onehot encoded)
    max_iteration: highest number of times gradient descent process is repeated
    Eta: Learning Rate
    Mu: Regularization weight
    """
    y_ohe = ohe.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], y_ohe.shape[1]))
    curr_step = 0
    step_list = []
    loss_list = []
    W_list = []

    while curr_step < max_iteration:
        curr_step += 1
        dW = Eta * calc_grad(X, y_ohe, W, Mu, regularization)
        W = W - dW 
        curr_loss = calc_loss(X, y_ohe, W, Mu, regularization)
        step_list.append(curr_step)
        loss_list.append(curr_loss)
        W_list.append(W)

    history = pd.DataFrame( {'step': step_list, 'loss': loss_list })

    return history, W 



class LR_Multiclass:
    """
    Logistic Regression Class for multiclass.
    regularization     :   loss penalty term. default is 'l2'
    maxiter            :   number of times the weights are updated. default is 1000.
    eta                :   learning rate. default is 0.1
    mu                 :   regularization weight. default is 0.01 
    """
    def __init__(self, regularization='l2', maxiter=1000, eta=0.1, mu=0.01):
        self.regularization = regularization
        self.max_iter = maxiter
        self.eta = eta
        self.mu = mu 
        

    def fit(self, X, Y):
        """
        X   :   Samples to learn
        Y   :   Labels of the samples to learn
        Update the value of self.W and self.history
        """
        self.history, self.W = gradient_descent(X, Y, 
                                                regularization=self.regularization, 
                                                max_iteration = self.max_iter,
                                                Eta = self.eta, 
                                                Mu = self.mu)
    
    def predict(self, Xi):
        """
        Xi  :   Samples to predict class
        returns the class label of the given data samples
        """
        Z = - Xi @ self.W
        P = softmax(Z, axis=1)
        ll = []
        for i in list(range(P.shape[0])):
            cl = np.argmax(P[i:i+1])
            ll.append(cl)
        return np.array(ll)


    def plot_history(self):
        """
        This function plots the loss value over the iterations.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,3))
        self.history.plot(x='step', y='loss', xlabel='iteration', ylabel='loss')
        plt.show()
    
