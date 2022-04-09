#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import time
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# In[8]:


# For this assignment, assume that every hidden layer has the same number of neurons.

NUM_HIDDEN_LAYERS = 4
NUM_HIDDEN = 40
NUM_INPUT = 784
NUM_OUTPUT = 10
alpha = 1


def hyperparameters_list():
    hpl = []
    
    n_hidden = [3, 4, 5]
    hidden_units = [30, 40,50]
    batch_size = [20, 50]
    learning_rate = [0.001, 0.0001]
    n_epochs = [100,270]
    alpha = [0.5, 1, 2]
    
    for i in n_hidden:
        for j in hidden_units:
            for n in alpha:
                for l in learning_rate:
                    for k in batch_size:
                        for m in n_epochs:
                            hpl.append([i,j,k, l, m, n])
               
    return(hpl)


# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def relu(a):
    ans = np.where(a<0, 0, a)
    return(ans)

def d_relu(a):
    ans = np.where(a>=0, 1, 0)
    return(ans)
    
def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    zs = []
    hs = []
    zs.append(np.dot(Ws[0], x) + bs[0].reshape(-1,1))
    hs.append(relu(zs[0]))
    
    for layer in range(NUM_HIDDEN_LAYERS-1):
        bs_reshaped = bs[layer+1].reshape(-1,1)
        zs.append(np.dot(Ws[layer+1], hs[layer]) + bs_reshaped)
        hs.append(relu(zs[layer+1]))
#         if layer == len(weights)

    z_soft = np.exp(np.dot(Ws[-1], hs[-1]) + bs[-1].reshape(-1,1))
    z_soft = z_soft.T
    row_sum = np.sum(z_soft, axis=1)
    yhat = np.divide(z_soft.T, row_sum).T
    yhat = yhat.T
    yhat_log = np.log(yhat)
    
    loss = - np.sum(np.multiply(y, yhat_log))/y.shape[1]
    
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)

    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    
    Ws, bs = unpack(weightsAndBiases)
        
    # Initializing g for iteration 0
    g = (yhat-y)/y.shape[1]

    # TODO
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        
        # Adding all predictions for each row
        Jdb = np.sum(g, axis =1)
        dJdbs.append(Jdb)
        
        if (i == 0):
            JdW = np.dot(g,x.T) 
            dJdWs.append(JdW)
            break
        else:
            JdW = np.dot(g,(hs[i-1]).T) 
            dJdWs.append(JdW)
        
        # updating g for other layers
        g = np.multiply(np.dot(Ws[i].T, g),d_relu(zs[i-1]))
        
    # Reversing the list of gradients for forward propogation
    dJdWs = dJdWs[::-1]
    dJdbs = dJdbs[::-1]

    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

def minibatch_create(X,y,bs,i):
    n = X.shape[0]
    X_mini = X[bs*(i):bs*(i+1)]
    y_mini = y[bs*(i):bs*(i+1)]
    return X_mini,y_mini

def train(trainX, trainY, weightsAndBiases, H):
    trajectory = []
    best_loss = np.inf
    n_hidden = H[0]
    hidden_units = H[1]
    batch_size = H[2]
    learning_rate = H[3]
    n_epochs = H[4]
    alpha = H[5]  
    
    wb = initWeightsAndBiases ()
    

    for i in range(n_epochs):
        for j in range(trainX.shape[1]//batch_size):
            X_mini, y_mini = minibatch_create(trainX.T, trainY.T, batch_size, j)
            X_mini, y_mini = X_mini.T, y_mini.T
            
#             grad_wb = back_prop(X_mini, y_mini, wb)
                
#             wb = wb - learning_rate*grad_wb 
            
            w, b = unpack(wb)
                
                
            grad_wb = back_prop(X_mini, y_mini, wb)
            grad_w, grad_b = unpack(grad_wb)

            for arr in range(len(grad_w)):
                d_reg = (alpha/batch_size)*w[arr]

                w[arr] = w[arr] - learning_rate*(grad_w[arr] + d_reg)
                b[arr] = b[arr] - (learning_rate*grad_b[arr])            

            wb= np.hstack([ arr.flatten() for arr in w ] + [ arr.flatten() for arr in b ])
            
        if i > (n_epochs-20):
            fp = forward_prop(X_mini, y_mini, wb)
            print('Training loss for iteration %d'%i,':', fp[0])
            
        trajectory.append(wb)
    fp = forward_prop(trainX, trainY, wb)
    loss = fp[0]
    
    weightsAndBiases = wb
    
    return weightsAndBiases, trajectory


def findBestHyperparameters(X_tr, Y_tr, weightsAndBiases, X_val, Y_val):
    trajectory = []
    
    best_loss = np.inf
    iteration = 0
    tic = time.time()

    hpl = hyperparameters_list()
    for H in hpl:
        global NUM_HIDDEN 
        global NUM_HIDDEN_LAYERS 
        NUM_HIDDEN = H[0]
        NUM_HIDDEN_LAYERS = H[1]
        n_hidden = H[0]
        hidden_units = H[1]
        batch_size = H[2]
        learning_rate = H[3]
        n_epochs = H[4]
        alpha = H[5]  
        
        wb = initWeightsAndBiases()
        
        for i in range(n_epochs):
            for j in range(X_tr.shape[1]//batch_size):
                X_mini, y_mini = minibatch_create(X_tr.T, Y_tr.T, batch_size, j)
                X_mini, y_mini = X_mini.T, y_mini.T
                
#                 grad_wb = back_prop(X_mini, y_mini, wb)
                
#                 wb = wb - learning_rate*grad_wb 
                
                w, b = unpack(wb)
                
                
                grad_wb = back_prop(X_mini, y_mini, wb)
                grad_w, grad_b = unpack(grad_wb)
                
                for arr in range(len(w)):
                    d_reg = (alpha/batch_size)*w[arr]

                    w[arr] = w[arr] - learning_rate*(grad_w[arr] + d_reg)
                    b[arr] = b[arr] - (learning_rate*grad_b[arr])            

                wb= np.hstack([ arr.flatten() for arr in w ] + [ arr.flatten() for arr in b ])
            
        fp = forward_prop(X_val, Y_val, wb)
        loss = fp[0]
        
        if loss < best_loss:
            best_loss = loss
            H_best = H
            weightsAndBiases = wb
        iteration += 1
#         clear_output()
        toc = time.time()

    
    return H_best

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.
    
    #w, b = unpack(trajectory)
    
    
    pca = PCA(n_components = 2)
    wb_tuned = pca.fit_transform(trajectory)
  
    
    def toyFunction (x1, x2):
        wb = pca.inverse_transform([x1,x2])
        fp = forward_prop(trainX, trainY, wb)
        fCE_loss = fp[0]
        return np.sin(fCE_loss)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-3, +3, 0.3)  # Just an example
    axis2 = np.arange(-1, +1, 0.1)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    #Now superimpose a scatter plot showing the weights during SGD.
    
    X_compare = []
    Y_compare = []
    for i in range(len(wb_tuned)):
        X_compare.append(wb_tuned[i][0])
        Y_compare.append(wb_tuned[i][1])
    Z_compare=[]
    for i in range(len(X_compare)):
        Z_compare.append(toyFunction(X_compare[i], Y_compare[i]))
    ax.scatter(X_compare, Y_compare, Z_compare, color='r')
    
    plt.show()


if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    # trainX = ...
    # trainY = ...
    # ...

    trainX = np.load("fashion_mnist_train_images.npy")
    trainY = np.load("fashion_mnist_train_labels.npy")
    testX = np.load("fashion_mnist_test_images.npy")
    testY = np.load("fashion_mnist_test_labels.npy")

    
    # Storing original y_test as y_te for later use
    y_te = testY
    
    # One_hot encoding of y_tr
    n_max = np.max(trainY)+1
    trainY = np.eye(n_max)[trainY]
    
    # One-hot encoding of y_test
    n_max = np.max(testY) +1
    testY = np.eye(n_max)[testY]
    
    
    # Splitting the data into training and validation sets
    X_tr,X_val,Y_tr,Y_val = train_test_split(trainX,trainY,train_size=0.8,random_state=42)
     
    trainX = trainX.T
    trainY = trainY.T
    testX = testX.T
    testY = testY.T
    X_val = X_val.T
    Y_val = Y_val.T
    X_tr = X_tr.T
    Y_tr= Y_tr.T
    
    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()
    # Perform gradient check on 5 training examples
    
    print("\nGradient difference:")
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab)[0],                                    lambda wab: back_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab),                                    weightsAndBiases))
    H_best = findBestHyperparameters(X_tr, Y_tr, weightsAndBiases, X_val, Y_val)
    print("\nBest hyperparameters:", "\nHidden layers:",H_best[0] , "\nNumberof units:", H_best[1], 
          "\nBatch_size:", H_best[2], "\nlearning rate:", H_best[3], "\nNumber of epochs:", H_best[4], 
          "\nAlpha:", H_best[5])
    
    weightsAndBiases = initWeightsAndBiases()
    weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, H_best)
    
    
    fp_test = forward_prop(testX, testY, weightsAndBiases)
    test_loss, yhat = fp_test[0], fp_test[3]
    
    print("\nTest cost :", test_loss)    
    
    yhat = np.argmax(yhat, axis = 0)
    
    n_correct = len(np.where(np.subtract(yhat, y_te)==0)[0])
    n_total = y_te.shape[0]
    
    correct_percent = (n_correct/n_total) *100
    
    print("\nAccuracy:", correct_percent)
    
    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)


