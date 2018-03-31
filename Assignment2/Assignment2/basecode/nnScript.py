import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0 + np.exp(-z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    #mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    
    #Your code here
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    n_valid = 1000
    validation_data = np.concatenate((mat['train0'][0:n_valid,:],mat['train1'][0:n_valid,:],
                                      mat['train2'][0:n_valid,:],mat['train3'][0:n_valid,:],
                                      mat['train4'][0:n_valid,:],mat['train5'][0:n_valid,:],
                                      mat['train6'][0:n_valid,:],mat['train7'][0:n_valid,:],
                                      mat['train8'][0:n_valid,:],mat['train9'][0:n_valid,:]), 0)
    validation_label = np.concatenate((np.ones((n_valid,1),dtype = 'uint8'),
                                  2*np.ones((n_valid,1),dtype = 'uint8'),
                                  3*np.ones((n_valid,1),dtype = 'uint8'),
                                  4*np.ones((n_valid,1),dtype = 'uint8'),
                                  5*np.ones((n_valid,1),dtype = 'uint8'),
                                  6*np.ones((n_valid,1),dtype = 'uint8'),
                                  7*np.ones((n_valid,1),dtype = 'uint8'),
                                  8*np.ones((n_valid,1),dtype = 'uint8'),
                                  9*np.ones((n_valid,1),dtype = 'uint8'),
                                  10*np.ones((n_valid,1),dtype = 'uint8')),0)
    train_data = np.concatenate((mat['train0'][n_valid:,:],mat['train1'][n_valid:,:],
                                 mat['train2'][n_valid:,:],mat['train3'][n_valid:,:],
                                 mat['train4'][n_valid:,:],mat['train5'][n_valid:,:],
                                 mat['train6'][n_valid:,:],mat['train7'][n_valid:,:],
                                 mat['train8'][n_valid:,:],mat['train9'][n_valid:,:]), 0)
    train_label = np.concatenate((np.ones((mat['train0'][n_valid:,:].shape[0],1),dtype = 'uint8'),
                                  2*np.ones((mat['train1'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  3*np.ones((mat['train2'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  4*np.ones((mat['train3'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  5*np.ones((mat['train4'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  6*np.ones((mat['train5'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  7*np.ones((mat['train6'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  8*np.ones((mat['train7'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  9*np.ones((mat['train8'][n_valid:,:].shape[0],1L),dtype = 'uint8'),
                                  10*np.ones((mat['train9'][n_valid:,:].shape[0],1L),dtype = 'uint8')),0)
    test_label =np.concatenate((np.ones((mat['test0'].shape[0],1L),dtype = 'uint8'),
                                2*np.ones((mat['test1'].shape[0],1L),dtype = 'uint8'),
                                3*np.ones((mat['test2'].shape[0],1L),dtype = 'uint8'),
                                4*np.ones((mat['test3'].shape[0],1L),dtype = 'uint8'),
                                5*np.ones((mat['test4'].shape[0],1L),dtype = 'uint8'),
                                6*np.ones((mat['test5'].shape[0],1L),dtype = 'uint8'),
                                7*np.ones((mat['test6'].shape[0],1L),dtype = 'uint8'),
                                8*np.ones((mat['test7'].shape[0],1L),dtype = 'uint8'),
                                9*np.ones((mat['test8'].shape[0],1L),dtype = 'uint8'),
                                10*np.ones((mat['test9'].shape[0],1L),dtype = 'uint8')),0)
    test_data = np.concatenate((mat['test0'],mat['test1'],
                                mat['test2'],mat['test3'],
                                mat['test4'],mat['test5'],
                                mat['test6'],mat['test7'],
                                mat['test8'],mat['test9']), 0)
    
    
    n_feature = train_data.shape[1]
    sigma = np.std(train_data,0)
    new_train_data = train_data[:,[0]]
    new_validation_data = validation_data[:,[0]]
    new_test_data = test_data[:,[0]]
    for i in range(0,n_feature):
        if (sigma[i] > 0.001):
            new_train_data = np.concatenate((new_train_data,train_data[:, [i]]), 1)
            new_validation_data = np.concatenate((new_validation_data, validation_data[:, [i]]),1)
            new_test_data = np.concatenate((new_test_data, test_data[:, [i]]),1)
    
    train_data = new_train_data.astype(float)/255
    test_data = new_test_data.astype(float)/255
    validation_data = new_validation_data.astype(float)/255 
    
    
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    N, D = training_data.shape
    M = w1.shape[0]
    K = w2.shape[0]
    
    X = np.concatenate((training_data, np.ones((N, 1))),1)
    A = X.dot(w1.T);
    Z = sigmoid(A);
    Z = np.concatenate((Z ,np.ones((N,1))),1);
    B = Z.dot(w2.T);
    Y = sigmoid(B);
    
    for k in range(0,K) :
        t = training_label == (k+1)
        y = Y[:,k]
        y = y[np.newaxis,:]
        obj_val = obj_val + t.T.dot(np.log(y).T) + (1 - t).T.dot(np.log(1 - y).T)
    obj_val = -obj_val
    
    
    #Backpropagation
    grad_w1 = np.zeros((M, D+1))
    grad_w2 = np.zeros((K, M+1))
    for n in range(0, N):
        t = np.zeros((10, 1))
        t[training_label[n]-1] = 1
        y = Y[n, :].T
        y = y[:,np.newaxis]
        err3 = y - t
        grad_w2 = grad_w2 + np.outer(err3,Z[n, :]);
        z = Z[n, :].T
        z = z[:,np.newaxis]
        
        err2 = (w2.T.dot(err3)) * (1 - z) * z
        grad_w1 = grad_w1 + np.outer(err2[0: M],X[n, :]);
    # Adding regularization term
    grad_w1 = grad_w1 + lambdaval * w1;
    grad_w2 = grad_w2 + lambdaval * w2;
    grad_w1 = grad_w1 / N;
    grad_w2 = grad_w2 / N;
    
    t1 = w1*w1;
    t2 = w2*w2;
    obj_val = obj_val + (lambdaval / 2) * (sum(sum(t1)) + sum(sum(t2)));
    obj_val = obj_val / N;
    # Unroll gradients
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    N, D = data.shape
    
    X = np.concatenate((data, np.ones((N, 1))),1)
    A = X.dot(w1.T);
    Z = sigmoid(A);
    Z = np.concatenate((Z ,np.ones((N,1))),1);
    B = Z.dot(w2.T);
    Y = sigmoid(B);
    
    labels = np.argmax(Y,1)
    labels = labels[:, np.newaxis]
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

print train_data.shape

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.



#objv, obgg = nnObjFunction(initialWeights, args)

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

params = nn_params.get('x')
#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print '\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%'

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print '\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%'


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print '\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%'
