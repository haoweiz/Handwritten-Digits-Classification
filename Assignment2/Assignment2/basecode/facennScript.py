'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

# Do not change this
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



# Replace this with your sigmoid implementation
def sigmoid(z):
# Replace this with your nnObjFunction implementation
    return  1.0 / (1.0 + np.exp(-z))
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    # 716    50        10       (50000,716)    (50000,)        0

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    # w1:(50,717), w2:(10,51)

    # Your code here
    row = training_data.shape[0]         #50000
    column = training_data.shape[1]      #716
    number_w1 = w1.shape[0]              #50
    number_w2 = w2.shape[0]              #10

    # Feedforward
    from_input = np.concatenate((training_data,np.ones(shape = (row,1))),1) #(50000,717)
    out_hidden = sigmoid(w1.dot(np.transpose(from_input)))                  #(50,717) * (717,50000) = (50,50000)
    from_hidden = np.concatenate((out_hidden,np.ones(shape = (1,row))),0)   #(51,50000)
    out = sigmoid(w2.dot(from_hidden))                                      #(10,51) * (51,50000) = (10,50000)

    # BackPropagation
    new_training_label = np.zeros((n_class,row))  # (10,50000), Transform training_label to 0/1 form.
    for i in range(0,row):
        new_training_label[(int)(training_label[i])][i] = 1      # (50000,10), Set the correspond label to 1.
    
    obj_val_front = 0 - ((1/row) * (np.sum(np.multiply(new_training_label,np.log(out)) + np.multiply(np.subtract(1,new_training_label), np.log(np.subtract(1,out))))))
    obj_val_back = np.multiply((lambdaval/(2*row)), (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    obj_val = obj_val_front + obj_val_back
    
    # print (obj_val)

    delta_l = np.subtract(out,new_training_label)       #(10,50000)
    grad_w2 = np.multiply(1/(row), np.dot(delta_l,np.transpose(from_hidden)) + np.multiply(lambdaval,w2))

    a = np.multiply(np.subtract(1,out_hidden), out_hidden)        #(50,50000)
    w2_nobias = np.delete(w2,w2.shape[1]-1, axis = 1)             #(10,50)
    b = np.multiply(np.transpose(a), np.dot(np.transpose(delta_l), w2_nobias))  #(50000,50) * (50000,50)
    c = np.concatenate((training_data,np.ones(shape = (row,1))),axis = 1) #(50000,717)
    grad_w1 = np.multiply(1/(row),np.dot(np.transpose(b), c) + np.multiply(lambdaval, w1))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    # w1:(50,715)  w2:(10,51)  data:(10000,714)
    labels = np.array([])
    #Your code here
    data_bias = np.concatenate((data, np.ones((data.shape[0], 1))),1)    #(10000,715)
    out_hidden = sigmoid(data_bias.dot(np.transpose(w1)));       #(10000,715) * (715,50) = (10000,50)
    out_hidden_bias = np.concatenate((out_hidden ,np.ones((data.shape[0],1))),1);  #(10000,51)
    out = sigmoid(np.dot(out_hidden_bias,np.transpose(w2)));     #(10000,51) * (51,10) = (10000,10)
    
    labels = np.argmax(out,1)
    return labels
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
