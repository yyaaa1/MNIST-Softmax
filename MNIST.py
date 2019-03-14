import numpy as np
import random 
import os
#import the data#
trainimages = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_train_images.npy'))
trainlabels = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_train_labels.npy'))
validationimages = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_validation_images.npy'))
validationlabels = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_validation_labels.npy'))
testimages = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_test_images.npy'))
testlabels = np.asmatrix(np.load('/Users/york/Desktop/MNIST/mnist_test_labels.npy'))
def Softmax(X,W):
     e_matrix = np.exp(X.dot(W))
     e_boolean = np.isin(e_matrix, float('inf'))
     f_where=np.where(e_boolean == True)
     for i in f_where[0]:
          e_matrix[[i],:] = e_boolean[[i],:]
     
     sum_row= np.sum(e_matrix,axis=1)
     prediction = e_matrix/sum_row

     return prediction
     
    
def gradientdescent(W,Y,X,alpha,learningrate):
    n = X.shape[0]
    W_gradient = np.zeros((784,10))
    p = Softmax(X,W)
    for i in range(10):
        difference = Y[:,[i]]-p[:,[i]]
        gradient = (-1/(2*n))*np.sum((np.multiply(difference,X)),axis = 0).T+alpha*W[:,[i]]
        W_gradient[:,[i]] = gradient
    W=W-learningrate*W_gradient


    return W,W_gradient


def SGD(learningrate,Y,X,decayrate,alpha,K,epoch,batch):
    W = np.zeros((784,10))
    list_1 = [i for i in range(batch)]
    random.shuffle(list_1)
    n = int(X.shape[0]/batch)
    a=K
    for i in range(epoch):
        for j in list_1:
            if a%batch == 0 and a>=batch:
                 learningrate=decayrate*learningrate
            e=(j*n+1)
            f= ((j+1)*n)
            X_1=X[e:f,:]
            Y_1=Y[e:f,:]
            W,W_gradient= gradientdescent(W,Y_1,X_1,alpha,learningrate)
            a +=1
    return W
            
        
    
def cost_function(Y,W,X,alpha):
     prediction = Softmax(X,W)
     np.putmask(prediction, prediction<1E-323, 1E-323)
     n = Y.shape[0]
     cost =((-1/(2*n))*np.sum(np.multiply(Y,np.log(prediction))))+(alpha/2)*np.sum(np.square(W))
     return cost
     
def accuracy(Y,X,W):
     prediction = Softmax(X,W)
     prediction_tag = np.asmatrix((np.isin((prediction - np.amax(prediction,axis=1)),0)).astype(int))
     accuracy = 100-(np.sum(np.amax((Y-prediction_tag),axis=1))/(Y.shape[0]))*100
     
     return accuracy


def tuning_hp(learningrate,decayrate,alpha,K,epoch,batch):
     acc = 0
     hp_dict = {}
     for i in learningrate:
          for j in decayrate:
               for a in alpha:
                    for b in K:
                         for c in epoch:
                              for d in batch:
                                   W = SGD(i,trainlabels,trainimages,j,a,b,c,d)
                                   accuracy1 = accuracy(validationlabels,validationimages,W)
                                   if accuracy1 >acc:
                                        acc = accuracy1
                                        tuple_1 = i,j,a,b,c,d 
                              hp_dict[tuple_1] = acc
                              print(acc,tuple_1)
     a = max(np_dict)
     return a
                                        



learningrate = [i/10 for i in range(1,10,1)]
decayrate = [i/10 for i in range(1,10,2)]
alpha = [i/10000 for i in range(1,10,1)]
K = [2,4,6,8,10,15]
epoch = [5,10,15,20]
batch = [20,25,50]
a = tuning_hp(learningrate,decayrate,alpha,K,epoch,batch)




     

learningrate= 1
decayrate =0.99
alpha = 0.00012
K = 15
epoch = 20
batch = 25


W = SGD(learningrate,trainlabels,trainimages,decayrate,alpha,K,epoch,batch)
cost = cost_function(testlabels,W,testimages,alpha)
accuracy = accuracy(testlabels,testimages,W)


print(cost,accuracy)
                                                                     
    
    
    
