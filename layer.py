# Harsh Choudhary, 2103117

import numpy as np
from scipy.signal import correlate
# import cnn_util

class DenseLayer:

    def __init__(self):
        pass

    def init(self,W_shape:tuple,B_shape:tuple,a,lr):
        # Assume previous layer has n neurons and this has m, and as many neurons that many biases
        # a is the activation function object

        self.W = np.random.randn(*W_shape)          # shape (n,m) meaning this layer has m neurons
        self.B = np.random.randn(*B_shape)          # shape (1,m) meaning its a row vector
        self.a = a
        self.lr = lr

    def forward(self,X:np.ndarray)->np.ndarray:
        
        # X is the shape of (N,n)
        Z = (X @ self.W) + self.B  # shape (N,m)
        Y = self.a.forward(Z)      # shape (N,m)
        self.X = X                 # will be used in backward pass
        return Y

    def backward(self,gradLY:np.ndarray)->np.ndarray:

        # gradLY is of shape (N,m)
        gradLZ = self.a.backward(gradLY)
        gradLX = gradLZ @ self.W.T
        gradLW = self.X.T @ gradLZ
        gradLB = np.sum(gradLZ,axis=0,keepdims=True)

        # update
        self.W -= self.lr*gradLW
        self.B -= self.lr*gradLB

        # return 
        return gradLX


class ConvolutionalLayer2D:
    '''
    Valid Convolution for now
    '''
    def __init__(self):
        pass
    
    def init(self,W_shape:tuple,B_shape:tuple,a,lr,mode='valid'):
        # Assume previous layer has n neurons and this has m, and as many neurons that many biases
        # a is the activation function object

        self.W = np.random.randn(*W_shape)          # shape (Co,Ci,k,k) meaning Co kernel of shape (Ci,k,k), so no. of input channels = Ci and no. of output channels is Co
        self.B = np.random.randn(*B_shape)          # shape (Co,1) meaning a seperate bias for each kernel
        self.a = a                                  # activation function that acts on each cell of output
        self.lr = lr                         
        self.mode = mode                            # required to make it default argument so that call to it can be structured with other layers if value to this argument is not passsed       

    def forward(self,X:np.ndarray)->np.ndarray:
        
        if self.mode=='valid':
            
            self.X= X
            B_reshaped = self.B.reshape((*self.B.shape,1))
            Z = []                     # must be of shape (N,Co,_,_)
            for i in range(self.X.shape[0]):
                Z_i = []               # must be of shape (Co,_,_)
                for p in range(self.W.shape[0]):
                    Z_i.append(np.sum(correlate(X[i,:,:,:],self.W[p,:,:,:],mode='valid'),axis=0))
                Z_i = np.array(Z_i) + B_reshaped
                Z.append(Z_i)
                            
            Z = np.array(Z)
            # apply activation function
            Y = self.a.forward(Z)       # confirm that it works element wise, so Y is of shape (N,Co,_,_)
            return Y
        else:
            raise "Method Not Implemented for mode = "+self.mode

    def backward(self,gradLY:np.ndarray)->np.ndarray:

        # gradLY is of shape (N,Co,_,_)
        gradLZ = self.a.backward(gradLY)   # it is also of shape (N,Co,_,_)
        
        if self.mode == 'valid':
            
            # gradLB
            gradLB = np.sum(gradLZ,axis=(0,2,3)).reshape(*self.B.shape)
            
            # gradLW
            gradLW = []             # must be of shape of self.W's shape, i.e., (Co,Ci,k,k)
            for p in range(self.W.shape[0]):
                gradLW_p = []       # must be of shape (Ci,k,k)
                for c in range(self.X.shape[1]):
                    gradLW_p.append(np.sum(correlate(self.X[:,c,:,:],gradLZ[:,p,:,:],mode='valid'),axis=0))
                gradLW_p = np.array(gradLW_p)
                gradLW.append(gradLW_p)
            gradLW = np.array(gradLW)
            
            # gradLX
            gradLX = []                                     # must be of shape (N,Ci,m,n)
            W_rotated = np.rot90(self.W,k=2,axes=(2,3))     # shape (Co,Ci,k,k)
            for i in range(self.X.shape[0]):
                gradLX_i = []        # must be of shape (Ci,m,n)
                for c in range(self.W.shape[1]):
                    gradLX_i.append(np.sum(correlate(gradLZ[i,:,:,:],W_rotated[:,c,:,:],mode='full'),axis=0))
                gradLX_i = np.array(gradLX_i)
                gradLX.append(gradLX_i)
            gradLX = np.array(gradLX_i)

            # update the gradients
            self.W -= self.lr * gradLW
            self.B -= self.lr * gradLB

            # return gradLX
            return gradLX
        else:
            raise "Method Not implemented for mode = "+self.mode


class FlattenLayer:
    '''
    changes shape from (N,C,m,n) to (N,C*m*n)
    '''
    def __init__(self):
        pass

    def init(self,W_shape:tuple,B_shape:tuple,a,lr):
        # this layer typically has no weights or biases, but the function parameter is
        # kept consistent with other layers. Likewise, lr is also not needed
        self.a = a                                    # if need be, we can thus keep an activation function                                  

    def forward(self,X:np.ndarray)->np.ndarray:
        
        # X is of shape (N,C,m,n)
        self.X = X
        Z = self.X.reshape(self.X.shape[0],-1)    # shape (N,C*m*n)
        Y = self.a.forward(Z)           # shape (N,C*m*n)
        return Y

    def backward(self,gradLY:np.ndarray)->np.ndarray:

        # gradLY is of shape (N,C*m*n)
        gradLZ = self.a.backward(gradLY)            # gradLZ is of shape (N,C*m*n)
        gradLX = gradLZ.reshape(*self.X.shape)   # gradLX is of shape (N,C,m,n)

        # return 
        return gradLX

