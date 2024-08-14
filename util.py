# Harsh Choudhary, 2103117

import numpy as np

class MeanSquaredLossLayer:
    '''
    Useful only for regression problems
    '''
    def __init__(self):
        pass

    def forward(self,Y_pred:np.ndarray,Y:np.ndarray)->float:
        '''
        returns the loss which is a float value
        '''
        self.Y_pred = Y_pred
        numInstances = Y.shape[0]      # for regression problem, it is of shape (N,1)
        return np.sum((Y_pred-Y)**2)/numInstances

    def backward(self,Y:np.ndarray)->np.ndarray:
        '''
        returns gradLY_pred, which is gradLX wrt this layer, and it is of same shape as self.X
        '''
        numInstances = Y.shape[0]      # for regression problem, it is of shape (N,1)
        return 2/numInstances*(self.Y_pred - Y)

class SoftmaxLayer:
    '''
    used for multilabel classification
    '''
    def __init__(self):
        pass

    def forward(self,X:np.ndarray)->np.ndarray:
        self.X = X
        max_in_row = np.max(self.X,axis=1,keepdims=True)
        subtract_max_from_each_along_row = self.X - max_in_row
        exp_matrix = np.exp(subtract_max_from_each_along_row)
        sum_along_rows_exp_matrix = np.sum(exp_matrix,axis=1,keepdims=True)
        self.Y = exp_matrix/sum_along_rows_exp_matrix
        return self.Y.copy()

    def backward(self,gradLY:np.ndarray)->np.ndarray:
        # gradLY of shape (N,m)
        M = np.sum(gradLY * self.Y, axis=1, keepdims=True)   # hadamard product to get the dot product of corresponding rows, M is of shape (N,1)
        M_subtracted = gradLY - M                            # broadcasting
        gradLX = self.Y * M_subtracted                       # hadamard product
        return gradLX

class SigmoidLayer:
    '''
    for binary classification
    '''
    def __init__(self):
        pass

    def forward(self,X:np.ndarray)->np.ndarray:
        self.X = X
        self.Y = 1/(1+np.exp(-self.X))
        return self.Y.copy()

    def backward(self,gradLY:np.ndarray)->np.ndarray:
        one_minus_Y = 1 - self.Y
        return self.Y * one_minus_Y * gradLY          # hadamard product

class CrossEntropyLossLayer:
    '''
    for multilabel classification
    '''
    def __init__(self):
        pass

    def forward(self,Y_pred:np.ndarray,Y:np.ndarray)->float:
        # due to improved softmax implementation, there will be many zeros in Y_pred, i.e., self.X, owing to underflow issues
        # remember that only the term along with correct class persists for each instance but it still offers no guarantee against that term being non zero
        epsilon = 1e-15
        self.Y_pred = Y_pred
        M = np.clip(self.Y_pred,epsilon,1-epsilon)     # values smaller than epsilon set to epsilon, and those larger than 1-epsilon are set to 1-epsion
        return -np.sum(Y * np.log(M))/self.Y_pred.shape[0]             # hadamard product

    def backward(self,Y:np.ndarray):
        # return gradLY_pred, i.e., gradLX wrt this layer
        epsilon = 1e-15
        M = np.clip(self.Y_pred, epsilon, 1 - epsilon)
        return -Y/M/self.Y_pred.shape[0]
    
class BinaryCrossEntropyLossLayer:
    '''
    for binary classification
    '''
    def __init__(self):
        pass

    def forward(self,Y_pred:np.ndarray,Y:np.ndarray)->float:
        # due to sigmoid layer, there can be overflow error, and if sigmoid layer returns 0 due to underflow then loss in this layer will be Nan
        # remember that only the term along with correct class persists for each instance but it still offers no guarantee against that term being non zero
        epsilon = 1e-15
        self.Y_pred = Y_pred
        M = np.clip(self.Y_pred,epsilon,1-epsilon)                  
        one_minus_Y = 1 - Y
        one_minus_Y_pred = 1 - M

        return -np.sum(Y * np.log(M) + one_minus_Y * one_minus_Y_pred)/self.Y_pred.shape[0]      # hadamard product

    def backward(self,Y:np.ndarray):
        epsilon = 1e-15
        M = np.clip(self.Y_pred, epsilon, 1 - epsilon)
        one_minus_Y = 1 - Y
        one_minus_Y_pred = 1 - M
        return -(one_minus_Y/one_minus_Y_pred - Y/self.Y_pred)/self.Y_pred.shape[0]

class TanhLayer:
    
    def __init__(self):
        pass

    def forward(self,X:np.ndarray)->np.ndarray:
        self.X = X
        self.Y = np.tanh(self.X)
        return self.Y.copy()

    def backward(self,gradLY:np.ndarray)->np.ndarray:
        return gradLY * (1-np.tanh(self.X)**2)

class IDLayer:
    def __init__(self):
        pass

    def forward(self,X:np.ndarray)->np.ndarray:
        self.X = X
        self.Y = self.X
        return self.Y.copy()

    def backward(self,gradLY:np.ndarray)->np.ndarray:
        return gradLY
