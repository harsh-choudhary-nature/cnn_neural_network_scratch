# Harsh Choudhary, 2103117

import numpy as np

class NeuralNetwork:

    def __init__(self,layers,neurons_per_layer,activation_per_layer,lr,batch_size,epochs,loss_obj,X:np.ndarray,Y:np.ndarray):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_per_layer = activation_per_layer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_obj = loss_obj
        self.X = X
        self.Y = Y

    def init_layers(self):
        numFeatures = self.X.shape[1]

        # let n denote the number of neurons in the previous layer
        n = numFeatures     # initially preious layer is the input layer
        for i in range(len(self.layers)):
            self.layers[i].init((n,self.neurons_per_layer[i]),(1,self.neurons_per_layer[i]),self.activation_per_layer[i],self.lr)
            n = self.neurons_per_layer[i]

    def forward(self,X:np.ndarray)->np.ndarray:
        input_cur = X 
        for i in range(len(self.layers)):
            input_cur = self.layers[i].forward(input_cur)
        return input_cur
    
    def backward(self,gradLY_pred:np.ndarray)->np.ndarray:
        # returns gradLX though no need
        input_cur_backward = gradLY_pred
        for i in range(len(self.layers)-1,-1,-1):
            input_cur_backward = self.layers[i].backward(input_cur_backward)

        return input_cur_backward
    
    def train(self):

        # mini batch gradient descent
        for epoch in range(self.epochs):
            loss = 0

            for index in range(0,self.X.shape[0],self.batch_size):
                
                X = self.X[index:index+self.batch_size,:]
                Y = self.Y[index:index+self.batch_size,:]

                Y_pred = self.forward(X)
                loss += self.loss_obj.forward(Y_pred,Y)
                gradLY_pred = self.loss_obj.backward(Y)

                self.backward(gradLY_pred)
            
            print(f"epoch {epoch}: Loss = {loss}")

    def predict(self,X):
        return self.forward(X)