import numpy as np

class Pegasus:
    def __init__(self, features, learningRate, epochs):
        self.learningRate = learningRate
        self.epochs = epochs
        self.w = np.zeros(features + 1)

    def __extend_input(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def fit(self, x, y):
        x = self.__extend_input(x)
        
        for epoch in range(1, self.epochs):
            eta = 1/(self.learningRate * epoch)
            fac = (1 - (eta * self.learningRate)) * self.w
            
            for i in range(1, x.shape[0]):  
                prediction = np.dot(x[i], self.w)
                if (y[i] * prediction) < 1 :
                    self.w = fac + eta * y[i] * x[i]            
                else:
                    self.w = fac

    def predict(self, X_test):
        X_test = self.__extend_input(X_test)
        predictions = []
        
        for x in X_test:
            prediction = np.dot(self.w, x)
            prediction = 1 if (prediction > 0) else -1
            predictions.append(prediction)
        return np.array(predictions)