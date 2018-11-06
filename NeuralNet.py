#####################################################################################################################
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from sklearn.impute import SimpleImputer
from numpy import argmax

class NeuralNet:
    def __init__(self, train, activationfunction="sigmoid", header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        self.activationfunction=activationfunction
        raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        self.X,self.y = self.preprocess(raw_input)

        self.X, self.data_test, self.y, self.target_test = train_test_split(self.X, self.y, test_size=.30)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to maices intr network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanhfunction(self, x)
        if activation == "relu":
            self.__relu(self, x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanhfunction_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanhfunction(self, x):
        return np.tanh(x)

    def __relu(self, x):

        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] < 0:
                    x[i][j]=0
        return x


    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanhfunction_derivative(self, x):
        return 1 - (pow(x,2))

    def __relu_derivative(self, x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] < 0:
                    x[i][j]=0
                else:
                    x[i][j]=1
        return x


    def preprocess(self, data):
        ncols = len(data.columns)
        nrows = len(data.index)
        x = data.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        y = data.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        # Class labels are converted in Encoded formate
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        # print(y)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = y.reshape(len(y), 1)
        y = self.onehot_encoder.fit_transform(integer_encoded)
        #print(y)

        """inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
        print(inverted)"""

        x = pd.DataFrame(np.array(x).reshape(nrows, ncols - 1))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x = imp_mean.transform(x)

        x = pd.DataFrame(np.array(x).reshape(nrows, ncols - 1))
        #y = pd.DataFrame(np.array(y).reshape(nrows, len(y[0])))

        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        """
        scaler.fit(y)
        y = scaler.transform(y)
        """

        return x, y

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation=self.activationfunction)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation=self.activationfunction)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("After " + str(max_iterations) + " iterations, the average error is " + str(np.average(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation = "sigmoid"):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            return out
        if activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanhfunction(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanhfunction(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanhfunction(in3)
            return out
        if activation == "relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
            return out




    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanhfunction_derivative(out))
        if activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanhfunction_derivative(self.X23))
        if activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanhfunction_derivative(self.X12))
        if activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanhfunction_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer

    def predict(self, activationfunction = "sigmoid", header = True):
        if activationfunction == "sigmoid":
            in1 = np.dot(self.data_test, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if activationfunction == "tanh":
            in1 = np.dot(self.data_test, self.w01)
            self.X12 = self.__tanhfunction(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanhfunction(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanhfunction(in3)
        if activationfunction == "relu":
            in1 = np.dot(self.data_test, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)

        error = 0.5 * np.power((out - self.target_test), 2)

        length=len(out)
        #inverted = self.label_encoder.inverse_transform([argmax(out[8, :])])
        #print(self.target_test, out, inverted)

        print("Total Error on test data is: ",np.sum(error))
        print("Average Error on Test data is: ",(np.average(error)))
        i=1
        correct=0

        for i in range (length):
            inverted_target = self.label_encoder.inverse_transform([argmax(self.target_test[i - 1, :])])
            inverted_out = self.label_encoder.inverse_transform([argmax(out[i - 1, :])])
            if inverted_out == inverted_target:
                correct += 1

        print("Accuracy on Test Data is ", (correct/length))
        return np.average(error)


if __name__ == "__main__":
    function="sigmoid"
    neural_network = NeuralNet("Iris_Data.txt", function)
    neural_network.train()
    testError = neural_network.predict(function)

