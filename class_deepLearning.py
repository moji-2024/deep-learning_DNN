import numpy as np
import copy
import pandas as pd
class deepLearner:
    def __init__(self,parameters=None,costs=None):
        self.parameters = parameters
        self.costs = costs

    def fit(self,X,Y,layers_dims=None,learning_rate=0.0075,num_iterations=3000, print_cost=False,activation_output=True ):
        if layers_dims == None:
            layers_dims = [X.shape[0], 5, 1]
        parameters, costs = self.L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost,activation_output)
        self.parameters = parameters
        self.costs = costs
        return parameters, costs
    def initialize_parameters_deep(self,layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        self.parameters = parameters
        return parameters

    def linear_forward(self,A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def sigmoid(self,z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        # s= np.array([(1/(1+np.exp(-i))) for i in z])
        s = 1 / (1 + np.exp(-z))
        return s, z

    def relu(self,Z):
        return np.maximum(0, Z), Z

    def linear_activation_forward(self,A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters, activation_output=True):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
            caches.append(cache)
        if activation_output == True:
            AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
            caches.append(cache)
            return AL, caches
        else:
            ZL, linear_cache = self.linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
            cache = (linear_cache, None)
            caches.append(cache)
            return ZL, caches


    def compute_cost(self,AL, Y, activation_output=True):
        """
        Implement the cost function.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-desire_target, 1 if desire_target), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]
        if activation_output == True:
            cost = (np.sum(-(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)))) / m
            # cost = (np.sum(-(np.dot(np.log(new_y_predicted),Y.T) + np.dot(np.log(1-new_y_predicted),(1-Y).T))))/m
        else:
            cost = np.mean(np.power(AL - Y, 2))
            # cost = (np.sum(np.power(AL-Y,2)))/(m)

        cost = np.squeeze(cost)  # To make sure your cost's shape is float number. For example [[17]] into 17).
        return cost


    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (np.dot(dZ, A_prev.T)) / m
        db = (np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    def relu_backward(self, dA, Z):
      return dA * np.where(Z > 0, 1, 0)
    def sigmoid_backward(self, dA, Z):
      sigmoid_Z = 1 / (1 + np.exp(-Z))
      derivative_sigmoid = sigmoid_Z * (1 - sigmoid_Z)
      return dA * derivative_sigmoid


    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db


    def L_model_backward(self, AL, Y, caches, activation_output=True):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-desire_target, 1 if desire_target)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
        # Initializing the backpropagation

        current_cache = caches[-1]
        if activation_output == True:
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, 'sigmoid')
            grads["dA" + str(L - 1)] = dA_prev_temp
            grads["dW" + str(L)] = dW_temp
            grads["db" + str(L)] = db_temp
        else:
            dZ = 2 * (AL - Y)
            dA_prev_temp, dW_temp, db_temp = self.linear_backward(dZ, current_cache[0])
            # dA_prev_temp, dW_temp, db_temp = linear_backward(AL, current_cache[0])
            grads["dA" + str(L - 1)] = dA_prev_temp
            grads["dW" + str(L)] = dW_temp
            grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]

            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads


    def update_parameters(self, params, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        params -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

        self.parametersb = parameters
        return parameters


    def L_layer_model(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False,
                      activation_output=True):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector , of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        # np.random.seed(1)
        costs = []  # keep track of cost

        parameters = self.initialize_parameters_deep(layers_dims)
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

            AL, caches = self.L_model_forward(X, parameters, activation_output)

            cost = self.compute_cost(AL, Y, activation_output)

            # Backward propagation.

            grads = self.L_model_backward(AL, Y, caches, activation_output)

            # Update parameters.

            parameters = self.update_parameters(parameters, grads, learning_rate)
            # Print the cost every 100 iterations
            if i == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs


    def predict(self, X, parameters=None, output_activation=True):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model
        """
        if parameters == None:
            parameters = self.parameters
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        def threshold(imput):
            if imput >= 0.5:
                return 1
            return 0

        AL, caches = self.L_model_forward(X, parameters, output_activation)
        predictions = np.zeros(AL.shape)
        if output_activation == True:
            for i in range(AL.shape[1]):
                predictions[0, i] = threshold(AL[0, i])
        else:
            predictions = AL
        return predictions
