# A bit of setup
import numpy as np
import matplotlib.pyplot as plt

XAll = np.random.rand(100, 1) * 20 - 10
yAll = np.abs(XAll)   #** 2 #+ 0.1*np.random.randn(100, 1) # noise added
train_ratio = 0.85
num_train_datum = int(train_ratio * 100)
X = XAll[0:num_train_datum]  # training data
X_test = XAll[num_train_datum:]  # testing data
y = yAll[0:num_train_datum]
y_test = yAll[num_train_datum:]

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 1
    self.outputSize = 1
    self.hiddenSize = 300
  #weights
    self.W1 = np.random.randn(self.inputSize,  self.hiddenSize)     # weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize)     # weight matrix from hidden to output layer

    self.b1 = np.zeros((1, self.hiddenSize))
    self.b2 = np.zeros((1, self.outputSize))
    self.z  = None
    self.z2 = None
    self.z3 = None
    self.o  = None

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return self.sigmoid(s) * (1 - self.sigmoid(s))

  def cost(self, X, y):
      output = self.forward(X)
      cost = np.sum((output - y) ** 2) / 2.0
      return cost

  def costGradient(self, output, y):
      return output - y

  def dCdW(self, layer_input, layer_error):
      # Derivative of Cost with respect to any weight in the network
      return np.dot(layer_input.T, layer_error)

  def dCdB(self, curLayerError):
      # Derivative of Cost with respect to the Bias of any neuron in the network

      return np.sum(curLayerError, axis=0, keepdims=True)


  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) + self.b1           # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z)                  # activation function
    self.z3 = np.dot(self.z2, self.W2) + self.b2    # dot product of hidden layer (z2) and second set of 3x1 weights
    self.o = self.z3                                # final activation function
    return self.o

  def backward(self, X, y, output):
    # backward propagate through the network
    self.o_error  = self.costGradient(output, y) * self.sigmoidPrime(self.z3)
    self.z2_error = np.dot(self.o_error, self.W2.T) * self.sigmoidPrime(self.z)

    # Weight Gradient
    self.z2_delta = self.dCdW(X, self.z2_error)
    self.o_delta  = self.dCdW(self.z2,self.o_error)

    # Bias Gradients

    self.db1 = self.dCdB(self.z2_error)
    self.db2 = self.dCdB(self.o_error)

    return self.z2_delta, self.o_delta, self.db1, self.db2

  def update_weights(self, X, y, lr):
      output = self.forward(X)
      self.z2_delta, self.o_delta, self.db1, self.db2 = self.backward(X, y, output)

      # Update Weights
      self.W1 -= lr*self.z2_delta
      self.W2 -= lr*self.o_delta
      self.b1 -= lr*self.db1
      self.b2 -= lr*self.db2


  def train(self, X, y):
      self.output = self.update_weights(X, y, lr=0.001)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled): \n" + str(X_test));
    print("Output: \n" + str(self.forward(X_test)))

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  print ("Input (scaled): \n" + str(X))
  print ("Actual Output: \n" + str(y))
  print ("Predicted Output: \n" + str(NN.forward(X)))
  print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))  # mean sum squared loss
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()
output = NN.forward(X)
plt.plot(X, y, "ro")
plt.plot(X,output, "bo")
plt.show()