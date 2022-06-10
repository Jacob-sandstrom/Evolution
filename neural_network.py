import random
import numpy as np
import time
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

def max_index(array):
  max_i = 0
  i = 1
  while (i < len(array)):
    if (array[i] > array[max_i]):
      max_i = i
    i += 1
  return max_i

def mean(array):
  total = 0
  for num in array:
    total += num
  mean = total / len(array)
  return mean

class Network:

  def __init__(self, sizes, weights=None, biases=None):
    self.layers = len(sizes)
    self.sizes = sizes
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] if weights is None else weights 
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]] if biases is None else biases
    self.learn_rate = 0.05
    self.gamma = 0.99
    self.mutation_rate = 0.001
    self.bias_change_rate = 0.5
    self.weight_change_rate = 0.5
      
  def feedforward(self, a):
    values = [a]
    all_z = [[None]]
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, a) + b
      all_z.append(z)
      a = sigmoid(z)
      values.append(a)
    return values, all_z

  def backprop(self, all_n_values, z_values, all_w, actions, desiered_val):

    change_w = [np.zeros(w.shape) for w in self.weights]
    change_b = [np.zeros(b.shape) for b in self.biases]
    steps_from_start = 0

    #   network
    for val, z_val, w, a in zip(all_n_values, z_values, all_w, actions):
      l = 1
      change_v = [np.zeros(v.shape) for v in all_n_values[0]]
      # print(a)
      if desiered_val == 1:
        change_v[-1][a][0] = 1
      else:
        for index in range(len(change_v[-1])):
          change_v[-1][index][0] = 1
        change_v[-1][a][0] = 0

      #   layer
      while l < self.layers:

        #   neuron
        for i in range(len(val[-l])):
          #   weights connected to specific neuron
          #   j is a specific weight and also the neuron it is connected to in the previous layer
          change_b[-l][i] -= (sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate
          for j in range(len(w[-l][i])):
            change_w[-l][i][j] -= (val[(-l-1)][j] * sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate
            change_v[-l-1][j] -= (w[-l][i][j] * sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate

        l += 1
      steps_from_start += 1
    #   increase/decrease each weight by calculated value
    for layer in range(len(self.weights)):
      for neuron in range(len(self.weights[layer])):
        self.biases[layer][neuron] += change_b[layer][neuron]
        for weight in range(len(self.weights[layer][neuron])):
          self.weights[layer][neuron][weight] += change_w[layer][neuron][weight]

  def mutate(self):

    for neuron_layer_i, x in enumerate(self.biases):
      for bias_i, y in enumerate(x):
        self.biases[neuron_layer_i][bias_i] = y + (np.random.random_sample() - 0.5) * self.bias_change_rate

    for neuron_layer_i, x in enumerate(self.weights):
      for neuron_i, y in enumerate(x):
        for weight_i, z in enumerate(y):
          self.weights[neuron_layer_i][neuron_i][weight_i] = z + (np.random.random_sample() - 0.5) * self.weight_change_rate





    #     new_biases.each_with_index do |x, index|
    #     x.each_with_index do |y, i|
    #         y.each_with_index do |z, j|
    #             unless index == 0
    #                 new_biases[index][i][j] += (rand * (2) - 1) * change_rate
    #             end
    #         end 
    #     end
    # end
    
    # new_weights.each_with_index do |x, index|
    #     x.each_with_index do |y, i|
    #         y.each_with_index do |z, j|
    #             z.each_with_index do |t, k|
    #                 unless index == 0
    #                     new_weights[index][i][j][k] += (rand * (2) - 1) * change_rate
    #                 end
    #             end
    #         end 
    #     end
    # end               




# n = Network([2, 64, 128, 64, 2])

# print("weights = " + str(n.weights))


# a = [0.2,0.0005]

# for i in range(5000):
#     print("iteration: " + str(i))
#     print("weights = " + str(n.weights))
#     print("biases = " + str(n.biases))
#     n_val, z = n.feedforward(np.reshape(a, (len(a), 1)))
#     print("input = " + str([n_val[0]]))
#     print("output = " + str([n_val[-1]]))
#     print(i)
#     n.backprop([n_val], [z], [n.weights], [0], 1)