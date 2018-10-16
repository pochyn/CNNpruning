# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# basis of a code taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
# detailed explanation of tensorflow cnn http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/


"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

FLAGS = None

# main function to build cnn
def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  keep_prob = tf.placeholder(tf.float32)
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 784])

  # construction first layer of size 1000
  layer1 = 1000
  with tf.name_scope('layer1'):
      W_conv1 = weight_variable([784, layer1])
      A1 = tf.nn.relu(tf.matmul(x_image, W_conv1))

  # construction first layer of size 1000
  layer2 = 1000
  with tf.name_scope('layer2'):
      W_conv2 = weight_variable([layer1, layer2])
      A2 = tf.nn.relu(tf.matmul(A1, W_conv2))

  # construction first layer of size 500
  layer3 = 500
  with tf.name_scope('layer3'):
      W_conv3 = weight_variable([layer2, layer3])
      A3 = tf.nn.relu(tf.matmul(A2, W_conv3))

  # construction first layer of size 200
  layer4 = 200
  with tf.name_scope('layer4'):
      W_conv4 = weight_variable([layer3, layer4])
      A4 = tf.nn.relu(tf.matmul(A3, W_conv4))

  with tf.name_scope('layer5'):
      W_fc2 = weight_variable([layer4, 10])
      y_conv = tf.matmul(A4, W_fc2)

  return y_conv, keep_prob

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def weight_pruning(sess, sparsity):
  """weight pruning"""
  #get all trained variables
  all_variables = tf.global_variables()
  all_values = sess.run(all_variables)

  for var, val in zip(all_variables, all_values):
      if "layer5" not in var.name:
          matrix = np.array(val)
          if matrix.shape != ():

              #sizes
              x = matrix.shape[0]
              y = matrix.shape[1]
              size = x * y

              #find magnitude of matrix and indexes of SPARCITY% lowest values
              magnitude_matrix = np.absolute(matrix)
              magnitude_matrix = np.reshape(magnitude_matrix, [size])
              number_of_valuest = int((size * sparsity) / 100)
              idx = np.argpartition(magnitude_matrix, number_of_valuest)

              # set lowest to zero and return weight matrix
              matrix = np.reshape(matrix, [size])
              matrix[idx[0:number_of_valuest]] = 0

              initial = tf.reshape(matrix, [x, y])
              tf.assign(var, initial)

def l2_pruning(sess, sparsity):
  """l2 norm prunning"""

  # get all trained variables
  all_variables = tf.global_variables()
  all_values = sess.run(all_variables)

  for var, val in zip(all_variables, all_values):
      if "layer5" not in var.name:
          matrix = np.array(val)
          if matrix.shape != ():

              #sizes
              x = matrix.shape[0]
              y = matrix.shape[1]

              #find indeces of columns with smallest L2-norm value
              i = 0
              all_norms = []
              while i < y:
                  all_norms.append(np.linalg.norm(matrix[:, i], ord=2))
                  i += 1
              number_of_values = int((y * sparsity) / 100)
              idx = np.argpartition(all_norms, number_of_values)

              #set columns with smallest l2-norm to zero
              for index in idx[0:number_of_values]:
                  matrix[:,index] = 0

              initial = tf.reshape(matrix, [x, y])
              tf.assign(var, initial)

# training on mnist dataset when run first time
def train(mnist, accuracy, keep_prob, train_step, x, y_):
    for i in range(20000):
        batch = mnist.train.next_batch(250)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # compute in batches to avoid OOM on GPUs
    accuracy_l = []
    for _ in range(20):
        batch = mnist.test.next_batch(500, shuffle=False)
        accuracy_l.append(accuracy.eval(feed_dict={x: batch[0],
                                                   y_: batch[1],
                                                   keep_prob: 1.0}))
    print('test accuracy %g' % np.mean(accuracy_l))


# create adversarial and reevaluate with model
def eval(mnist, keep_prob, x, y_, accuracy):
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    prob = accuracy.eval(feed_dict={x: eval_data, y_: eval_labels, keep_prob: 1.0})
    return prob

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)
  with tf.name_scope('loss'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv)
      cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
      correct_prediction = tf.cast(correct_prediction, tf.float32)
      accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  prunings = ['weight', 'l2']
  weight_pruning_probs = []
  l2_pruning_probs = []
  sparsities = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

  for pruning in prunings:
      for sparsity in sparsities:
          if pruning == "weight":
              init = tf.global_variables_initializer()
              with tf.Session() as sess:
                  sess.run(init)
                  train(mnist, accuracy, keep_prob, train_step, x, y_)
                  weight_pruning(sess, sparsity)
                  prob = eval(mnist, keep_prob, x, y_, accuracy)
                  weight_pruning_probs.append((sparsity, prob))
          if pruning == "l2":
              init = tf.global_variables_initializer()
              with tf.Session() as sess:
                  sess.run(init)
                  train(mnist, accuracy, keep_prob, train_step, x, y_)
                  l2_pruning(sess, sparsity)
                  prob = eval(mnist, keep_prob, x, y_, accuracy)
                  l2_pruning_probs.append((sparsity, prob))

  print(weight_pruning_probs)
  print(l2_pruning_probs)

  for sprs in weight_pruning_probs:
      plt.scatter(sprs[0], sprs[1])
  plt.xlabel("sparsity")
  plt.ylabel("probability")
  plt.show()

  for sprs in l2_pruning_probs:
      plt.scatter(sprs[0], sprs[1])
  plt.xlabel("sparsity")
  plt.ylabel("probability")
  plt.show()

  print(weight_pruning_probs)
  print(l2_pruning_probs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
