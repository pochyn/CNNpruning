# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# basis of a code taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
# detailed explanation of tensorflow cnn http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# hide TensorFlow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_dir", "data/", "input directory")
tf.flags.DEFINE_string("output_dir", "runs/", "output directory")
tf.flags.DEFINE_integer("train_steps", 2000, "number of train steps")


def dense_layer(inputs, shape, activation=None, name="dense_layer"):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    w = tf.get_variable(
        name="weight",
        shape=shape,
        dtype=tf.float32,
        initializer=tf.initializers.truncated_normal)
    layer = tf.matmul(inputs, w)
    if activation is not None:
      layer = activation(layer)
    return layer


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
  x_image = tf.reshape(x, [-1, 784])

  layer1 = dense_layer(x_image, [784, 1000], tf.nn.relu, name="dense_layer1")

  layer2 = dense_layer(layer1, [1000, 1000], tf.nn.relu, name="dense_layer2")

  layer3 = dense_layer(layer2, [1000, 500], tf.nn.relu, name="dense_layer3")

  layer4 = dense_layer(layer3, [500, 200], tf.nn.relu, name="dense_layer4")

  layer5 = dense_layer(layer4, [200, 10], None, name="output_layer")

  return layer5


def weight_pruning(sess, sparsity):
  """weight pruning"""
  layers = ['dense_layer1', 'dense_layer2', 'dense_layer3', 'dense_layer4']

  for layer in layers:
    # get trained variable
    with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
      w = tf.get_variable(name="weight")

    shape = tf.shape(w)
    size = tf.reduce_prod(shape)

    flatten = tf.reshape(w, shape=[size])
    magnitude = tf.abs(flatten)
    k = tf.cast(
        tf.cast(size, dtype=tf.float32) * (1 - sparsity / 100), dtype=tf.int32)

    # get top k
    top_k = tf.nn.top_k(magnitude, k=k)

    # get the mask from the bottm_k indices
    mask = tf.sparse_to_dense(
        top_k.indices,
        output_shape=[size],
        sparse_values=1.0,
        default_value=0.0,
        validate_indices=False)
    # set bottom k indexes to 0
    new = tf.reshape(tf.multiply(flatten, mask), shape=shape)

    # assign new matrix to weight
    sess.run(tf.assign(w, new))


def l2_pruning(sess, sparsity):
  """l2 norm prunning"""

  layers = ['dense_layer1', 'dense_layer2', 'dense_layer3', 'dense_layer4']

  for layer in layers:
    # get trained variable
    with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
      w = tf.get_variable(name="weight")

    shape = tf.shape(w)
    size = tf.reduce_prod(shape)

    flatten = tf.reshape(w, shape=[size])

    k = tf.cast(
        tf.cast(shape[1], dtype=tf.float32) * (1 - sparsity / 100),
        dtype=tf.int32)

    norms = tf.norm(w, ord=2, axis=0)
    top_k = tf.nn.top_k(norms, k=k)
    mask = tf.sparse_to_dense(
        top_k.indices,
        output_shape=[tf.cast(shape[1], dtype=tf.int32)],
        sparse_values=1.0,
        default_value=0.0,
        validate_indices=False)

    sparsed = tf.multiply(w, mask)
    sess.run(tf.assign(w, sparsed))


# training on mnist dataset when run first time
def train(mnist, accuracy, train_step, x, y_):
  for i in range(FLAGS.train_steps):
    batch = mnist.train.next_batch(250)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  # compute in batches to avoid OOM on GPUs
  accuracy_l = []
  for _ in range(20):
    batch = mnist.test.next_batch(500, shuffle=False)
    accuracy_l.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
  print('test accuracy %g' % np.mean(accuracy_l))


# create adversarial and reevaluate with model
def eval(mnist, x, y_, accuracy):
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  prob = accuracy.eval(feed_dict={x: eval_data, y_: eval_labels})
  return prob


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.input_dir)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="x")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None], name="y_")

  # Build the graph for the deep net
  y_conv = deepnn(x)

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

  print('Saving graph to: %s' % FLAGS.output_dir)
  train_writer = tf.summary.FileWriter(FLAGS.output_dir)
  train_writer.add_graph(tf.get_default_graph())

  prunings = ['weight', 'l2']
  weight_pruning_probs = []
  l2_pruning_probs = []
  sparsities = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train(mnist, accuracy, train_step, x, y_)
    saved_path = saver.save(sess, os.path.join(FLAGS.output_dir, 'model.ckpt'))
    print("saved model at %s\n" % saved_path)

  for pruning in prunings:
    print("%s pruning" % pruning)
    if pruning == "weight":
      for sparsity in sparsities:
        with tf.Session() as sess:
          saver.restore(sess, saved_path)
          print("restored model from %s" % saved_path)
          weight_pruning(sess, sparsity)
          prob = eval(mnist, x, y_, accuracy)
          print('test accuracy %g sparsity %d\n' % (prob, sparsity))
          weight_pruning_probs.append((sparsity, prob))
    else:
      for sparsity in sparsities:
        with tf.Session() as sess:
          saver.restore(sess, saved_path)
          print("restored model from %s" % saved_path)
          l2_pruning(sess, sparsity)
          prob = eval(mnist, x, y_, accuracy)
          print('test accuracy %g sparsity %d\n' % (prob, sparsity))
          l2_pruning_probs.append((sparsity, prob))

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


if __name__ == '__main__':
  tf.app.run()
