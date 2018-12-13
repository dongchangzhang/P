#!/usr/bin/env python

import numpy as np
import tensorflow as tf

class CNN:
    def __init__(self, n_row, n_col, n_dep, n_output, learning_rate):
        self.n_row = n_row
        self.n_col = n_col
        self.n_dep = n_dep
        self.n_output = n_output
        self.learning_rate = learning_rate

        self.global_step = tf.get_variable(name='GlobalStep', dtype=tf.int32, trainable=False, initializer=tf.constant(0))
        self.keep_prob = 0.75

        self._input_data()
        self._build_cnn()
        self._loss()
        self._optimizer()
        self._eval()

    def _input_data(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_row, self.n_col, self.n_dep], name='Image')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_output], name='Result')

    def _conv_relu(self, scope_name, x, kernel_x, kernel_y, n_channels, n_filters, strides, padding='SAME'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            conv_filter = tf.get_variable('filter', shape=[kernel_x, kernel_y, n_channels, n_filters], initializer=tf.truncated_normal_initializer())
            conv_biases = tf.get_variable('biases', shape=[n_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input=x, filter=conv_filter, strides=strides, padding=padding)
        return tf.nn.relu(conv + conv_biases, name=scope.name)
    
    def _max_pool(self, scope_name, x, k, padding='VALID'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            pool = tf.nn.max_pool(value=x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)
        return pool
    
    def _fully_connected(self, scope_name, x, n_input, n_output):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            weight = tf.get_variable('weight', shape=[n_input, n_output], initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', shape=[n_output], initializer=tf.constant_initializer(0.0))
            out = tf.add(tf.matmul(x, weight), biases)
        return out
    
    def _build_cnn(self):
        k = 2
        kernel_x = 3
        kernel_y = 3

        padding_conv = 'SAME'
        strides_conv = [1, 1, 1, 1]

        n_filters_conv1 = 32
        n_channels_conv1 = 1
        conv1 = self._conv_relu('conv1', self.x, kernel_x, kernel_y, n_channels_conv1, n_filters_conv1, strides_conv, padding_conv)
        pool1 = self._max_pool('pool1', conv1, k)

        n_filters_conv2 = 64
        n_channels_conv2 = n_filters_conv1
        conv2 = self._conv_relu('conv2', pool1, kernel_x, kernel_y, n_channels_conv2, n_filters_conv2, strides_conv, padding_conv)
        pool2 = self._max_pool('pool2', conv2, k)

        fc1_input_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        fc1_input = tf.reshape(pool2, [-1, fc1_input_dim])
        fc1_output_dim =  64

        fc1 = self._fully_connected('fc_1', fc1_input, fc1_input_dim, fc1_output_dim)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        
        self.logits = self._fully_connected('fc_2', fc1, fc1_output_dim, self.n_output)

    
    def _loss(self):
        labels = tf.stop_gradient(self.y)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits)
        self.loss = tf.reduce_mean(entropy, name='loss')
    
    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    
    def _eval(self):
        pred = tf.nn.softmax(self.logits)
        predict_result = tf.argmax(pred, 1)
        tf.add_to_collection('predict_result', predict_result)
        correct_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if __name__ == '__main__':
    cnn = CNN(28, 28, 1, 10, 0.001)

