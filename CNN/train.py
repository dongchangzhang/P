#!/usr/bin/env/ python3

from cnn import CNN
from data import DATA

import tensorflow as tf
import progressbar 


def train(model, data, epoches):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_accuracy = 0
        for epoch in range(epoches):
            data.init()
            train_iter = data.next_train_batch_iters()
            valid_iter = data.next_valid_batch_iters()
            step1 = 0
            loss1 = 0
            accuracy1 = 0
            bar = progressbar.ProgressBar(maxval=data.get_train_size()).start()
            for x, y in train_iter:
                bar.update(data.now())
                step1 += 1
                loss, accuracy, _ = sess.run([model.loss, model.accuracy, model.optimizer], feed_dict={model.x: x, model.y: y})
                loss1 += loss
                accuracy1 += accuracy
            print(' Epoch', epoch, 'Loss', loss1 / step1, accuracy1 / step1)
            # validation
            step2 = 0
            loss2 = 0
            accuracy2 = 0
            for x, y in valid_iter:
                step2 += 1
                loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict={model.x: x, model.y: y})
                loss2 += loss
                accuracy2 += accuracy
            print(' Valid', epoch, 'Loss', loss2 / step2, accuracy2 / step2)
            if epoch > 10 and accuracy2 > last_accuracy:
                saver.save(sess, '/home/z/Models/digit/mnist', global_step=model.global_step.eval())
                print(' - Model saved - ')
                last_accuracy = accuracy2


if __name__ == '__main__':
    n_row = 28
    n_col = 28
    n_dep = 1
    n_out = 10
    epoches = 100
    batch_size = 32
    learning_rate = 0.001

    mnist_train_path = '/home/z/DATA/mnist/train.csv'

    model = CNN(n_row, n_col, n_dep, n_out, learning_rate)
    data = DATA(mnist_train_path, None, batch_size, n_out)
    train(model, data, epoches)