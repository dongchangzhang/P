#!/usr/bin/env python3

from data import DATA
import tensorflow as tf
import numpy as np

model = '/home/z/Models/digit/mnist-105198.meta'
model_path = '/home/z/Models/digit/'

def predict(test_data, result_name):
    result = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('Image').outputs[0]
        y = graph.get_collection('predict_result')[0]
        for digit in test_data:
            digit = digit.reshape([1, 28, 28, 1])
            r = sess.run(y, feed_dict={x: digit})
            result.append(r)
        
    id = np.array(range(1, len(result) + 1)).reshape(-1, 1)
    result = np.array(result).reshape(-1, 1)
    print(id.shape, result.shape)
    np_result = np.concatenate([id, np.array(result)], axis=1)
    header = 'Imageid,Label'
    print(np_result.shape)
    print(np_result[:10])
    np.savetxt(result_name, np_result, fmt="%d", delimiter=',', header=header)

if __name__ == '__main__':
    result_name = 'result.csv'
    mnist_test_path = '/home/z/DATA/mnist/test.csv'
    data = DATA(None, mnist_test_path).test
    predict(data, result_name)
    