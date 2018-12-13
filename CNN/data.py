#!/usr/bin/env python3

import numpy as np
import random

mnist_test_path = '/home/z/DATA/mnist/test.csv'
mnist_train_path = '/home/z/DATA/mnist/train.csv'



class DATA:
    def __init__(self, train_path, test_path, batch_size=128, n_class=10, valid_ratio=0.1):
        print(' Loading data ...')
        self.test_path = test_path
        self.train_path = train_path
        self.n_class = n_class
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size

        if train_path is not None:
            self.load_train()
        if test_path is not None:
            self.load_test()
    
    def load_test(self):
        self.raw_data_test = np.loadtxt(fname=self.test_path, dtype=float, delimiter=',', skiprows=1)
        self.test = self.normalize(self.raw_data_test).reshape([-1, 28, 28, 1])
    
    def load_train(self):
        self.raw_data_train = np.loadtxt(fname=self.train_path, dtype=float, delimiter=',', skiprows=1)
        self.label_all = self.raw_data_train[:, 0].astype('int')
        self.label_one_hot_all = self.one_hot(self.label_all)
        self.train_all = self.normalize(self.raw_data_train[:, 1:])
        self.train_all_zipped = [[self.train_all[i].reshape([28, 28, 1]), 
            self.label_one_hot_all[i]] for i in range(self.train_all.shape[0])]
        self.valid_len = int(len(self.train_all_zipped) * self.valid_ratio)

    def init(self):
        self.global_index = 0
        random.shuffle(self.train_all_zipped)
        self.train, self.valid = self.split()
    
    def one_hot(self, y):
        return np.eye(self.n_class)[y]
    
    def normalize(self, data):
        return data.astype('float') / 255
    
    def split(self):
        valid = self.train_all_zipped[:self.valid_len]
        train = self.train_all_zipped[self.valid_len:]
        return train, valid
    
    def get_train_size(self):
        return len(self.train)
    
    def now(self):
        return self.loc
    
    def next_batch(self, data):
        loc = 0
        self.loc = 0
        while loc < len(data):
            this_batch = data[loc:min(loc + self.batch_size, len(data))]
            x = np.array([x[0] for x in this_batch])
            y = np.array([x[1] for x in this_batch])
            yield x, y
            loc += self.batch_size
            self.loc = loc
    
    def next_train_batch_iters(self):
        return self.next_batch(self.train)
    def next_valid_batch_iters(self):
        return self.next_batch(self.valid)

if __name__ == '__main__':
    import time
    start = time.time()
    data = DATA(mnist_train_path, mnist_test_path, 128)
    data.init()
    import time
    time.sleep(10)
    sum = 0
    for i in data.next_train_batch_iters():
        sum += 128
    print('------', sum)
    sum = 0
    for i in data.next_train_batch_iters():
        sum += 128
    print(sum, len(data.train), len(data.valid))
    end = time.time()
    print('The time of loading data is', end - start)
