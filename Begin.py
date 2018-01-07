# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Model:
    def __init__(self, input_size, unit_size, output_size, learn_rate, sita1, sita2):
        # overflow 
        self.w1 = np.random.random((input_size + 1, unit_size)) / 200
        self.w2 = np.random.random((unit_size + 1, output_size)) / 200
        self.r = learn_rate
        self.pattern_size = input_size
        self.unit_size = unit_size
        self.output_size = output_size
        self.sita1 = sita1
        self.sita2 = sita2

    def train(self, data, file=None):
        images = data.images
        labels = data.labels
        
        for k in range(len(images)):
            image = images[k]
            # bias
            input_vector = np.append(image, self.sita1)
            label = labels[k]
            
            # calculate E
            r1 = np.dot(input_vector, self.w1)
            self.sigmoid(r1)
            # bias
            r1 = np.append(r1, self.sita2)
            r2 = np.dot(r1, self.w2)
            self.sigmoid(r2)

            delta = np.subtract(r2, label)
            e = np.inner(delta, delta) / 2
            print("--" + str(k) + "--:" + str(e))
            if (file != None):
                file.write("-")
                file.write(str(k))
                file.write("-:")
                file.write(str(e))
                file.write("\n")
            #bp
            de2 = []
            for i in range(self.output_size):
                de2.append(delta[i] * r2[i] * (1 - r2[i]))
            f = r1

            de1 = []
            for i in range(self.unit_size + 1):
                de1.append(np.inner(de2, self.w2[i]) * f[i] * (1 - f[i]))
            # update the w1
            for i in range(self.pattern_size + 1):
                for j in range(self.unit_size):
                    self.w1[i][j] = self.w1[i][j] - input_vector[i] * self.r * de1[j]
            
            # update the w2
            for i in range(self.unit_size + 1):
                for j in range(self.output_size):
                    self.w2[i][j] = self.w2[i][j] - self.r * de2[j] * f[i]
            
            

    def test(self, data, file=None):
        images = data.images
        labels = data.labels
        
        correct = 0
        for k in range(len(images)):
            image = images[k]
            label = labels[k]
            
            r1 = np.dot(np.append(image, self.sita1), self.w1)
            self.sigmoid(r1)
            r1 = np.append(r1, self.sita2)
            r2 = np.dot(r1, self.w2)
            self.sigmoid(r2)
            result = np.argmax(r2)
            if (label[result] == 1):
                correct += 1
        print("correct : " + str(correct / len(data.labels)))
        if (file != None):
            file.write("correct : ")
            file.write(str(correct / len(data.labels)))
            file.write("\n")
            file.write("\n")
        
    def sigmoid(self, units):
        l = len(units)
        for i in range(l):
            units[i] = 1 / (1 + np.math.exp(- units[i]))



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
model = Model(784, 50, 10, 1, 0.0025, 0.0025)

with open("log_t", "w") as f:
    model.train(mnist.train, f)
with open("log_c", "w") as f:
    f.write("train-")
    model.test(mnist.train, f)
    f.write("validation-")
    model.test(mnist.validation, f)
    f.write("test-")
    model.test(mnist.test, f)
