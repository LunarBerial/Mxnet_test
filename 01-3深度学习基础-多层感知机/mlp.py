#coding:utf-8

import gluonbook as gb
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter,test_iter = gb.load_data_fashion_mnist(batch_size)

num_imput = 784
num_output = 10
num_hidden = 1024

W1 = nd.random.normal(scale=0.01, shape=(num_imput,num_hidden))
b1 = nd.zeros(num_hidden)
W2 = nd.random.normal(scale=0.01, shape=(num_hidden, num_output))
b2 = nd.zeros(num_output)

params = [W1,b1, W2,b2]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X,0)

def net(X):
    X = X.reshape((-1,num_imput))
    H = relu(nd.dot(X,W1)+b1)
    return nd.dot(H,W2) + b2

loss = gloss.SoftmaxCrossEntropyLoss()

num_epoch = 5

lr = 0.5
gb.train_ch3(net,train_iter,test_iter, loss, num_epoch, batch_size,params,lr)