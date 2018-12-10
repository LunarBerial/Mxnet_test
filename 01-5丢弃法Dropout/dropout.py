#coding:utf-8

import gluonbook as gb
from mxnet import nd,autograd,gluon
from mxnet.gluon import loss as gloss

def dropout(x, drop_prob):
    assert 0<=drop_prob<=1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return x.zeros_like()
    mask = nd.random.uniform(0,1,x.shape) < keep_prob
    return mask * x /keep_prob

num_outputs = 10
num_inputs = 784
num_hiddens1 = 256
num_hiddens2 = 256

w1 = nd.random.normal(scale=0.01, shape=(num_inputs,num_hiddens1))
b1 = nd.zeros(num_hiddens1)
w2 = nd.random.normal(scale=0.01,shape=(num_hiddens1,num_hiddens2))
b2 = nd.zeros(num_hiddens2)
w3 = nd.random.normal(scale=0.01,shape=(num_hiddens2,num_outputs))
b3 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2, w3, b3]
for param in params:
    with autograd.record():
        param.attach_grad()

drop_prob1 = 0.1
drop_prob2 = 0.8

def net(X):
    X = X.reshape((-1,num_inputs))
    H1 = (nd.dot(X,w1)+b1).relu()
    if autograd.is_training():
        H1 = dropout(H1,drop_prob1)
    H2 = (nd.dot(H1,w2)+ b2).relu()
    if autograd.is_training():
        H2 = dropout(H2,drop_prob2)
    return nd.dot(H2,w3) + b3



batch_size = 256
num_epoch = 10

lr = 0.5
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch3(net,train_iter,test_iter,loss,num_epoch,batch_size,params,lr)
