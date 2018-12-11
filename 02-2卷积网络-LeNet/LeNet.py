#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn
from time import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))
gb.train_ch3()
# X = nd.random.uniform(shape=(1,1,28,28))
# print(X)
# # net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name,'output shape: ',X.shape)

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size)

print(len(train_iter), len(test_iter))

def evaluate_accuracy(data_iter,net):
    acc = nd.array([0])
    for X,y in data_iter:
        acc += gb.accuracy(net(X),y)
    return acc.asscalar()/ len(data_iter)

def train_ch5(net,train_iter,test_iter,loss,batch_size, trainer,num_epochs):
    for epoch in range(1,num_epochs +1):
        train_l_sum = 0
        train_acc_sum = 0
        start = time()
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_hat,y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %1f sec"%(epoch, train_l_sum/len(train_iter), train_acc_sum/len(train_iter), test_acc, time()-start))


lr = 0.8
num_epochs = 5
net.initialize(init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_ch5(net,train_iter,test_iter,loss,batch_size,trainer,num_epochs)