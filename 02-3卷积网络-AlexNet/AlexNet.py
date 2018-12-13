#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata
import os,sys
from time import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=96,kernel_size=11,strides=4,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(channels=256,kernel_size=5,padding=2,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(10))

# X = nd.random.uniform(shape=(1,1,224,224))
# # print(X)
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name,'output shape: ',X.shape)


batch_size = 128
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,resize=224)

print(len(train_iter), len(test_iter))

lr = 0.01
num_epochs = 5
ctx = gb.try_gpu()
net.initialize(ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)


