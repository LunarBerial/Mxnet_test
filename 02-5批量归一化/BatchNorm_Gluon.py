#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata


net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)

lr = 1.0
num_epochs = 5
batch_size = 256
ctx = mx.gpu(1)
net.initialize(ctx = ctx, init = init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()

train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size)
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
