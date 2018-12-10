#coding:utf-8

import re, codecs
from mxnet import autograd, nd

num_input = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1,shape=(num_examples,num_input))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

from mxnet.gluon import data as gdata

batch_size = 10

dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

from mxnet.gluon import nn

net = nn.Sequential() #建立一个net容器
net.add(nn.Dense(1))  # 自动计算Input点数

from mxnet import init
net.initialize(init.Normal(sigma=0.01)) # initial

from mxnet.gluon import loss as gloss
loss = gloss.L2Loss() # L2 loss

from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

num_epoch = 3
for epoch in range(1, num_epoch+1):

    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print('epoch {0} , loss {1}'.format(epoch,loss(net(features),labels).mean().asnumpy()))

dense = net[0]
print(dense.weight.data())
print(dense.bias.data())  # output the parameters of dense in net