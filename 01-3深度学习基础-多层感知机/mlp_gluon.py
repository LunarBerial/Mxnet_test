#coding:utf-8

import gluonbook as gb
from  mxnet import gluon,init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'))
# net.add(nn.Dense(1024,activation='sigmoid'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate':0.5})
num_epochs = 50

gb.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)