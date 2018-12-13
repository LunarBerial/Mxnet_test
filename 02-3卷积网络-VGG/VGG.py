#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata

def vgg_block(num_convs,numchannels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(numchannels,kernel_size=3,padding=1,activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg(conv_arch):
    net = nn.Sequential()
    for(num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs,num_channels))
    net.add(nn.Dense(4096,activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096,activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

ratio = 4
small_conv_arch = [(pair[0],pair[1] // ratio) for pair in conv_arch]
print(small_conv_arch)
net = vgg(small_conv_arch)
ctx = gb.try_gpu()
net.initialize(ctx = ctx, init=init.Xavier())

lr = 0.05
num_epochs = 15
batch_size = 64
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,resize=224)
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
