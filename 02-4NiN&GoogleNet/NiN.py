#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata


def nin_block(num_channels, kernal_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernal_size, strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk

net = nn.Sequential()
net.add(
    nin_block(96,kernal_size=11,strides=4, padding=0),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(256, kernal_size=5, strides=1, padding=2),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(384, kernal_size=3, strides=1, padding=1),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Dropout(0.5),
    nin_block(10,kernal_size=3,strides=1, padding=1),
    nn.GlobalAvgPool2D(),
    nn.Flatten()
)

# X = nd.random.uniform(shape=(1, 1, 224, 224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'out put shape:', X.shape)
lr = 0.1
num_epochs = 5
batch_size = 128
ctx = mx.gpu(1)
net.initialize(ctx = ctx, init = init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()

train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size, resize=224)
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
