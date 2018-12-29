#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata


class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False,strides = 1, **kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)



# blk = BottleNeck(4,use_1x1conv=True, strides=2)
# blk.initialize()
# X = nd.random.uniform(shape = (4,4,6,6))
# print(blk(X).shape)
net = nn.Sequential()
#64 for ResNet; 256 for BottleNeck
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2, padding=1))

def resnet_block(num_channels, num_residuals, first_block = False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels,use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


net.add(resnet_block(64,2, first_block=True),
        resnet_block(128,2),
        resnet_block(256,2),
        resnet_block(512,2),
        nn.GlobalAvgPool2D(),nn.Dense(10))




# X = nd.random.uniform(shape=(1, 64, 56, 56))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'out put shape:', X.shape)

lr = 0.05
num_epochs = 5
batch_size = 256
ctx = mx.gpu(2)
net.initialize(ctx = ctx, init = init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()

train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size, resize=96)
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
