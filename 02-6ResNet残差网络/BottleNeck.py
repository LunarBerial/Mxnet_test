#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata


class BottleNeck(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False,strides = 1, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels//4, kernel_size=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size= 1, strides=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = nd.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        return nd.relu(Y+X)

# blk = BottleNeck(4,use_1x1conv=True, strides=2)
# blk.initialize()
# X = nd.random.uniform(shape = (4,4,6,6))
# print(blk(X).shape)
net = nn.Sequential()
#64 for ResNet; 256 for BottleNeck
net.add(nn.Conv2D(256, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2, padding=1))

def bottleneck_block(num_channels, num_residuals, first_block = False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(BottleNeck(num_channels, use_1x1conv=True,strides=2))
        else:
            blk.add(BottleNeck(num_channels))
    return blk

net.add(bottleneck_block(256,3, first_block=True),
        bottleneck_block(512,4),
        bottleneck_block(1024,6),
        bottleneck_block(2048,3),
        nn.GlobalAvgPool2D(),nn.Dense(10))

# X = nd.random.uniform(shape=(1, 64, 56, 56))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'out put shape:', X.shape)

lr = 0.05
num_epochs = 15
batch_size = 256
ctx = mx.gpu(1)
net.initialize(ctx = ctx, init = init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()

train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size, resize=96)
gb.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
