#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss, nn, data as gdata


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentun):
    if not autograd.is_training():
        X_hat = (X - moving_mean)/nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)

        if len(X.shape) == 2:
            mean = X.mean(axis=0)
            var = ((X-mean)**2).mean(axis=0)
        else:
            mean = X.mean(axis=(0,2,3), keepdims = True)
            var = ((X-mean)**2).mean(axis=(0,2,3), keepdims = True)
        X_hat = (X-mean)/nd.sqrt(var+eps)
        moving_mean = momentun * moving_mean + (1.0 - momentun)* mean
        moving_var = momentun * moving_var + (1.0 - momentun) * var
    Y = gamma *X_hat + beta
    return Y, moving_mean, moving_var

class BatchNorm(nn.Block):
    def __init__(self,num_feature, num_dims, **kwargs):
        super(BtachNorm,self).__init__(**kwargs)
        shape = (1,num_feature) if num_dims == 2 else (1, num_feature, 1, 1)
        self.beta = self.params.get('beta', shape=shape, init = init.Zero())
        self.gamma = self.params.get('gamma', shape = shape, init = init.One())
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.ones(shape)

    def forward(self, X):
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var, eps = 1e-5, momentun=0.9)
        return Y

net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    BatchNorm(120, num_dims=2),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    BatchNorm(84,num_dims=2),
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
