#coding:utf-8

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
from time import time

def resnet18(num_classes):
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    def resnet_block(num_channels, num_residuals, first_block = False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(gb.Residual(num_channels, use_1x1conv=False, strides=2))
            else:
                blk.add(gb.Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(),nn.Dense(num_classes))
    return net



loss = gloss.SoftmaxCrossEntropyLoss()
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('training on ', ctx)
    net = resnet18(10)
    net.initialize(init=init.Normal(sigma=0.01), ctx = ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(5):
        start = time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y) for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]

            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        print('epoch %d, training time: %.1f sec' %
              (epoch, time() - start))
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('validation accuracy %.4f' % (test_acc))

train(num_gpus=2, batch_size=512, lr = 0.3)

