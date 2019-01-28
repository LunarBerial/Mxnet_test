#coding:utf-8
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
from time import time

scale = 0.01
W1 = nd.random.normal(scale=scale, shape=(20,1,3,3))
B1 = nd.zeros(shape=20)
W2 = nd.random.normal(scale=scale, shape=(50,20,5,5))
B2 = nd.zeros(shape=50)
W3 = nd.random.normal(scale=scale, shape=(800, 128))
B3 = nd.zeros(shape=128)
W4 = nd.random.normal(scale=scale, shape=(128,10))
B4 = nd.zeros(shape=10)
params = [W1, B1, W2, B2, W3, B3, W4, B4 ]

def lenet(X, params):
    h1_conv = nd.Convolution(data = X, weight=params[0], bias=params[1], kernel=(3,3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='avg', kernel=(2,2), stride=(2,2))
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3], kernel=(5,5), num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='avg', kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat

loss = gloss.SoftmaxCrossEntropyLoss()

def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])

def spilt_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n //k
    assert m*k == n, '#example is not divided by #devices'
    return [data[i*m:(i+1)*m].as_in_context(ctx[i]) for i in range(k)]

def train_batch(X, y, gpu_params, ctx, lr):
    gpu_Xs = spilt_and_load(X, ctx)
    gpu_ys = spilt_and_load(y, ctx)

    with autograd.record():
        ls = [loss(len(gpu_X, gpu_W), gpu_y) for gpu_X, gpu_W, gpu_y in zip(gpu_Xs, gpu_params, gpu_ys)]

    for l in ls:
        l.backward()

    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx))])

    for param in gpu_params:
        gb.sgd(param, lr, X.shape[0])


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print("training on:", ctx)
    gpu_params = [get_params(params,c) for c in ctx]

    for epoch in range(5):
        start = time()
        for X, y in train_iter:
            train_batch(X, y, gpu_params, ctx, lr)
        nd.waitall()
        print('epoch %d, time: %.1f sec' % (epoch, time()-start))
        net = lambda x: lenet(x,gpu_params[0])
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('validation accuracy: %.4f'%test_acc)

# 在单GPU上训练
train(num_gpus=1, batch_size=256, lr=0.3)

#在多GPU上训练
# train(num_gpus=2,batch_size=512, lr = 0.3)