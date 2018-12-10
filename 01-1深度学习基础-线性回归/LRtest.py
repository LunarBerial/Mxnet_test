#coding:utf-8

import re, codecs
from IPython.display import set_matplotlib_formats
from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random



def set_figsize(figsize = (3.5,2.5)):
    set_matplotlib_formats('retina')
    plt.reParams['figure.figsize'] = figsize

# set_figsize()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i+batch_size,num_examples)])
        yield features.take(j),labels.take(j)


def linreg(X,w,b):
    return nd.dot(X,w) + b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) **2 /2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr* param.grad / batch_size # [:] 这种写法是为了不开辟新的内存空间，保持赋值后param地址不变化，外部引用时可以直接拿数据，不用return


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
print(2*features[:,0])
labels = true_w[0] * features[:,0] +true_w[1] * features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape= labels.shape)

print("features, labels")
print(features[0],labels[0])

# plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1)
# plt.show()

w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
b = nd.zeros(shape=(1,))
params = [w, b]
for param in params:
    param.attach_grad()


batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(1,num_epochs + 1):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X,w,b),y)
        l.backward()
        sgd([w,b], lr, batch_size)

    print('epoch {0} , loss {1}'.format(epoch,loss(net(features,w,b),labels).mean().asnumpy()))

print([w,b])

