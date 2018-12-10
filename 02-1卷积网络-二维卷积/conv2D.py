#coding:utf-8

import gluonbook as gb
from mxnet import nd,autograd,gluon
from mxnet.gluon import loss as gloss, nn

def corr2d(X,weight):
    h,w = weight.shape
    Y = nd.zeros((X.shape[0] -h + 1, X.shape[1] -w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] *weight).sum()
    return Y

class Conv2D(nn.Block):
    def __init__(self, kernal_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape= kernal_size)
        self.bias = self.params.get('bias',shape = (1,))

    def forward(self, x):
        return corr2d(x,self.weight.data()) + self.bias.data()


conv2d = nn.Conv2D(1,kernel_size=(1,2))
conv2d.initialize()

X = nd.ones((6,8))
X[:,2:6] = 0
Y = corr2d(X, nd.array([[1,-1]]))
X = X.reshape((1,1,6,8))
print(Y.shape)
Y = Y.reshape((1,1,6,7))
for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) **2
        if i%2 == 1:
            print('batch %d, loss %.3f'%(i,l.sum().asscalar()))
    l.backward()

    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()


print(conv2d.weight.data())