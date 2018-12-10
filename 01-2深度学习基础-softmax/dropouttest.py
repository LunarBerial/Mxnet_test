#coding:utf-8

import gluonbook as gb
from mxnet import autograd,nd
from mxnet.gluon import loss as gloss

def dropout(X,drop_prob):
    assert 0<= drop_prob <=1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return X.zero_like()
    mask = nd.random.uniform(0,1,X.shape) < keep_prob
    print(mask)
    return mask * X / keep_prob

X = nd.arange(20).reshape((5,4))
print(X)
print(dropout(X,0.5))
