#coding:utf-8

import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train = 20
n_test = 100
num_inputs = 200
true_w = nd.ones((num_inputs,1)) * 0.01
true_b = 0.05

features = nd.random.normal(shape=(n_train+n_test,num_inputs))
labels = nd.dot(features,true_w) + true_b

train_feature, test_feature = features[:n_train,:], features[n_train:,:]
train_labels, test_labels = labels[:n_train],labels[n_train:]

num_epoch = 10
lr = 0.003

batch_size = 1

train_iter = gdata.DataLoader(gdata.ArrayDataset(train_feature,train_labels),batch_size,shuffle=True)
loss = gloss.L2Loss()

def fit_and_polt(weight_decay):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))

    trainer_w = gluon.Trainer(net.collect_params('.*weight'),'sgd',{'learning_rate':lr, 'wd':weight_decay})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'),'sgd',{'learning_rate':lr, 'wd':weight_decay})

    train_ls = []
    test_ls = []
    for _ in range(num_epoch):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()

            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_feature),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_feature),test_labels).mean().asscalar())
    gb.semilogy(range(1, num_epoch + 1), train_ls, 'epoch', 'loss', range(1, num_epoch + 1), test_ls, ['train', 'test'])
    return net[0].weight.data()[:,:10], net[0].bias.data()

print(fit_and_polt(0))