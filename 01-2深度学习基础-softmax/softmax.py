#coding:utf-8
from mxnet.gluon import data as gdata
from mxnet import nd, autograd
import gluonbook as gb
import sys


def transform(data,label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

features, labels = mnist_train[0]

def show_fishon_imgs(images):
    _, figs = gb.plt.subplots(1,len(images), figsize= (15,15))
    for f,img in zip(figs,images):
        f.imshow(img.reshape((28,28)).asnumpy())
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

    gb.plt.show()

def get_text_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

    return [text_labels[int(i)] for i in labels]

X,y =mnist_train[0:9]
# show_fishon_imgs(X)
print(get_text_labels(y))

batch_size = 256
transformer = gdata.vision.transforms.ToTensor() # ToTensor函数此处的作用是将图片数据从uint8 转换成float32 的数据，并除以255 使之归一化。
num_workers = 0 if sys.platform.startswith('win32') else 4
print(num_workers)
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size,
                              shuffle=True,
                              num_workers=num_workers)
# 同上一节中的data_iter是一个东西,可多进程读取数据。 num_worker为进程数 transform_first 使得transformer的转换作用在data的第一个元素（即（pic，label）中的pic）上

test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size,
                             shuffle=True,
                             num_workers=num_workers)

num_inputs = 784
num_output = 10

w = nd.random.normal(scale=0.01, shape=(num_inputs,num_output))
b = nd.zeros(num_output)
params = [w,b]
for param in params:
    param.attach_grad()

#define sofmax
def softmax(X):
    exp = X.exp()
    partition = exp.sum(axis=1, keepdims = True)
    return exp/partition # 广播

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),w) + b)

# 交叉熵
#注： softmax中有取exp的操作，交叉熵中将这种操作还原取log。实验证明这种分开运算的方式对结果的准确率有影响。所以再后续的gluon版本中，将这两步的计算合二为一。称为 gluon.SoftmaxCrossEntropyLoss()

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat.log(),y)

def accuracy(y_hat, y):
    return (y_hat.argmax(axis =1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for X,y in data_iter:
        acc += accuracy(net(X), y)

    return acc/ len(data_iter)

num_epochs = 5
lr = 0.1

loss = cross_entropy

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params = None, lr = None, trainer=None):
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0

        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat,y)

        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch {0}, loss {1}, train accuracy {2}, test accuracy {3}'.format(epoch, train_l_sum/len(train_iter), train_acc_sum/len(train_iter),test_acc))


train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


data, label = mnist_test[0:9]
show_fishon_imgs(data)
print('labels:', get_text_labels(label))

predicts_label = [net(transformer(x)).argmax(axis = 1).asscalar()
                  for x in data]

print('predict_label:', get_text_labels(predicts_label))