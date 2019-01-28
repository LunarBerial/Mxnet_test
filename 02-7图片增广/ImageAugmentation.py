import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
from time import time


# gb.set_figsize()
# img = image.imread('./img/cat.png')
# print(img)
# gb.plt.imshow(img.asnumpy())
# gb.plt.show() # 重点：不加就无法显示图片

def show_image(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols *scale, num_rows * scale)
    _, axes = gb.plt.subplot(num_rows, num_cols, figsize = figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows = 2, num_cols = 4, scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    # show_image(Y, num_rows, num_cols, scale)
    return Y
shape_aug = gdata.vision.transforms.RandomResizedCrop((200,200), scale=(0.1,1), ratio=(0.5,2))
color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# Y = apply(img, shape_aug)
# print(len(Y))
# print(Y[0])
# gb.plt.imshow(Y[0].asnumpy())
# gb.plt.savefig('cat_01.png')
# gb.plt.show()

augs = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(),
    color_aug,
    shape_aug,
    gdata.vision.transforms.ToTensor()
])
test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor()
])

def load_cifa10(is_train, augs, batch_size):
    return gdata.DataLoader(
        gdata.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size,shuffle=is_train,num_workers=2
    )



def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels,ctx),
            features.shape[0])

def evaluate_accuracy(data_iter, net, ctx = [mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
        acc.wait_to_read()
    return acc.asscalar()/n

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('trianing on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        start = time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat,y) for y_hat,y in zip(y_hats,ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis-1) ==y ).sum().asscalar() for y_har, y in zip(y_hats,ys)])
            train_l_sum += sum([l.sum().asscalar for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc: %.3f, test %1f sec'%(epoch, train_l_sum/n, train_acc_sum/m, test_acc, time()-start))


def train_with_data_aug(train_augs, test_augs, lr = 0.01):
    batch_size = 256
    ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2)]
    net = gb.resnet18(10)
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifa10(True, train_augs, batch_size)
    test_iter = load_cifa10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=8)

train_with_data_aug(train_augs, test_augs)


