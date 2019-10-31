from keras import utils, Model
from keras.datasets import cifar10
from keras.layers import Input
import numpy as np
import random

import vgg_model
from art.classifiers import KerasClassifier
from art.attacks import TargetedUniversalPerturbation

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
norm2_mean = 0
for im in x_train:
    norm2_mean += np.linalg.norm(im.flatten(), ord=2)
norm2_mean /= len(x_train)

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
    x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

# labels to categorical
num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# trainから10000枚
trainIdx = []
num_each_class = 1000
for class_i in range(10):
    np.random.seed(111)
    idx = np.random.choice( np.where(y_train[:, class_i]==1)[0], num_each_class, replace=False ).tolist()
    trainIdx = trainIdx + idx
random.shuffle(trainIdx)
x_train, y_train = x_train[trainIdx], y_train[trainIdx]

# build resnext model
num_blocks = 3
img_input = Input(shape=(32, 32, 3), name='input')
img_prediction = vgg_model.vgg_model(img_input, num_classes, num_blocks)
model = Model(img_input, img_prediction)
model.load_weights("model/vgg_20_1553198546.h5")

# targeted UAP
classifier = KerasClassifier(model=model)
adv_crafter = TargetedUniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted':True, 'eps':0.006},
    max_iter=5,
    eps=0.15,
    norm=np.inf)

target = 1
y_train_adv_tar = np.zeros(y_train.shape)
for i in range(y_train.shape[0]):
    y_train_adv_tar[i, target] = 1.0

_ = adv_crafter.generate(x_train, y=y_train_adv_tar)

noise = adv_crafter.noise[0,:]
np.save('noise.npy', noise)

rescaled_noise = noise.copy()
for i in range(3):
    rescaled_noise[:, :, i] = rescaled_noise[:, :, i] * channel_std[i]
norm2 = np.linalg.norm(rescaled_noise.flatten(), ord=2)
normInf = np.abs(rescaled_noise.flatten()).max()
print('norm2: {} %'.format(int(norm2/norm2_mean*100)))

x_train_adv = x_train + noise
x_test_adv = x_test + noise

preds_train = np.argmax(classifier.predict(x_train), axis=1)
preds_test = np.argmax(classifier.predict(x_test), axis=1)
acc_train = np.sum(preds_train == np.argmax(y_train, axis=1)) / len(x_train) 
acc_test = np.sum(preds_test == np.argmax(y_test, axis=1)) / len(x_test)
print('acc_train: {} %'.format(int(acc_train*100)))
print('acc_test: {} %'.format(int(acc_test*100)))

preds_train_adv = np.argmax(classifier.predict(x_train_adv), axis=1)
preds_test_adv = np.argmax(classifier.predict(x_test_adv), axis=1) 
acc_train_adv = np.sum(preds_train_adv == np.argmax(y_train, axis=1)) / len(x_train)     
acc_test_adv = np.sum(preds_test_adv == np.argmax(y_test, axis=1)) / len(x_test)
print('acc_train_adv: {} %'.format(int(acc_train_adv*100)))
print('acc_test_adv: {} %'.format(int(acc_test_adv*100)))

targeted_success_rate_train = np.sum(preds_train_adv == target) / len(x_train)
targeted_success_rate_test = np.sum(preds_test_adv == target) / len(x_test) 
print('targeted_success_rate_train: {} %'.format(int(targeted_success_rate_train*100)))
print('targeted_success_rate_test: {} %'.format(int(targeted_success_rate_test*100)))

