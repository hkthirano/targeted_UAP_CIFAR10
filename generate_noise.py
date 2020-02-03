from keras import utils, Model
from keras.datasets import cifar10
from keras.layers import Input
import numpy as np
import random

import vgg_model
from art.classifiers import KerasClassifier
from art.attacks import TargetedUniversalPerturbation
from art.utils import random_sphere

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

# randomly selecting 1000 images for each class in 10 classes 
trainIdx = []
num_each_class = 10
for class_i in range(10):
    np.random.seed(111)
    idx = np.random.choice( np.where(y_train[:, class_i]==1)[0], num_each_class, replace=False ).tolist()
    trainIdx = trainIdx + idx
random.shuffle(trainIdx)
x_train, y_train = x_train[trainIdx], y_train[trainIdx]

# build vgg model
num_blocks = 3
img_input = Input(shape=(32, 32, 3), name='input')
img_prediction = vgg_model.vgg_model(img_input, num_classes, num_blocks)
model = Model(img_input, img_prediction)
model.load_weights("model/vgg_20_1553198546.h5")

# build targeted UAP
classifier = KerasClassifier(model=model)
adv_crafter = TargetedUniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted':True, 'eps':0.006},
    max_iter=10,
    eps=5.5,
    norm=2)

# set target label
target = 1
y_train_adv_tar = np.zeros(y_train.shape)
for i in range(y_train.shape[0]):
    y_train_adv_tar[i, target] = 1.0

# generate noise
_ = adv_crafter.generate(x_train, y=y_train_adv_tar)
noise = adv_crafter.noise[0,:]
norm2_ori = np.linalg.norm(noise.flatten(), ord=2)

# targeted UAP result
print('=== Targeted UAP ===')
rescaled_noise = noise.copy()
for i in range(3):
    rescaled_noise[:, :, i] = rescaled_noise[:, :, i] * channel_std[i]
norm2 = np.linalg.norm(rescaled_noise.flatten(), ord=2)
normInf = np.abs(rescaled_noise.flatten()).max()
print('norm2: {:.1f} %'.format(norm2/norm2_mean*100))

x_train_adv = x_train + noise
x_test_adv = x_test + noise

preds_train_adv = np.argmax(classifier.predict(x_train_adv), axis=1)
preds_test_adv = np.argmax(classifier.predict(x_test_adv), axis=1) 
targeted_success_rate_train = np.sum(preds_train_adv == target) / len(x_train)
targeted_success_rate_test = np.sum(preds_test_adv == target) / len(x_test) 
print('targeted_success_rate_train: {:.1f} %'.format(targeted_success_rate_train*100))
print('targeted_success_rate_test: {:.1f} %'.format(targeted_success_rate_test*100))

np.save('noise.npy', noise)

# random noise result 
print('=== Random Noise ===')
rescaled_noise_rand = random_sphere(nb_points=1,nb_dims=(32*32*3),radius=norm2_ori,norm=2)
rescaled_noise_rand = rescaled_noise_rand.reshape(32,32,3)
noise_rand = rescaled_noise_rand.copy()
for i in range(3):
    noise_rand[:, :, i] = noise_rand[:, :, i] * channel_std[i]
norm2_rand = np.linalg.norm(noise_rand.flatten(), ord=2)
normInf_rand = np.abs(noise_rand.flatten()).max()
print('norm2_rand: {:.1f} %'.format(norm2_rand/norm2_mean*100))

x_train_adv_rand = x_train + noise_rand
x_test_adv_rand = x_test + noise_rand

preds_train_adv_rand = np.argmax(classifier.predict(x_train_adv_rand), axis=1)
preds_test_adv_rand = np.argmax(classifier.predict(x_test_adv_rand), axis=1)
targeted_success_rate_train_rand = np.sum(preds_train_adv_rand == target) / len(x_train)
targeted_success_rate_test_rand = np.sum(preds_test_adv_rand == target) / len(x_test)
print('targeted_success_rate_train_rand: {:.1f} %'.format(targeted_success_rate_train_rand*100))
print('targeted_success_rate_test_rand: {:.1f} %'.format(targeted_success_rate_test_rand*100))

