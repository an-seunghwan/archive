#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import re
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/archive')
# os.chdir(r'D:\Semi_Mixture_VAE')

import models
#%%
PARAMS = {
    "batch_size": 128,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "sigma": 1, # variance
    "epochs": 100, 
    "learning_rate": 0.001,
}
#%%
# data
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

labeled = 1000
np.random.seed(520)
# ensure that all classes are balanced 
lidx = np.concatenate([np.random.choice(np.where(y_train == i)[0], int(labeled / PARAMS['class_num']), replace=False) 
                        for i in range(PARAMS['class_num'])])
# lidx = np.random.choice(np.arange(len(x_train)), labeled, replace=False)
uidx = np.array([x for x in np.arange(len(x_train)) if x not in lidx])
x_train_U = x_train[uidx]
y_train_U = y_train_onehot[uidx]
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]

batch_size_U = PARAMS['batch_size']
batch_size_L = 16

train_dataset_U = tf.data.Dataset.from_tensor_slices((x_train_U))
total_length = sum(1 for _ in train_dataset_U)
train_dataset_L = tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L))
#%%
def loss_M2(xhat, x, mean, logvar, PARAMS):
    # reconstruction
    error = - tf.reduce_mean(tf.reduce_sum((x * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.0)) + (1. - x) * tf.math.log(tf.clip_by_value(1. - xhat, 1e-10, 1.0))), axis=-1))
    
    # KL loss by closed form
    kl_loss = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.math.pow(mean, 2) / PARAMS['sigma'] 
                                            - 1 
                                            - tf.math.log(1 / PARAMS['sigma']) 
                                            + tf.math.exp(logvar) / PARAMS['sigma']
                                            - logvar), axis=-1))
    
    return error + kl_loss, error, kl_loss
#%%
model = models.KingmaM2(PARAMS)
learning_rate = PARAMS["learning_rate"]
optimizer = tf.keras.optimizers.Adam(learning_rate)

alpha = 0.1 * len(x_train)

for epoch in range(PARAMS['epochs']):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=len(x_train)).batch(batch_size=PARAMS['batch_size'], drop_remainder=True)
    shuffle_and_batch2 = lambda dataset: dataset.shuffle(buffer_size=labeled).batch(batch_size=32, drop_remainder=True)

    labels = [tf.one_hot([i] * PARAMS['batch_size'], PARAMS['class_num']) for i in range(PARAMS['class_num'])]

    iteratorU = iter(shuffle_and_batch(train_dataset_U))
    iteratorL = iter(shuffle_and_batch2(train_dataset_L))

    iteration = total_length // PARAMS['batch_size'] 
    progress_bar = tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(train_dataset_U))
            imageU = next(iteratorU)
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batch2(train_dataset_L))
            imageL, labelL = next(iteratorL)
            
        with tf.GradientTape(persistent=True) as tape:
            meanL, logvarL, logitsL, zL, xhatL = model(imageL, labelL)
            lossL, reconL, klL = loss_M2(xhatL, imageL, meanL, logvarL, PARAMS)
            
            cls_loss = tf.reduce_sum(- tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(logitsL, 1e-10, 1.0)), axis=-1))
            
            lossU_total = 0
            for i in range(PARAMS['class_num']):
                meanU, logvarU, logitsU, zU, xhatU = model(imageU, labels[i])
                lossU, reconU, klU = loss_M2(xhatU, imageU, meanU, logvarU, PARAMS)    
                lossU_total += tf.reduce_sum(tf.multiply(lossU, tf.reduce_sum(tf.multiply(logitsU, labels[i]), axis=-1))) 
            lossU_total += tf.reduce_sum(logitsU * tf.math.log(tf.clip_by_value(logitsU, 1e-10, 1.0))) 
            
            loss = lossL + lossU_total + alpha * cls_loss

        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        
        loss_avg(loss)
        recon_loss_avg(reconL)
        kl_loss_avg(klL)
        _, _, probL, _, _ = model(imageL, labelL, training=False)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
                'EPOCH': f'{epoch:04d}',
                'Loss': f'{loss_avg.result():.4f}',
                'Recon': f'{recon_loss_avg.result():.4f}',
                'KL': f'{kl_loss_avg.result():.4f}',
                'Accuracy': f'{accuracy.result():.3%}'
        })
    
    _, _, logits, _, _ = model(x_test, y_test_onehot)
    print('EPOCH: {}, classification error: '.format(epoch), len(np.where(y_test - np.argmax(logits.numpy(), axis=-1) != 0)[0]) / len(y_test))
#%%
'''classification error'''
_, _, logits, _, _ = model(x_test, y_test_onehot)
print('classification error: ', len(np.where(y_test - np.argmax(logits.numpy(), axis=-1) != 0)[0]) / len(y_test))
model.save_weights('models/model_M2.h5', save_format="h5")
#%%
# # z grid points
# a = np.arange(-5, 5.1, 1)
# b = np.arange(-5, 5.1, 1)
# aa, bb = np.meshgrid(a, b, sparse=True)
# grid = []
# for b_ in reversed(bb[:, 0]):
#     for a_ in aa[0, :]:
#         grid.append(np.array([a_, b_]))

# for num in range(PARAMS['class_num']):
#     dummy_y = np.zeros((len(grid), PARAMS['class_num']))
#     dummy_y[:, num] = 1
#     grid_output = model.decoder(tf.cast(np.array(grid), tf.float32), dummy_y)
#     grid_output = grid_output.numpy()
#     plt.figure(figsize=(10, 10))
#     for i in range(len(grid)):
#         plt.subplot(len(b), len(a), i+1)
#         plt.imshow(grid_output[i].reshape(28, 28), cmap='gray')    
#         plt.axis('off')
#         plt.tight_layout() 
#     plt.savefig('./result/{}/reconstruction_{}_{}.png'.format(key, num, key), 
#                 dpi=200, bbox_inches="tight", pad_inches=0.1)
#     plt.show()
#     plt.close()
# #%%
# # sampled z
# zmat = []
# mean, logvar, logits, z, xhat = model(x_test, y_test_onehot)
# zmat.extend(z.numpy().reshape(-1, PARAMS['latent_dim']))
# zmat = np.array(zmat)
# plt.figure(figsize=(10, 10))
# plt.rc('xtick', labelsize=10)   
# plt.rc('ytick', labelsize=10)   
# plt.scatter(zmat[:, 0], zmat[:, 1], c=y_test, s=10, cmap=plt.cm.Reds, alpha=1)
# plt.savefig('./result/{}/zsample_{}.png'.format(key, key), 
#             dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
# plt.close()
# #%%
# # learning phase
# plt.rc('xtick', labelsize=10)   
# plt.rc('ytick', labelsize=10)   
# fig, ax = plt.subplots(figsize=(15, 7))
# ax.plot(elbo, color='red', label='ELBO')
# leg = ax.legend(fontsize=15, loc='lower right')
# plt.savefig('./result/{}/learning_phase_{}.png'.format(key, key), 
#             dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
# plt.close()
# #%%