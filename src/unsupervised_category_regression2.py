#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import models
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
#%%
'''
1. 다른 그룹에서 \beta_0 + \beta_1 * x에서 \beta_0, \beta_1은 고정
2. 새로운 상수 \alpha를 더하여 \alpha만을 추정
'''
#%%
n = 2000
x = np.random.normal(size=(n, 1))
beta0 = [-1, 2, 7]
beta1 = 3

index = np.random.choice(np.arange(len(beta0)), n, replace=True)
y = np.take(beta0, index)[:, None] + x * beta1 + np.random.normal(size=(n, 1))
#%%
plt.figure(figsize=(7, 7))
plt.scatter(x, y)
plt.show()
plt.close()
#%%
input_layer = layers.Input((1)) 
dense = layers.Dense(1)
output_layer = dense(input_layer)

model = models.Model(input_layer, output_layer)
model.summary()
#%%
optimizer = K.optimizers.SGD(0.1)

for i in range(1000):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = tf.reduce_mean(tf.abs(y - pred))
        
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    print(i, loss)

'''바꿔야 하는 값'''
# fitted = model.weights[1][0].numpy() + model.weights[0][0][0].numpy() * x
fitted = model(x) 
#%%
plt.figure(figsize=(10, 10))
plt.scatter(x, y)
plt.plot(x, fitted, linewidth=2, color='orange')
plt.show()
plt.close()
#%%
indicator = np.array(list(map(lambda i: 1 if y[i] >= fitted[i] else 0, np.arange(n))))
plt.figure(figsize=(10, 10))
plt.scatter(x[np.where(indicator == 1)], y[np.where(indicator == 1)])
plt.plot(x, fitted, linewidth=2, color='orange')
plt.show()
plt.close()
#%%
x_A = x[np.where(indicator == 1)]
y_A = y[np.where(indicator == 1)]

residual_A = y_A - fitted.numpy()[np.where(indicator == 1)]

optimizer = K.optimizers.SGD(0.1)

alpha_A = tf.Variable(1, trainable=True, dtype=tf.float32)

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(alpha_A)
        loss = tf.reduce_mean(tf.square(residual_A - alpha_A))
        
    grad = tape.gradient(loss, [alpha_A])
    optimizer.apply_gradients(zip(grad, [alpha_A]))
    
    print(i, loss)

fitted_A = fitted.numpy()[np.where(indicator == 1)] + alpha_A
#%%
plt.figure(figsize=(10, 10))
plt.scatter(x, y)
plt.plot(x, fitted, linewidth=2, color='orange')
plt.plot(x_A, fitted_A, linewidth=2, color='orange')
plt.show()
plt.close()
#%%
x_B = x[np.where(indicator == 0)]
y_B = y[np.where(indicator == 0)]

residual_B = y_B - fitted.numpy()[np.where(indicator == 0)]

optimizer = K.optimizers.SGD(0.1)

alpha_B = tf.Variable(1, trainable=True, dtype=tf.float32)

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(alpha_B)
        loss = tf.reduce_mean(tf.square(residual_B - alpha_B))
        
    grad = tape.gradient(loss, [alpha_B])
    optimizer.apply_gradients(zip(grad, [alpha_B]))
    
    print(i, loss)

fitted_B = fitted.numpy()[np.where(indicator == 0)] + alpha_B
#%%
plt.figure(figsize=(10, 10))
plt.scatter(x, y)
plt.plot(x, fitted, linewidth=2, color='orange')
plt.plot(x_A, fitted_A, linewidth=2, color='orange')
plt.plot(x_B, fitted_B, linewidth=2, color='orange')
plt.show()
plt.close()
#%%
from icecream import ic
ic(alpha_B, alpha_A)
ic(alpha_B + model.weights[1][0].numpy(), alpha_A + model.weights[1][0].numpy())
ic(beta0)
#%%