#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
'''MCP penalty'''

@tf.function
def MCP(weight, lambda_, r):
    penalty1 = lambda_ * tf.abs(weight) - tf.math.square(weight) / (2. * r)
    penalty2 = tf.math.square(lambda_) * r / 2
    return tf.reduce_sum(penalty1 * tf.cast(tf.abs(weight) <= r * lambda_, tf.float32) + penalty2 * tf.cast(tf.abs(weight) > r * lambda_, tf.float32))
#%%
plt.figure(figsize=(6, 4))
plt.plot([MCP(tf.cast(x, tf.float32), 3., 1.) for x in np.linspace(-5, 5, 1000)], label='MCP')
plt.plot(3. * np.abs(np.linspace(-5, 5, 1000)), label='lasso')
plt.legend()
#%%
'''MCP penalty with NN'''

class MCP(layers.Layer):
    def __init__(self, lambda_, r):
        super(MCP, self).__init__()
        self.lambda_ = lambda_
        self.r = r

    def call(self, weight):
        penalty1 = self.lambda_ * tf.abs(weight) - tf.math.square(weight) / (2. * self.r)
        penalty2 = tf.math.square(self.lambda_) * self.r / 2
        return tf.reduce_sum(penalty1 * tf.cast(tf.abs(weight) <= self.r * self.lambda_, tf.float32) +
                            penalty2 * tf.cast(tf.abs(weight) > self.r * self.lambda_, tf.float32))
#%%
class CustomLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        
        self.output_dim = output_dim
        self.lambda_ = lambda_
        self.r = r
    
    def build(self, input_shape): # 이전 layer의 output을 받지 않아도 됨
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                initializer="random_normal",
                                trainable=True)
        self.b = self.add_weight(shape=(), 
                                initializer="random_normal", 
                                trainable=True)
        
    def call(self, x):
        w_repeated = tf.repeat(self.w, self.output_dim, axis=-1)
        b_repeated = tf.repeat(self.b, self.output_dim)
        h = tf.matmul(x, w_repeated) + b_repeated # h = xW + b
        # h = tf.nn.relu(h) # nonlinear activation
        return h
#%%
p = 30
n = 10000
output_dim = 2
lambda_ = 10.
r = 1.
#%%
beta = np.random.uniform(low=-2., high=2., size=(p, output_dim))
X = np.random.normal(size=(n, p))
y = X @ beta + np.random.normal(size=(n, output_dim))
#%%
# penalty 없는 모형
input_layer = layers.Input((p))
dense1 = layers.Dense(10, activation='linear')
h = dense1(input_layer)
custom_layer = CustomLayer(output_dim)
outputs = custom_layer(h)

model = K.models.Model(input_layer, outputs)
model.summary()
#%%
optimizer = K.optimizers.SGD(0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        yhat = model(X)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y, yhat))
        loss += sum(model.losses)
            
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))

    if i % 10:
        print(i, loss)
        
nopenalty = custom_layer.weights[0]
#%%
# penalty 있는 모형
input_layer = layers.Input((p))
dense1 = layers.Dense(10, activation='linear')
h = dense1(input_layer)
custom_layer = CustomLayer(output_dim)
outputs = custom_layer(h)

model = K.models.Model(input_layer, outputs)
model.summary()

# MCP penalty 추가
mcp = MCP(lambda_, r)
model.add_loss(lambda: mcp(custom_layer.weights[0]))
#%%
optimizer = K.optimizers.SGD(0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        yhat = model(X)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y, yhat))
        loss += sum(model.losses)
            
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))

    if i % 10:
        print(i, loss)
        
withpenalty = custom_layer.weights[0]
#%%
# custom layer weight 값 확인
plt.figure(figsize=(10, 5))
plt.bar(np.arange(10), nopenalty.numpy()[:, 0], label='original')
plt.bar(np.arange(10, 20), withpenalty.numpy()[:, 0], label='MCP')
plt.legend()
#%%