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
class CustomLayer(K.layers.Layer):
    def __init__(self, h, output_dim, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        # self.input_dim = input_dim
        self.input_dim = h.shape[-1]
        self.output_dim = output_dim
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(self.input_dim, 1),
                                            dtype='float32'),
                                            trainable=True) # MCP penalty
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(),
                                            dtype='float32'),
                                            trainable=True)
        self.w_repeated = tf.repeat(self.w, self.output_dim, axis=-1)
        self.b_repeated = tf.repeat(self.b, self.output_dim)

    def call(self, x):
        h = tf.matmul(x, self.w_repeated) + self.b_repeated # h = xW + b
        h = tf.nn.relu(h) # nonlinear activation
        return h
#%%
output_dim = 5

input_layer = layers.Input((64, 15))
dense1 = layers.Dense(10)
h = dense1(input_layer)

custom_layer = CustomLayer(h, output_dim)
outputs = custom_layer(h)

model = K.models.Model(input_layer, outputs)
model.summary()
#%%
inputs = tf.random.normal((64, 15))
dense1 = layers.Dense(10)
h = dense1(inputs)

custom_layer = CustomLayer(h, output_dim)
outputs = custom_layer(h)
outputs
#%%
import tensorflow as tf
import tensorflow_probability as tfp

# Pretend to load synthetic data set.
n = int(1e+5)
p = 1
# features2 = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
features = tf.random.normal(shape=(n, p), dtype = 'float32')
labels = tf.squeeze(tfp.distributions.Bernoulli(logits=1.618 * features).sample())

# Specify model.
model = tfp.glm.Bernoulli()
# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features,
    response=tf.cast(labels, dtype=tf.float32),
    model=model)

# ==> coeffs is approximately [1.618] (We're golden!)
print(coeffs)
#%%
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf.random.set_seed(1)

# Pretend to load synthetic data set.
n = int(1e+2)
p = 1
# features2 = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
features = tf.random.normal((n, p), dtype = 'float32')
type(features)
labels = tfp.distributions.NegativeBinomial(total_count=10, logits=1.618 * features).sample()
labels = tf.squeeze(labels)

model = tfp.glm.NegativeBinomial(total_count=10)
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix = features,
    response = labels,
    model = model
)
#%%
tf.random.set_seed(2)

# Pretend to load synthetic data set.
n = int(1e+2)
p = 2
# features2 = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
features = tf.random.normal((n, p), dtype = 'float32')
params = tf.cast(np.array([[1.618, 0.125]]), tf.float32)
labels = tfp.distributions.NegativeBinomial(total_count=10, logits=tf.matmul(features, params, transpose_b=True)).sample()
labels = tf.squeeze(labels)

model = tfp.glm.NegativeBinomial(total_count=10)
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix = features,
    response = labels,
    model = model
)
#%%
'''Negative Binomial'''
import numpy as np
import statsmodels.api as sm

np.random.seed(1)
n = int(1e+3)
p = 2

features = np.random.normal(size=(n, p))
beta = np.random.normal(size=(p, 1))
print(beta)

logits = features @ beta
probs = np.exp(logits) / (1 + np.exp(logits))
labels = np.random.negative_binomial(n=10, p=probs, size=(n, 1))

model = sm.GLM(labels, features, family=sm.families.NegativeBinomial())
results = model.fit()
results.summary()
#%%