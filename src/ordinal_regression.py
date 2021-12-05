#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data = pd.read_csv('/Users/anseunghwan/Documents/uos/bostonhousing_ord.csv')
X = np.array(data.iloc[:, 1:])       
y = np.array(data.iloc[:,0]) -1

X = (X - tf.math.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0) # scaling
K = len(tf.unique(y)[0])                 # number of category
y_true = y
y = tf.keras.utils.to_categorical(y)  # one-hot encoding
y = tf.cast(y, tf.float32)
X = tf.cast(X, tf.float32)
n, p = X.shape
#%%
class BuildModel(tf.keras.models.Model): # 부모 class
    def __init__(self, n, p, K): # initial
        super(BuildModel, self).__init__() # 상속이 이루어지는 부분
        
        self.n = n
        self.p = p
        self.K = K
        
        self.alpha = tf.Variable(tf.random.normal([1, self.K-1], 0, 1), trainable=True)
        self.beta = tf.Variable(tf.random.normal([self.p, 1], 0, 1), trainable=True)
        
    def call(self, X):
        '''reparametrization for ordered condition'''
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]
        
        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat1 = tf.concat((mat1, tf.ones((self.n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat2 = tf.concat((tf.zeros((self.n, 1)), mat2), axis=-1)
        
        return mat1 - mat2
    
    def predict(self, x):
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]
        
        return tf.nn.sigmoid(theta + tf.matmul(x, self.beta))

    def accuracy(self, X, y_true):
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]

        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat1 = tf.concat((mat1, tf.ones((self.n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat2 = tf.concat((tf.zeros((self.n, 1)), mat2), axis=-1)

        y_pred = tf.argmax(mat1 - mat2, axis=1).numpy()

        table = pd.crosstab(y_true, y_pred,rownames=['True'], colnames=['Predicted'], margins=True)
        acc = np.sum(np.diag(table)[:-1]) / self.n
        return table, acc
#%%
iteration = 300
lr = 0.3

exmodel = BuildModel(n, p, K)
print(exmodel.alpha, exmodel.beta)
optimizer = tf.keras.optimizers.SGD(lr)
#%%
for j in range(iteration):
    with tf.GradientTape() as tape:
        result = exmodel(X)
        loss = -tf.reduce_mean(tf.multiply(y, tf.math.log(result + 1e-8)))         
    # update
    grad = tape.gradient(loss, exmodel.trainable_weights)
    optimizer.apply_gradients(zip(grad, exmodel.trainable_weights)) # 1 update
    
    print(loss)
#%%
print(exmodel.alpha, exmodel.beta)
#%%
table, acc = exmodel.accuracy(X, y_true)
print(table)
print(acc)
#%%
new_x = tf.cast(np.array(np.ones((1, 13))), tf.float32)
pred = exmodel.predict(new_x)
pred_prob = [pred.numpy()[0, 0]] + \
    [pred.numpy()[0, i+1] - pred.numpy()[0, i] for i in range(K - 1 - 1)] + \
    [1 - pred.numpy()[0, -1]]
#%%