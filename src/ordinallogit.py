#%% import module

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% data preprocessing

# read data
data = pd.read_csv(r'C:\Users\uos\iCloudDrive\ordinal\bostonhousing_ord.csv')
# data = pd.read_csv(r'C:\Users\uos\iCloudDrive\ordinal\kinematics.csv')
# data = pd.read_csv(r'C:\Users\uos\iCloudDrive\ordinal\stock_ord.csv')
print(np.shape(data))                 # dimension 
print(data.head())
y = np.array(data['response']) - 1     
K = len(np.unique(y))                 # number of category
y = tf.keras.utils.to_categorical(y)  # one-hot encoding
y = tf.cast(y, tf.float32)

X = np.array(data.iloc[:, 1:])        # design matrix
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # scaling
X = tf.cast(X, tf.float32)

n = len(data)                         # number of observation
p = len(data.columns) - 1             # number of predictor

#%% initial value
# alpha = tf.Variable([np.zeros(K-1)])
# alpha = tf.cast(alpha, tf.float32)
# beta = tf.Variable(np.transpose([np.random.normal(0,1,p)]))
# beta = tf.cast(beta, tf.float32)

alpha = tf.Variable([np.random.normal(0,1,K-1)])
alpha = tf.cast(alpha, tf.float32)
beta = tf.Variable(np.transpose([np.random.normal(0,1,p)]))
beta = tf.cast(beta, tf.float32)

#%%

lr = 0.3  # learning rate
for j in tqdm(range(500)):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(alpha)
        tape.watch(beta)
        
        '''reparametrization for positive condition'''
        theta = []
        theta.append(alpha[0, 0])
        for i in range(1, K-1):
            theta.append(tf.square(alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]

        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, beta))
        mat1 = tf.concat((mat1, tf.ones((n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, beta))
        mat2 = tf.concat((tf.zeros((n, 1)), mat2), axis=-1)

        loss = -tf.reduce_sum(tf.multiply(y, tf.math.log(mat1 - mat2 + 1e-8))) / n
    
    grad1 = tape.gradient(loss, alpha)
    grad2 = tape.gradient(loss, beta)
    tf.reduce_sum(tf.concat([grad1**2, tf.transpose(grad2)**2], axis=1))
        
    alpha = alpha - lr * grad1
    beta = beta - lr * grad2

grad = 10
while grad > 1e-6:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(alpha)
        tape.watch(beta)
        
        '''reparametrization for positive condition'''
        theta = []
        theta.append(alpha[0, 0])
        for i in range(1, K-1):
            theta.append(tf.square(alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]

        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, beta))
        mat1 = tf.concat((mat1, tf.ones((n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, beta))
        mat2 = tf.concat((tf.zeros((n, 1)), mat2), axis=-1)

        loss = -tf.reduce_sum(tf.multiply(y, tf.math.log(mat1 - mat2 + 1e-8))) / n
    
    grad1 = tape.gradient(loss, alpha)
    grad2 = tape.gradient(loss, beta)
    grad = tf.reduce_sum(tf.concat([grad1**2, tf.transpose(grad2)**2], axis=1))
        
    alpha = alpha - lr * grad1
    beta = beta - lr * grad2



#%% prediction & model performence

y_pred = np.argmax(np.array(mat1 - mat2), axis=1)
y_true = np.array(data['response']) - 1
con_table = pd.crosstab(y_true, y_pred,rownames=['True'], colnames=['Predicted'], margins=True)
acc = np.sum(np.diag(con_table)[:-1])/n
mae = np.mean(np.abs(y_pred - y_true)) 

print(loss)
print(theta)
print(beta)
print(con_table) # confusion mattrix
print(acc)       # accrucy
print(mae)       # mean square error
print(mae)       # mean absolute error
# print(np.array(mat1 - mat2))
# kendalls tau

#%%

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(y_pred,bins=K, range=[-0.5, 4.5], alpha=0.33,label='pred',color='blue')
plt.hist(y_true,bins=K, range=[-0.5, 4.5], alpha=0.33,label='true',color='orange')
plt.title('Boston Housing - Validation - Output Distribution',fontsize=14)
plt.xlabel('PREDICTED',fontsize=16)
plt.ylabel('COUNT',fontsize=16)
plt.legend()
plt.tight_layout()

#%% Q n A

# why loss convex?
# Interpretation? 
# k-fold cv or LOOCV ?
# variable selection?
# goodness of git of model ?
# 초기값, learning rate 설정?
# std error estimation & wald test of estimator
# regularization? (when design matrix is singular)

# 피셔 스코어링 으로 ML 핏 함?

