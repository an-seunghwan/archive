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
n = 1000
x = np.random.normal(size=(n, 1))
beta0 = [-1, 2, 7]
beta1 = 1

index = np.random.choice(np.arange(len(beta0)), n, replace=True)
y = np.take(beta0, index)[:, None] + x * beta1 + np.random.normal(size=(n, 1))
#%%
plt.figure(figsize=(7, 7))
plt.scatter(x, y)
plt.show()
plt.close()
#%%
import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(x))
regr = model.fit()
print(regr.summary())

fitted = regr.params[0] + regr.params[1] * x
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

model_A = sm.OLS(y_A, sm.add_constant(x_A))
regr_A = model_A.fit()
print(regr_A.summary())

fitted_A = regr_A.params[0] + regr_A.params[1] * x_A
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

model_B = sm.OLS(y_B, sm.add_constant(x_B))
regr_B = model_B.fit()
print(regr_B.summary())

fitted_B = regr_B.params[0] + regr_B.params[1] * x_B
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
ic(regr_B.params[0], regr.params[0], regr_A.params[0])
ic(beta0)
#%%