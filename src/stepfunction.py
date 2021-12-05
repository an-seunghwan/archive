#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#%%
x = np.arange(100)
y = [0.1] * 40  + [1] * 60
plt.figure(figsize=(5, 3))
plt.step(x, y)
plt.xlabel('픽셀 위치')
plt.ylabel('픽셀 값')
plt.show()
#%%
x = np.arange(100)
y = 1 / (1 + np.exp(-(x-40)/5))
plt.figure(figsize=(5, 3))
plt.plot(x, y)
plt.xlabel('픽셀 위치')
plt.ylabel('픽셀 값')
plt.show()
#%%