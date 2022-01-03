#%%
import pandas as pd
import numpy as np
#%%
regions = ['aa', 'bb', 'cc']
n = 24
data = np.random.choice(regions, size=(100, n), replace=True)
df = pd.DataFrame(data)
#%%
for col in df.columns:
    df[col] = df[col].apply(lambda x: [str(ord(y)) for y in x])
    df[col] = df[col].apply(lambda x: int(str(''.join(x))))
#%%
'''naive'''
idx = 4
count = 0
y = df.iloc[idx][0]
for x in df.iloc[idx]:
    if y != x:
        count += 1
        y = x
count
#%%
'''flex'''
np.diff(np.array(df.iloc[idx]))
len(np.diff(np.array(df.iloc[idx])))
(np.diff(np.array(df.iloc[idx])) != 0).sum()
#%%
df['change'] = df.apply(lambda x: (np.diff(np.array(x)) != 0).sum(), axis=1)
df['change'][idx]
#%%