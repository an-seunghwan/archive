#%%
import pandas as pd
import numpy as np
import os
os.chdir('/Users/anseunghwan/Documents/uos/통계청+skt/전수_인구패널데이터_가구_20220107_78241_데이터')
#%%
df2017 = pd.read_csv('./전수_인구패널데이터_가구_2017_20220107_78241.csv', encoding='cp949')
df2019 = pd.read_csv('./전수_인구패널데이터_가구_2019_20220107_78241.csv', encoding='cp949')
#%%
# np.unique(df2017['가구번호'].apply(lambda x: str(x)[:2]))
# np.unique(df2019['가구번호'].apply(lambda x: str(x)[:2]))
df17 = df2017[df2017['가구번호'].apply(lambda x: str(x)[:2]) == '17']
df19 = df2019[df2019['가구번호'].apply(lambda x: str(x)[:2]) == '19']
#%%
'''only seoul'''
df17 = df17[df17['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
# len(np.unique(df17['가구번호']))
# np.unique(df17['가구번호'].apply(lambda x: str(x)[:2]))
df17['가구번호'] = df17['가구번호'].apply(lambda x: str(x)[3:])

df19 = df19[df19['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
# len(np.unique(df19['가구번호']))
# np.unique(df19['가구번호'].apply(lambda x: str(x)[:2]))
df19['가구번호'] = df19['가구번호'].apply(lambda x: str(x)[3:])
#%%
'''가구번호를 기준으로 병합'''
df = pd.merge(df17[['가구번호', '가구원수']], df19[['가구번호', '가구원수']], how='inner', on='가구번호')
#%%
'''marginal prob'''
result = {}
for idx in range(1, 4): # idx = 가구원수
    if idx != 3:
        df['가구원수_x{}'.format(idx)] = df['가구원수_x'].apply(lambda x: 0 if x != idx else 1)
        df['가구원수_y{}'.format(idx)] = df['가구원수_y'].apply(lambda x: 0 if x != idx else 1)
    else:
        df['가구원수_x{}'.format(idx)] = df['가구원수_x'].apply(lambda x: 0 if x < idx else 1)
        df['가구원수_y{}'.format(idx)] = df['가구원수_y'].apply(lambda x: 0 if x < idx else 1)

    probs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            probs[i, j] = ((df['가구원수_x{}'.format(idx)] == i).astype(float) * (df['가구원수_y{}'.format(idx)] == j).astype(float)).sum() / df.shape[0]
    result[idx] = probs
result
#%%