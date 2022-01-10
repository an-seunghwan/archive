#%%
import pandas as pd
import numpy as np
import os
os.chdir('/Users/anseunghwan/Documents/uos/통계청+skt/')
#%%
gagu2017 = pd.read_csv('./전수_인구패널데이터_가구_20220107_78241_데이터/전수_인구패널데이터_가구_2017_20220107_78241.csv', encoding='cp949')
gagu2019 = pd.read_csv('./전수_인구패널데이터_가구_20220107_78241_데이터/전수_인구패널데이터_가구_2019_20220107_78241.csv', encoding='cp949')
#%%
ingu2017 = pd.read_csv('./전수_인구패널데이터_인구_20220110_01924_데이터/전수_인구패널데이터_인구_2017_20220110_01924.csv', encoding='cp949', header=None)
ingu2019 = pd.read_csv('./전수_인구패널데이터_인구_20220110_01924_데이터/전수_인구패널데이터_인구_2019_20220110_01924.csv', encoding='cp949', header=None)
ingu2017.columns = [
    '기준년도',
    '행정구역코드(시군구)',
    '가구번호',
    '가구원번호',
    '내외국인구분코드',
    '성별코드',
    '만나이',
    '가구주관계코드',
    '1년전거주지행정구역코드',
    '국적코드',
    '입국연도',
    '국적취득연도',
    '횡단가중값',
    '종단가중값',
]
ingu2019.columns = [
    '기준년도',
    '행정구역코드(시군구)',
    '가구번호',
    '가구원번호',
    '내외국인구분코드',
    '성별코드',
    '만나이',
    '가구주관계코드',
    '1년전거주지행정구역코드',
    '국적코드',
    '입국연도',
    '국적취득연도',
    '횡단가중값',
    '종단가중값',
]
#%%
'''seoul'''
gagu2017_seoul = gagu2017[gagu2017['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
gagu2019_seoul = gagu2019[gagu2019['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
ingu2017_seoul = ingu2017[ingu2017['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) == '11']
ingu2019_seoul = ingu2019[ingu2019['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) == '11']

gagu2017_not = gagu2017[gagu2017['행정구역시군구코드'].apply(lambda x: str(x)[:2]) != '11']
gagu2019_not = gagu2019[gagu2019['행정구역시군구코드'].apply(lambda x: str(x)[:2]) != '11']
ingu2017_not = ingu2017[ingu2017['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) != '11']
ingu2019_not = ingu2019[ingu2019['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) != '11']
#%%
'''select 2017, 2019 each'''
gagu2017_seoul = gagu2017_seoul[gagu2017_seoul['기준년도'] == 2017]
gagu2019_seoul = gagu2019_seoul[gagu2019_seoul['기준년도'] == 2019]
ingu2017_seoul = ingu2017_seoul[ingu2017_seoul['기준년도'] == 2017]
ingu2019_seoul = ingu2019_seoul[ingu2019_seoul['기준년도'] == 2019]

gagu2017_not = gagu2017_not[gagu2017_not['기준년도'] == 2017]
gagu2019_not = gagu2019_not[gagu2019_not['기준년도'] == 2019]
ingu2017_not = ingu2017_not[ingu2017_not['기준년도'] == 2017]
ingu2019_not = ingu2019_not[ingu2019_not['기준년도'] == 2019]
#%%
'''가구 데이터와 인구 데이터 병합'''
df17_seoul = pd.merge(gagu2017_seoul[['가구번호', '가구원수']], ingu2017_seoul[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')
df19_seoul = pd.merge(gagu2019_seoul[['가구번호', '가구원수']], ingu2019_seoul[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')

df17_not = pd.merge(gagu2017_not[['가구번호', '가구원수']], ingu2017_not[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')
df19_not = pd.merge(gagu2019_not[['가구번호', '가구원수']], ingu2019_not[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')
#%%
'''가구번호 & 가구원번호를 기준으로 병합 (조사단위 = 개인)'''
df_seoul = pd.merge(df17_seoul, df19_seoul, how='inner', on=['가구번호', '가구원번호'])
df_seoul = df_seoul.reindex(sorted(df_seoul.columns), axis=1)

df_not = pd.merge(df17_not, df19_not, how='inner', on=['가구번호', '가구원번호'])
df_not = df_not.reindex(sorted(df_not.columns), axis=1)
#%%
'''transition probabilty (alpha)'''
alpha = {}
for idx in range(1, 4): # idx = 가구원수
    if idx != 3:
        df_seoul['가구원수_x{}'.format(idx)] = df_seoul['가구원수_x'].apply(lambda x: 0 if x != idx else 1)
        df_seoul['가구원수_y{}'.format(idx)] = df_seoul['가구원수_y'].apply(lambda x: 0 if x != idx else 1)
    else:
        df_seoul['가구원수_x{}'.format(idx)] = df_seoul['가구원수_x'].apply(lambda x: 0 if x < idx else 1)
        df_seoul['가구원수_y{}'.format(idx)] = df_seoul['가구원수_y'].apply(lambda x: 0 if x < idx else 1)

    probs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            probs[i, j] = ((df_seoul['가구원수_x{}'.format(idx)] == i).astype(float) * (df_seoul['가구원수_y{}'.format(idx)] == j).astype(float)).sum() / df_seoul.shape[0]
    alpha[idx] = [probs[1, 1] / np.sum(probs[1, :]), 
                  1 - probs[1, 1] / np.sum(probs[1, :])]
    # df_seoul[df_seoul['가구원수_x'] == 1][df_seoul['가구원수_y'] == 1].shape[0] / df_seoul[df_seoul['가구원수_x'] == 1].shape[0]
alpha
#%%
beta = {}
for idx in range(1, 4):
    if idx != 3:
        p1 = gagu2017_seoul['가구원수'][gagu2017_seoul['가구원수'] == idx].sum() / gagu2017_seoul['가구원수'].sum()
        p2 = gagu2019_seoul['가구원수'][gagu2019_seoul['가구원수'] == idx].sum() / gagu2019_seoul['가구원수'].sum()
    else:
        p1 = gagu2017_seoul['가구원수'][gagu2017_seoul['가구원수'] >= idx].sum() / gagu2017_seoul['가구원수'].sum()
        p2 = gagu2019_seoul['가구원수'][gagu2019_seoul['가구원수'] >= idx].sum() / gagu2019_seoul['가구원수'].sum()
    
    beta[idx] = [(alpha[idx][0] * p1 - p2 + 1 - p1) / (1 - p1) , 
                 1 - (alpha[idx][0] * p1 - p2 + 1 - p1) / (1 - p1)]
beta
#%%
'''joint prob'''
result = {}
for idx in range(1, 4):
    if idx != 3:
        p1 = gagu2017_seoul['가구원수'][gagu2017_seoul['가구원수'] == idx].sum() / gagu2017_seoul['가구원수'].sum()
        p2 = gagu2019_seoul['가구원수'][gagu2019_seoul['가구원수'] == idx].sum() / gagu2019_seoul['가구원수'].sum()
    else:
        p1 = gagu2017_seoul['가구원수'][gagu2017_seoul['가구원수'] >= idx].sum() / gagu2017_seoul['가구원수'].sum()
        p2 = gagu2019_seoul['가구원수'][gagu2019_seoul['가구원수'] >= idx].sum() / gagu2019_seoul['가구원수'].sum()
        
    joint = np.zeros((2, 2))
    joint[1, 1] = alpha[idx][0] * p1
    joint[1, 0] = (1 - alpha[idx][0]) * p1
    joint[0, 1] = (1 - beta[idx][0]) * (1 - p1)
    joint[0, 0] = beta[idx][0] * (1 - p1)
    result[idx] = joint
result
#%%
'''transition probabilty (alpha)'''
alpha = {}
for idx in range(1, 4): # idx = 가구원수
    if idx != 3:
        df_not['가구원수_x{}'.format(idx)] = df_not['가구원수_x'].apply(lambda x: 0 if x != idx else 1)
        df_not['가구원수_y{}'.format(idx)] = df_not['가구원수_y'].apply(lambda x: 0 if x != idx else 1)
    else:
        df_not['가구원수_x{}'.format(idx)] = df_not['가구원수_x'].apply(lambda x: 0 if x < idx else 1)
        df_not['가구원수_y{}'.format(idx)] = df_not['가구원수_y'].apply(lambda x: 0 if x < idx else 1)

    probs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            probs[i, j] = ((df_not['가구원수_x{}'.format(idx)] == i).astype(float) * (df_not['가구원수_y{}'.format(idx)] == j).astype(float)).sum() / df_not.shape[0]
    alpha[idx] = [probs[1, 1] / np.sum(probs[1, :]), 1 - probs[1, 1] / np.sum(probs[1, :])]
    # df_not[df_not['가구원수_x'] == 1][df_not['가구원수_y'] == 1].shape[0] / df_not[df_not['가구원수_x'] == 1].shape[0]
alpha
#%%
beta = {}
for idx in range(1, 4):
    if idx != 3:
        p1 = gagu2017_not['가구원수'][gagu2017_not['가구원수'] == idx].sum() / gagu2017_not['가구원수'].sum()
        p2 = gagu2019_not['가구원수'][gagu2019_not['가구원수'] == idx].sum() / gagu2019_not['가구원수'].sum()
    else:
        p1 = gagu2017_not['가구원수'][gagu2017_not['가구원수'] >= idx].sum() / gagu2017_not['가구원수'].sum()
        p2 = gagu2019_not['가구원수'][gagu2019_not['가구원수'] >= idx].sum() / gagu2019_not['가구원수'].sum()
    
    beta[idx] = [(alpha[idx][0] * p1 - p2 + 1 - p1) / (1 - p1) , 1 - (alpha[idx][0] * p1 - p2 + 1 - p1) / (1 - p1)]
beta
#%%
'''joint prob'''
result = {}
for idx in range(1, 4):
    if idx != 3:
        p1 = gagu2017_not['가구원수'][gagu2017_not['가구원수'] == idx].sum() / gagu2017_not['가구원수'].sum()
        p2 = gagu2019_not['가구원수'][gagu2019_not['가구원수'] == idx].sum() / gagu2019_not['가구원수'].sum()
    else:
        p1 = gagu2017_not['가구원수'][gagu2017_not['가구원수'] >= idx].sum() / gagu2017_not['가구원수'].sum()
        p2 = gagu2019_not['가구원수'][gagu2019_not['가구원수'] >= idx].sum() / gagu2019_not['가구원수'].sum()
        
    joint = np.zeros((2, 2))
    joint[1, 1] = alpha[idx][0] * p1
    joint[1, 0] = (1 - alpha[idx][0]) * p1
    joint[0, 1] = (1 - beta[idx][0]) * (1 - p1)
    joint[0, 0] = beta[idx][0] * (1 - p1)
    result[idx] = joint
result
#%%