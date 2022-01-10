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
'''only seoul'''
gagu2017 = gagu2017[gagu2017['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
gagu2019 = gagu2019[gagu2019['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']

ingu2017 = ingu2017[ingu2017['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) == '11']
ingu2019 = ingu2019[ingu2019['행정구역코드(시군구)'].apply(lambda x: str(x)[:2]) == '11']
#%%
'''select 2017 and 2019 each'''
'''
각 기준년도마다 서로 겹치는 가구번호가 존재
기준년도를 필요한 년도만 추출하고 (2017, 2019)
기준년도에 존재하는 가구번호를 이용해 병합
즉, 15-~, 16-~ 등의 번호는 unique한 id임
'''
gagu2017 = gagu2017[gagu2017['기준년도'] == 2017]
gagu2019 = gagu2019[gagu2019['기준년도'] == 2019]

ingu2017 = ingu2017[ingu2017['기준년도'] == 2017]
ingu2019 = ingu2019[ingu2019['기준년도'] == 2019]
#%%
'''가구 데이터와 인구 데이터 병합'''
df17 = pd.merge(gagu2017[['가구번호', '가구원수']], ingu2017[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')
df19 = pd.merge(gagu2019[['가구번호', '가구원수']], ingu2019[['가구번호', '가구원번호', '성별코드', '만나이']], how='inner', on='가구번호')
#%%
'''가구번호 & 가구원번호를 기준으로 병합 (조사단위 = 개인)'''
df = pd.merge(df17, df19, how='inner', on=['가구번호', '가구원번호'])
df = df.reindex(sorted(df.columns), axis=1)
#%%
'''marginal prob'''
result = {}
for idx in range(1, 4): # idx = 가구원수
    df['가구원수_x{}'.format(idx)] = df['가구원수_x'].apply(lambda x: 0 if x != idx else 1)
    df['가구원수_y{}'.format(idx)] = df['가구원수_y'].apply(lambda x: 0 if x != idx else 1)

    probs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            probs[i, j] = ((df['가구원수_x{}'.format(idx)] == i).astype(float) * (df['가구원수_y{}'.format(idx)] == j).astype(float)).sum() / df.shape[0]
    result[idx] = probs
result
#%%
df.to_csv('./panel.csv', encoding='cp949')
#%%