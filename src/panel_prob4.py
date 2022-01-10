#%%
import pandas as pd
import numpy as np
import os
os.chdir('/Users/anseunghwan/Documents/uos/통계청+skt/')
#%%
gagu2017 = pd.read_csv('./전수_인구패널데이터_가구_20220107_78241_데이터/전수_인구패널데이터_가구_2017_20220107_78241.csv', encoding='cp949')
gagu2018 = pd.read_csv('./전수_인구패널데이터_가구_20220107_78241_데이터/전수_인구패널데이터_가구_2018_20220107_78241.csv', encoding='cp949')
gagu2019 = pd.read_csv('./전수_인구패널데이터_가구_20220107_78241_데이터/전수_인구패널데이터_가구_2019_20220107_78241.csv', encoding='cp949')
#%%
'''only seoul'''
gagu2017 = gagu2017[gagu2017['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
gagu2018 = gagu2018[gagu2018['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
gagu2019 = gagu2019[gagu2019['행정구역시군구코드'].apply(lambda x: str(x)[:2]) == '11']
#%%
'''select 2017, 2018, 2019 each'''
'''
각 기준년도마다 서로 겹치는 가구번호가 존재
기준년도를 필요한 년도만 추출하고 (2017, 2018, 2019)
기준년도에 존재하는 가구번호를 이용해 병합
즉, 15-~, 16-~ 등의 번호는 unique한 id임
'''
gagu2017 = gagu2017[gagu2017['기준년도'] == 2017]
gagu2018 = gagu2018[gagu2018['기준년도'] == 2018]
gagu2019 = gagu2019[gagu2019['기준년도'] == 2019]
#%%
# transition = np.zeros((2, 2))
# for idx in range(1, 4):
idx = 1
A = np.zeros((2, 2))
B = np.zeros((2, 1))

p1 = gagu2017['가구원수'][gagu2017['가구원수'] == idx].sum() / gagu2017['가구원수'].sum()
p2 = gagu2018['가구원수'][gagu2018['가구원수'] == idx].sum() / gagu2018['가구원수'].sum()
p3 = gagu2019['가구원수'][gagu2019['가구원수'] == idx].sum() / gagu2019['가구원수'].sum()

A[0, 0] = p1
A[0, 1] = p1 - 1
A[1, 0] = p2
A[1, 1] = p2 - 1

B[0, 0] = p1 + p2 - 1
B[1, 0] = p2 + p3 - 1

alpha = (np.linalg.inv(A) @ B)[0, 0]
beta = (np.linalg.inv(A) @ B)[1, 0]
#%%

#%%
# transition = np.zeros((2, 2))
# A = np.zeros((2, 2))
# B = np.zeros((2, 1))
# p1 = 0.5
# p2 = 0.3
# p3 = 0.4

# A[0, 0] = p1
# A[0, 1] = p1 - 1
# A[1, 0] = p2
# A[1, 1] = p2 - 1

# B[0, 0] = p1 + p2 - 1
# B[1, 0] = p2 + p3 - 1

# np.linalg.inv(A) @ B
#%%