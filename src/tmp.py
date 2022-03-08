#%%
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# from sqlalchemy import asc
# font_path = "C:/Windows/Fonts/malgunsl.ttf"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
import numpy as np
#%%
data = pd.read_csv('/Users/anseunghwan/Downloads/자동차인게이지먼트_누적.csv', encoding='cp949')
#%%
namelist=data.sort_values(by=['like_cnt'],axis=0,ascending=False)['ch_name'].unique().tolist()
zerolist=[None for _ in range(len(namelist))]
xlist=[None for _ in range(len(namelist))]

fig = plt.figure(figsize=(20,20)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성


x1=data[data['ch_name']=='기아']['data_date']

for i,c in enumerate(namelist):
#follower top10
    zerolist[i]=data[data['ch_name']==c]['like_cnt'].sort_index()
    xlist[i]=data[data['ch_name']==c]['data_date'].sort_index()
for i,c in enumerate(namelist):    
    if c=='기아': 
        plt.scatter(xlist[i],zerolist[i],color='#c02ad1',label='기아',linewidth='5') ## 선그래프 생성
    else:    
        plt.scatter(xlist[i],zerolist[i],label=c) ## 선그래프 생성
       

      # X축의 범위: [xmin, xmax]
    # Y축의 범위: [ymin, ymax]

# plt.ylim(ymin=-4000,ymax=1500)     # Y축의 범위: [ymin, ymax]

plt.legend()

plt.xticks(rotation=45) ## x축 눈금 라벨 설정 - 45도 회전 
plt.title('누적 좋아요  추이(기아자동차, 업계경향)',fontsize=20) ## 타이틀 설정
plt.show()
#%%