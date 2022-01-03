#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%
'''OLS'''
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head()
#%%
# categorical variables
res = smf.ols(formula='Lottery ~ Literacy + Wealth + Literacy*Wealth + C(Region)', data=df).fit()
res.summary()
# np.unique(df['Region'])
#%%
'''binomial'''
star98 = pd.DataFrame(sm.datasets.star98.load_pandas().data)
# no constant
formula = 'SUCCESS ~ -1 + LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT + \
           PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
# constant
formula = 'SUCCESS ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT + \
           PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
dta = star98[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP',
              'PCTCHRT', 'PCTYRRND', 'PERMINTE', 'AVYRSEXP', 'AVSALK',
              'PERSPENK', 'PTRATIO', 'PCTAF']].copy()
endog = dta['NABOVE'] / (dta['NABOVE'] + dta['NBELOW'])
del dta['NABOVE']
dta['SUCCESS'] = endog
#%%
model = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
model.summary()
model.params
#%%
'''poisson'''
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
#%%
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
#%%
formula = "PRICE ~ RM + PTRATIO + LSTAT"
# link = sm.genmod.families.links.log
# family = sm.families.Poisson(link=link)
model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
result = model.fit() 
result.summary()
# print('AIC:', result.aic)
#%%