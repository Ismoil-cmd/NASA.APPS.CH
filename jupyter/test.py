import numpy as np 
import pandas as pd 
from spacepy import pycdf
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import datetime

cdf_dsc1 = pycdf.CDF('dscovr_h0_mag_20220101_v01.cdf')
cdf_dsc2 = pycdf.CDF('dscovr_h0_mag_20210101_v01.cdf')

cdf_swe1 = pycdf.CDF('wi_h1_swe_20220101_v01.cdf')
cdf_swe2 = pycdf.CDF('wi_h1_swe_20210101_v01.cdf')


df2 = pd.DataFrame(cdf_dsc1[1][:579])
df3 = pd.DataFrame(cdf_dsc1[2][:579])
df5 = pd.DataFrame(cdf_dsc1[4][:579])
df6 = pd.DataFrame(cdf_dsc1[5][:579])
df7 = pd.DataFrame(cdf_dsc1[6][:579])

df1_2 = pd.DataFrame(cdf_dsc2[1][:579])
df1_3 = pd.DataFrame(cdf_dsc2[2][:579])
df1_5 = pd.DataFrame(cdf_dsc2[4][:579])
df1_6 = pd.DataFrame(cdf_dsc2[5][:579])
df1_7 = pd.DataFrame(cdf_dsc2[6][:579])


df2 = pd.DataFrame(df2[2])
df2.rename(columns={2:'B1GSE'}, inplace=True)
df3.rename(columns={0:'B1RTN'}, inplace=True)
df5.rename(columns={0:'B1SDGSE'}, inplace=True)
df7.rename(columns={0:'x', 1:'y', 2:'z'}, inplace=True)

df1_2 = pd.DataFrame(df1_2[2])
df1_2.rename(columns={2:'B1GSE'}, inplace=True)
df1_3.rename(columns={0:'B1RTN'}, inplace=True)
df1_5.rename(columns={0:'B1SDGSE'}, inplace=True)
df1_7.rename(columns={0:'x', 1:'y', 2:'z'}, inplace=True)

datas = [df2,df3,df5,df6,df7]

datas_2 = pd.concat([df1_2,df1_3,df1_5, df1_6, df1_7],axis=1)

df_dsc_mn = pd.concat(datas, axis=1)
df_dsc_mn = pd.concat([df_dsc_mn, datas_2], axis=0)

df_dsc_mn.reset_index(drop=True, inplace=True)

print(df_dsc_mn)

print(cdf_swe1)

df2 = pd.DataFrame(cdf_swe1[1][:579])
df11 = pd.DataFrame(cdf_swe1[10][:579])
df23 = pd.DataFrame(cdf_swe1[23][:579])
df31 = pd.DataFrame(cdf_swe1[31][:579])
df32 = pd.DataFrame(cdf_swe1[32][:579])
df42 = pd.DataFrame(cdf_swe1[42][:579])
df43 = pd.DataFrame(cdf_swe1[43][:579])

df1_2 = pd.DataFrame(cdf_swe2[1][:579])
df1_11 = pd.DataFrame(cdf_swe2[10][:579])
df1_23 = pd.DataFrame(cdf_swe2[23][:579])
df1_31 = pd.DataFrame(cdf_swe2[31][:579])
df1_32 = pd.DataFrame(cdf_swe2[32][:579])
df1_42 = pd.DataFrame(cdf_swe2[42][:579])
df1_43 = pd.DataFrame(cdf_swe2[43][:579])


df2.rename(columns={0:'Alpha_VX_nonlin'}, inplace=True)
df11.rename(columns={0:'Alpha_sigmaVZ_nonlin'}, inplace=True)
df23.rename(columns={0:'Epoch'}, inplace=True)
df31.rename(columns={0:'Proton_VZ_moment'}, inplace=True)
df32.rename(columns={0:'Proton_VZ_nonlin'}, inplace=True)
df42.rename(columns={0:'Proton_sigmaVX_nonlin'}, inplace=True)
df43.rename(columns={0:'Proton_sigmaVY_nonlin'}, inplace=True)


df1_2.rename(columns={0:'Alpha_VX_nonlin'}, inplace=True)
df1_11.rename(columns={0:'Alpha_sigmaVZ_nonlin'}, inplace=True)
df1_23.rename(columns={0:'Epoch'}, inplace=True)
df1_31.rename(columns={0:'Proton_VZ_moment'}, inplace=True)
df1_32.rename(columns={0:'Proton_VZ_nonlin'}, inplace=True)
df1_42.rename(columns={0:'Proton_sigmaVX_nonlin'}, inplace=True)
df1_43.rename(columns={0:'Proton_sigmaVY_nonlin'}, inplace=True)


datas_swe = [df2,df23,df11,df31,df32,df42,df43]

datas_swe2 = pd.concat([df1_2,df1_23,df1_11,df1_31,df1_32,df1_42,df1_43], axis=1)


df_swe_mn = pd.concat(datas_swe, axis=1)
df_swe_mn = pd.concat([df_swe_mn, datas_swe2], axis=0)


df_swe_mn.rename(columns={'Alpha_VX_nonlin':'Time'}, inplace=True)

df_swe_mn.reset_index(drop=True, inplace=True)

print(df_swe_mn)

for i in range(len(df_swe_mn['Time'])):
    df_swe_mn['Time'][i] = datetime.datetime.time(df_swe_mn['Time'][i])

print(df_swe_mn['Time'][0])

for i in range(len(df_swe_mn['Time'])):
    df_swe_mn['Time'][i] = float((df_swe_mn['Time'][i].hour%24) + (df_swe_mn['Time'][i].minute%60) + (df_swe_mn['Time'][i].second%3600))

df_swe_mn['Time'] = df_swe_mn['Time'].astype('float')

df_main = pd.concat([df_swe_mn, df_dsc_mn], axis=1)

df_main = df_main.drop(['Proton_VZ_nonlin'], axis=1)

df_main = df_main.drop([1,'Proton_sigmaVX_nonlin'], axis=1)

print(df_main.corr())

y = df_main['Time']
x = df_main.drop(['Time'], axis=1)

model = xgboost.XGBRegressor(n_estimators=800)
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Точность:', round(model.score(x_test, y_test)*100,4), 'процентов')

