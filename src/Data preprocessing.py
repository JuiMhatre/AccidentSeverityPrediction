import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

df = pd.read_csv('/content/drive/US_Accidents_Dec20_updated.csv')
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
# missing_df = missing_df.loc[missing_df['missing_count']>0 ]
missing_df = missing_df.loc[missing_df['missing_count']/df.shape[0]>0.05]
missing_df = missing_df.sort_values(by='missing_count')
display(missing_df)
ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots()
rects = ax.barh(ind,missing_df.missing_count.values,color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.ticklabel_format(useOffset=False, style='plain', axis='x')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show();
df.drop(['Wind_Chill(F)','Precipitation(in)', 'Number','Wind_Speed(mph)'], axis=1, inplace = True)
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0 ]
for cols in df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns : 
  if cols in missing_df.column_name.values:
    df[cols].fillna(value=df[cols].mean(), inplace=True)
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0 ]
display(missing_df)
# for cols in missing_df.column_name.values:
#   df.drop([cols], axis=1, inplace = True)


df.drop(['City'], axis=1, inplace = True)
df.drop(['Zipcode'], axis=1, inplace = True)
df.drop(['Timezone'], axis=1, inplace = True)
df.drop(['Airport_Code'], axis=1, inplace = True)

missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0 ]
df.drop(['Description'], axis=1, inplace = True)
df.drop(['Street'], axis=1, inplace = True)
df.drop(['County'], axis=1, inplace = True)
df.drop(['State'], axis=1, inplace = True)
df.drop(['Country'], axis=1, inplace = True)
df.drop(['ID'], axis=1, inplace = True)
df.head()
df["Amenity"] = df["Amenity"].astype(int)
df["Bump"] = df["Bump"].astype(int)
df["Give_Way"] = df["Give_Way"].astype(int)
df["Crossing"] = df["Crossing"].astype(int)
df["Junction"] = df["Junction"].astype(int)
df["No_Exit"] = df["No_Exit"].astype(int)
df["Railway"] = df["Railway"].astype(int)
df["Roundabout"] = df["Roundabout"].astype(int)
df["Station"] = df["Station"].astype(int)
df["Stop"] = df["Stop"].astype(int)
df["Traffic_Calming"] = df["Traffic_Calming"].astype(int)
df["Traffic_Signal"] = df["Traffic_Signal"].astype(int)
df["Turning_Loop"] = df["Turning_Loop"].astype(int)
from sklearn.preprocessing import OneHotEncoder

#creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
df['Wind_Direction']= label_encoder.fit_transform(df['Wind_Direction']) 
df['Weather_Condition']= label_encoder.fit_transform(df['Weather_Condition']) 
df = pd.get_dummies(df, columns = ['Sunrise_Sunset',	'Civil_Twilight',	'Nautical_Twilight',	'Astronomical_Twilight', 'Side'])
df['year'] = pd.DatetimeIndex(df['Start_Time']).year
df['month'] = pd.DatetimeIndex(df['Start_Time']).month
df.drop(['Start_Time'], axis=1, inplace = True)
df.drop(['End_Time'], axis=1, inplace = True)
df.drop(['Weather_Timestamp'], axis=1, inplace = True)

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
X = df.loc[:, df.columns != 'Severity']
Y = df.loc[:,['Severity']]
oversample = SMOTE()
# oversample = RandomOverSampler(sampling_strategy='')
X, Y = oversample.fit_resample(X, Y)
scaler = StandardScaler()

#export file after preprocessing and use for scala program
X = scaler.fit_transform(X)
