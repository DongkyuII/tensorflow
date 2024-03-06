
import pandas as pd

data = pd.read_csv('gpascore.csv')

#빈칸 세기
data.isnull().sum()
#print(data.isnull().sum())

#NaN/빈값 있는 행 제거
data = data.dropna()

# #NaN/빈값에 원하는 값 넣기 (보통 평균값 넣음)
# data.fillna(100)

# #원하는 컬럼만 고를때 
# print(data['gpa'])

# #원하는 컬럼의 최저값, 최대값, 로우 개수
# print(data['gpa'].min())
# print(data['gpa'].max())
# print(data['gpa'].count())

#data['admit']의 값을 리스트로 받아올때 
ydata = data['admit'].values
xdata = []
for i, rows in data.iterrows():
    xdata.append([rows['gre'], rows['gpa'], rows['rank']])

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# x데이터 : 학습데이터, y데이터 : 정답
# epochs=10 -> 몇번 학습할건지 

model.fit( np.array(xdata), np.array(ydata), epochs=1000)

#예측하기
예측값 = model.predict( [ [750,3.70,3], [400,2.20,1] ] )
print(예측값)

