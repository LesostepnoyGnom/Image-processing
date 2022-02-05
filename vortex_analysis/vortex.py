# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:44:53 2022

@author: 1618047
"""

import sympy as sy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

t = -1 # константы для уравнений
C1 = 10
C2 = 5

Lcr = sy.Symbol('Lcr') # искомые переменные
Lci = sy.Symbol('Lci') 

'''Расчёт лямбд'''
# =========================================================================
os.chdir("D:/abyss_of_work/main/Aleksandr_G.K/2021/Image-processing/vortex_analysis/index")
num_of_images = len(os.listdir('.'))
for j in range(num_of_images):
    
    df = pd.read_csv('exp_'+str(j)+'.txt', sep='\t') # читаем таблицу
    df.to_excel('output.xlsx', 'Sheet1', index=False) #перевд в эксель
    
    data = pd.read_excel('output.xlsx')
    
    lambda_ci1 = [] # пустые массивы для расчёта
    lambda_ci2 = []
    
    for i in range(len(data)): # решаем уравнения
        X = data.loc[i, 'u']
        Y = data.loc[i, 'v']
        
        U1 = sy.exp(Lcr*t) * (C1*sy.cos(Lci*t) + C2*sy.sin(Lci*t)) - X
        U2 = sy.exp(Lcr*t) * (C2*sy.cos(Lci*t) + C1*sy.sin(Lci*t)) - Y
    
        L = sy.solve([U1, U2], [Lcr, Lci])
    
        lambda_ci1.append(L[1][0])
        lambda_ci2.append(L[1][1])
    
        print(i,',', L[1][0],',', L[1][1])
    
    print(lambda_ci1, lambda_ci2)
    
    dataf = pd.DataFrame({'Lci1': lambda_ci1, 'Lci2': lambda_ci2})
    writer = pd.ExcelWriter("Lci_"+str(j)+".xlsx", engine='xlsxwriter')
    dataf.to_excel(writer, sheet_name='1')
    writer.save()
# =========================================================================

Lci_data = pd.read_excel('Lci_0.xlsx')

Real1 = []
Image1 = []
Real2 = []
# =========================================================================
'''Делаем сетку координат'''
x = data['# x'].to_list()
y = data['y'].to_list()

count_x = x.count(x[0])
count_y = y.count(y[0])

xx = x[:count_y]
yy = y[::count_y]

X, Y = np.meshgrid(tuple(xx), tuple(yy))
# =========================================================================
'''
Вытаскиваем данные из таблицы в переменные
Image1 - Мнимая часть первого решения
Real1 - Действительная часть первого решения
Real2 - Действительная часть второго решения
У второго решения нет мнимой части
'''

for i in range(len(data)):
    x.append(data.loc[i, '# x'])
    y.append(data.loc[i, 'y'])

    Image1.append(float(Lci_data.loc[i, 'Lci1'][-18:-2]))
    Real1.append(float(Lci_data.loc[i, 'Lci1'][0:15]))
    Real2.append(Lci_data.loc[i, 'Lci2'])
# =========================================================================
"""Находим максимальные и минимальные значения в полученных массивах для построения графиков"""
Image1_min, Image1_max = -np.abs(Image1).max(), np.abs(Image1).max()
Real1_min, Real1_max = -np.abs(Real1).max(), np.abs(Real1).max()
Real2_min, Real2_max = -np.abs(Real2).max(), np.abs(Real2).max()
# =========================================================================
print('X =',len(X[1]))
print('Y =',len(Y))
 
Im1 = []
Re1 = []
Re2 = []

count = 0

for i in range(len(Y)):
    a=[]
    b=[]
    c=[]
    for j in range(len(X[1])):
        a.append(Image1[count])
        b.append(Real1[count])
        c.append(Image1[count])
        count += 1
        # print(count)
    Im1.append(a)
    Re1.append(b)
    Re2.append(c)
Im1 = np.array(Im1)
Re1 = np.array(Re1)
Re2 = np.array(Re2)

fig, ax = plt.subplots(figsize=(12,10),dpi=300)

Real1_c = ax.pcolormesh(X, Y, Re1, cmap='RdBu', vmin=Real1_min, vmax=Real1_max)
# Image1_c = ax.pcolormesh(x, y, Image1, cmap='RdBu', vmin=Image1_min, vmax=Image1_max) 
# Real2_c = ax.pcolormesh(x, y, Real2, cmap='RdBu', vmin=Real2_min, vmax=Real2_max)

ax.set_title('Real1')
# ax[2].set_title('Image1')
# ax[3].set_title('Real2')

ax.axis([X.min(), X.max(), Y.min(), Y.max()])
# ax[2].axis([x.min(), x.max(), y.min(), y.max()])
# ax[3].axis([x.min(), x.max(), y.min(), y.max()])

fig.colorbar(Real1_c, ax=ax)
# fig.colorbar(Image1_c, ax=ax[2])
# fig.colorbar(Real2_c, ax=ax[3])

plt.show()