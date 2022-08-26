# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:39:12 2021

@author: Duecon
"""
#import scipy
import scipy.stats as stats
from scipy.stats import norm
import numpy

print('METRICAS DA QUALIDADE')

# define constants
md = float(input("Entre com a media:  "))
dp = float(input("Entre com Desvio padrao :  "))
lse = float(input("Entre com Limite Superior de Especificao:  "))
lie = float(input("Entre com Limite Inferior de Especificao:  "))
print('''                      
      ''')
print('''Valores das Métricas ''')

# calcula  z-transform
z1 = round((( md - lie) / dp), 4)
#zgrafico = round((( lie - md) / dp), 4)
z2 = round((( lse - md ) / dp), 4)
print(' Score Z do LIE = ',z1)
print(' Score Z do LSE = ',z2)

#calculo do ppm atraves do z-score
ppm_lie = round(((1-stats.norm.cdf(z1))*(10**6)),2)
ppm_lse = round(((1-stats.norm.cdf(z2))*(10**6)),2)
ppm_total = round(( ppm_lie + ppm_lse),2)
print(' PPM LIE = ',ppm_lie)
print(' PPM LSE = ',ppm_lse)
print(' PPM TOTAL = ',ppm_total)

#calculo valor zbench e yeld
z1_2 = (1-((1-stats.norm.cdf(z1))+(1-stats.norm.cdf(z2))))
zbench =round( norm.ppf(z1_2),4)
print(' ZBench =',zbench)
#print(f' yeld = {z1_2*100:.8f}%')
print('''        ''')
#formula para calculo nivel sigma com deslocamento zbench+1.5=nivel sigma
print(' Metricas da Qualidade')
yeldcp = round(norm.cdf(zbench),10)
dpmocp = round((1-yeldcp)*10**6,3)
nsigmacp = round(zbench+1.5, 4)
print(f' Nivel Sigma curto Prazo = {nsigmacp}')
print(f' yeld = {yeldcp*100:.8f}%')
print(f' DPMO = {dpmocp}')
print(f' Nivel Sigma longo Prazo = {zbench}')





