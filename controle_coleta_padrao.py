# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:38:31 2022

@author: Duecon
"""
import pandas as pd
import pyodbc 
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import numpy as np
import streamlit as st
from PIL import Image
import plotly.offline as py
import plotly.graph_objects as go

from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date
import datetime
import time

from st_aggrid import GridOptionsBuilder, AgGrid, JsCode





server = 'tcp:177.94.211.154\EGA_PCPMASTER,20133' 
database = 'EGA_CEP' 
username = 'ronaldo.cavicchioli' 
password = 'Eg@Cav$18$' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#cursor = cnxn.cursor()



#carta=2022030718091210
carta=2021052016200733
#carta=2022072016195511




sql_query = pd.read_sql_query(f'''SELECT 
                              [ChartControlID]
      ,[MeasuringPoint]
      ,[SubGroup]
      ,[SampleIndex]
      ,[NewValue]
      ,[Value]
      ,[UtcCreatedAt]
      
  FROM [EGA_CEP].[dbo].[ChartControlSample] 
  where ChartControlID={carta} and SubGroup < '1000' 
  order by SubGroup, MeasuringPoint
  ''',cnxn)

sql_query.drop(columns=['NewValue',], inplace=True)

for i in sql_query.index:
    sql_query['SubGroup'][i]= sql_query['SubGroup'][i]+1  

for i in range(len(sql_query)):
    sql_query['MeasuringPoint'][i] = sql_query.loc[i,'MeasuringPoint'] + 1
    
    

sql_variavel= pd.read_sql_query(f'''
           SELECT  [ChartControlID]    
       ,[Description]
      ,[UnitType]
      ,[SubGroupSize]
       ,[LIE]
      ,[Nominal]
      ,[LSE]
      ,[MinInput]
      ,[MaxInput]
      ,[SampleFrequency]
     
  FROM [EGA_CEP].[dbo].[ChartControlVariable] where ChartControlID={carta}                      
                                 
                                 ''',cnxn)

sql_out_sample= pd.read_sql_query(f'''
                                 
SELECT  [ID]
      ,[ChartControlID]
      ,[MeasuringPoint]
      ,[SubGroup]
      ,[SampleIndex]
      ,[Value]
      ,[UtcCreatedAt]
      
  FROM [EGA_CEP].[dbo].[ChartOutOfRangeSample] where ChartControlID={carta}  
''',cnxn)


sql_out_sample.info()                                

sql_header= pd.read_sql_query(f'''
                SELECT  [ChartControlID]
       ,[OperationDescription]
       ,[PartDescription]
       ,[MachineName]
       ,[UtcCreatedAt]
      ,[UtcChangedAt]
      ,[ProductionOrder]
     
    
  FROM [EGA_CEP].[dbo].[ChartControlHeader]
  where ChartControlID={carta}             
                              
                              ''', cnxn)
                              

#informações da carta
sql_header=sql_header.rename(columns={
           'ChartControlID': 'Carta',
           'MachineName': 'Maquina',
           'OperationDescription': 'Operacao',
           'PartDescription':'Peca',
           
           'ProductionOrder':'Ordem',
           'UtcCreatedAt': 'Data'
    })                              

sql_header= sql_header[['Carta', 'Maquina', 'Ordem',
                        'Operacao','Peca','Data']]



statis=sql_query['Value'].describe()




#ver como transformar em dataframe
#df_gro= sql_query.groupby(['SubGroup','MeasuringPoint'], as_index=False)

#tabela Pivot
#df igual ao geral(todos os pontos) no SPC
#df_geral=sql_query.pivot(['MeasuringPoint','SampleIndex'], columns='SubGroup',values='Value').reset_index()
#df ptos medicao gerar boxplot
#df_box=sql_query.pivot(['SubGroup','SampleIndex'], columns='MeasuringPoint',values='Value').reset_index()
#df_pto=sql_query.pivot(['SubGroup','MeasuringPoint'], columns='SampleIndex',values='Value').reset_index()


tam_subgr=sql_variavel['SubGroupSize'].values[0]



#Gerar array usando tam_subgr +1
subgrupo=np.arange(start=1, stop=tam_subgr+1)
#gerar dataframes 

#gerar carta por pto de medicao e usar no boxplot

df_pto_ind=sql_query.pivot(['MeasuringPoint','SubGroup'], columns='SampleIndex',values='Value').reset_index()

#for i in range(len(df_pto_ind)):
#    df_pto_ind['MeasuringPoint'][i] = df_pto_ind.loc[i,'MeasuringPoint'] + 1
    
    

df=sql_query.loc[:,['UtcCreatedAt','MeasuringPoint','SubGroup', 'Value']]
#cria indice ptos medicao tamanho do subgrupo
#seq= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
seq=subgrupo
df['seq']=pd.Series(seq)
seq1=df['seq']
seq1.dropna(inplace=True)

for i in range(len(df['seq'])):
    df.loc[df['seq'].isnull(),'seq']= seq1.reindex(np.arange(df['seq'].isnull().sum())).values

    
df['seq']=df[['seq']].astype(int)
#reordenar df
df= df[['UtcCreatedAt','MeasuringPoint','SubGroup', 'seq', 'Value']]




#alterar os subgrupos e pontos de medição

#for i in df.index:
#    df['SubGroup'][i]= df['SubGroup'][i]+1  

#for i in range(len(df)):
#    df['MeasuringPoint'][i] = df.loc[i,'MeasuringPoint'] + 1
    
    
    
df['Value']=df[['Value']].astype(float,2) 
#df.round({'Value':2})
 
#df.info()

df_carta= df.pivot(index='SubGroup', columns='seq',values='Value')





df_ptos=sql_query.pivot(['SampleIndex','SubGroup'], columns='MeasuringPoint', values='Value').reset_index()



    
#df_ptos['Value']=df_ptos[['Value']].astype(float,2) 


#estudar forma de inserir horario da amostra
#df_carta= df.pivot(['UtcCreatedAt','SubGroup'], columns='seq',values='Value')


#resulta media e dsv analise da variavel
#df_grouped= sql_query.groupby('SubGroup').agg(['mean','std'])['Value'].reset_index()

#df=df_grouped['mean']

#df_grouped['med_acm']= df_grouped['mean'].cumsum()
#df_pto_gr=sql_query.groupby('MeasuringPoint').agg(['mean','std'])['Value'].reset_index()



#Criar dados da carta calculos estatistica
#print(df_carta.describe())


#tamanho da amostra para constantes
subgrupo_n=df_carta.shape[1]
#numero de subgrupos na carta
nun_subgrupos= df_carta.shape[0]

#constantes para chartXRS  X(A3) XS (B3-B4) conforme tamanho subgrupos
A3=0.78854109
B3=0.428199542
B4=1.571800458
#calculo dsp within
c4=0.982316177162653





#df_carta.info()



#media subgrupo
#df_carta['medSub']= df_carta.loc[:,1:15].mean(axis=1)
df_carta['medSub']=df_carta.mean(axis=1)

#dsv do subgrupo
df_carta['dsv']= df_carta.loc[:,df_carta.columns !='medSub'].std(axis=1)
#df_carta['dsv']= df_carta.iloc[:,0:15].std(axis=1)


#media dsv da amostra
s_bar=df_carta['dsv'].mean()
#media das medias amostras
x_bar=round(df_carta['medSub'].mean(),2)


#formula lscx md_medias(x_bar+(A3*s_bar) licx= md_medias(x_bar -(A3*s_bar))
#formula XS lscs = B4*s_bar   lics = s_bar*B3

def lscx(x_bar,s_bar,A3):
    lscx= x_bar+(A3*s_bar)
    return lscx

def licx(x_bar,s_bar,A3):
    licx= x_bar-(s_bar*A3)
    return licx

Lcsx= round(lscx(x_bar,s_bar,A3),2)
Licx= round(licx(x_bar,s_bar,A3),2)


#calculo para desp Overal e Within
Dspw= round(s_bar/c4,4)
DspO= round(statis['std'], 4)
maximo_entrada=round(statis['max'], 2)
minima_entrada=round(statis['min'],2)
#retorna uma serie
Lse_serie=sql_variavel['LSE']
#retornar valor
Lse=round(Lse_serie[0],2)
Lie_serie=sql_variavel['LIE']
Lie=round(Lie_serie[0],2)
Nominal_serie= sql_variavel['Nominal']
Nominal=round(Nominal_serie[0],2)


#funcao calculo CPK PPK

def Cp(Dspw,Lie,Lse):
    Cp= (Lse-Lie)/(6*Dspw)
    return Cp


def Pp(DspO,Lie,Lse):
    Pp= (Lse-Lie)/(3*DspO)
    return Pp


Cp=round(Cp(Dspw,Lie,Lse),2)
Cps= round((Lse-x_bar)/(3*Dspw),2)
Cpi= round((x_bar-Lie)/(3*Dspw),2)
Cpk=np.min([Cps,Cpi])

Pp=round(Pp(DspO,Lie,Lse),2)
Ppi=round((x_bar-Lie)/(3*DspO),2)
Pps=round((Lse-x_bar)/(3*DspO),2)
Ppk= np.min([Pps,Ppi])
   


#hora atual
hora_atual=time.strftime('%H:%M:%S', time.localtime())






# calcula  z-transform within
z1 = round((( x_bar - Lie) / Dspw), 4)
#zgrafico = round((( lie - md) / dp), 4)
z2 = round((( Lse - x_bar ) / Dspw), 4)
#calculo do ppm atraves do z-score within
ppm_lie = round(((1-stats.norm.cdf(z1))*(10**6)),2)
ppm_lse = round(((1-stats.norm.cdf(z2))*(10**6)),2)
ppm_total = round(( ppm_lie + ppm_lse),2)

#calculo valor zbench e yeld curto prazo
z1_2 = (1-((1-stats.norm.cdf(z1))+(1-stats.norm.cdf(z2))))
zbench =round( norm.ppf(z1_2),4)
#print(f' yeld = {z1_2*100:.8f}%')

# calcula  z-transform overall
z3 = round((( x_bar - Lie) / DspO), 4)
#zgrafico = round((( lie - md) / dp), 4)
z4 = round((( Lse - x_bar ) / DspO), 4)

#calculo do ppm atraves do z-score overall
ppm_lie = round(((1-stats.norm.cdf(z3))*(10**6)),2)
ppm_lse = round(((1-stats.norm.cdf(z4))*(10**6)),2)
ppm_total = round(( ppm_lie + ppm_lse),2)

#calculo valor zbench e yeld longo prazo
z3_4 = (1-((1-stats.norm.cdf(z3))+(1-stats.norm.cdf(z4))))
zbench_lp =round( norm.ppf(z3_4),4)
#print(f' yeld = {z1_2*100:.8f}%')

#formula para calculo nivel sigma com deslocamento zbench+1.5=nivel sigma
yeldcp = round(norm.cdf(zbench_lp),10)
dpmocp = round((1-yeldcp)*10**6,3)
nsigmacp = round(zbench_lp+1.5, 4)

#calculo do processo real
x_process_cp = np.arange((x_bar-(6*Dspw)), (x_bar+(6*Dspw)), 0.001)
y_process_cp = norm.pdf(x_process_cp, x_bar,Dspw)

x_proces = np.arange((x_bar - (6*DspO)), (x_bar + (6*DspO)), 0.001)
y_proces = norm.pdf(x_proces,x_bar,DspO)
#calculo para linha da media no grafico para cada distruibuição
ygmd = max(y_proces) #processo





#calculo para coletas

val=dt.now().minute
#val=time.strftime('%H:%M:%S', time.localtime())

#data atual
d_atual= dt.now()
#inicio da coleta
#inicio_coleta= df['UtcCreatedAt'].min()
#inicio_coleta=inicio_coleta.to_pydatetime
inicio_coleta= sql_query['UtcCreatedAt'].min()

#inicio_col=df['UtcCreatedAt'].values[0]
#inicio_col=inicio_col.to_pydatetime()
#retorna numero inteiro representando a duração da frequencia dos subgrupos
freq=sql_variavel['SampleFrequency'].values[0]
freq=int(freq)
numero_frequencia=int(freq*nun_subgrupos)

#função transforma inteiro em minutos-pode ser segundos alterando calculo numero_frequencia
def convert(freq_acumulada): 
    return (datetime.timedelta(minutes = freq_acumulada)) 

freq_acumulada=convert(numero_frequencia)

freq_minutos=convert(freq)


#reorna a ultima coleta realizada
ultima_coleta=inicio_coleta+freq_acumulada

#diferença de tempo entre a ultima coleta e dia e hora atual
gap=d_atual-ultima_coleta
#retorna em minutos o tempo
gap1=gap.total_seconds()/60
#transforma em inteiro
gap1=int(gap1)
#retorna o numero de subgrupos referente ao gap
gap_subgroup=int(gap1/freq)
#retorna o tempo em numero inteiro do gap
periodo_gap=gap_subgroup*freq
#periodo_gap=gap_subgroup*5
#converte para de inteiro para periodo de tempo
periodo_gap_mnt=convert(periodo_gap)



#retorna a data hora atual coleta
coleta_atual=ultima_coleta+periodo_gap_mnt
#Proxima coleta adiciona a frequencia em tempo a atual coleta
proxima_coleta=coleta_atual+freq_minutos



#teste grafico frequencia
#current_time = datetime.datetime.now()
#val_min=dt.now().minute-freq_minutos

#val_1=proxima_coleta-coleta_atual
#val_1=val_1/td(minutes=1)


#apresenta no grafico minutos e segundos
val_1=proxima_coleta-dt.now()
val_1=int(val_1/td(minutes=1))



val=dt.now().minute

val_hr_at=dt.now().time()

#val_1=dt.now().time()-(dt.now().time()-freq_minutos)

#hora_atual=dt.datetime.now()

#freqteste=(355783/5)
#freqteste=int(freqteste)




#retornar numero de coletas por ponto de medicao
subgrupo_mx=df_pto_ind['SubGroup'].max()
#retorna numero de pontos de medicao
ptos_medicao=df_pto_ind[df_pto_ind['SubGroup']==subgrupo_mx]['MeasuringPoint'].count()



df_r=df.pivot_table(values=['Value'], index=['SubGroup','MeasuringPoint']
                      ,aggfunc=[len])
x1=df['SubGroup'].iloc[-1]
x1=int(tam_subgr/ptos_medicao)
#print(df[['MeasuringPoint','SubGroup','Value']].tail(x1))     
     
#print(df_r.loc[(7,4)])
#print(df_r.loc[(7,0):(7,4)])
#print(df_r.iloc[-1:])


mx_valor=sql_variavel['MaxInput']
mx_valor=float(mx_valor)
mn_valor=sql_variavel['MinInput']
mn_valor=float(mn_valor)

#print(sql_query.Value >=mx_valor)

#for i in range(len(sql_query['Value'])):
#    if sql_query[(sql_query['Value']>=mx_valor) & (sql_query['SubGroup']== subgrupo_mx)] :
#        mx_permitido= sql_query['Value']
        
#    else:
#xxc=(sql_query[(sql_query['Value']>=mx_valor) & (sql_query['SubGroup']== 1)] )    
    
#mx_permitido=sql_query.apply(lambda x: x['Value'] if x['Valure']>=mx_valor
#                        and x['SubGroup']== subgrupo_mx, axis=1)



def vlr_extr():
    for i in range(len(sql_out_sample["Value"])):
        
      if sql_out_sample.loc[i,'Value']  > 0:
         
          vlr_extr=sql_out_sample['Value'].iloc[-1]
          #vlr_extr=sql_out_sample.loc[i,'Value']
          return vlr_extr
      else:
           #vlr_extr=0
           return 0
    
vlr_extr=vlr_extr()



vlr_extr1=sql_out_sample['Value'].iloc[-1]
#dt_hora_out=sql_out_sample['UtcCreatedAt'].iloc[-1]

#ERRO na função alterar para indice >0

#def dt_hora_out():
#    for i in range(len(sql_out_sample["UtcCreatedAt"])):
        
#      if sql_out_sample['UtcCreatedAt'][i]  > 0:
          
#          vlr_extr=sql_out_sample['UtcCreatedAt'].iloc[-1]
#      else:
#           vlr_extr= dt.now()
           
           
#funcao com alteração para o value >0
def dt_hora_out():
    for i in range(len(sql_out_sample["Value"])):
        
      if sql_out_sample['Value'][i]  > 0:
          
          dt_out=sql_out_sample['UtcCreatedAt'].iloc[-1]
          return dt_out
      else:
           #dt_outr= dt.now()   
           return dt.now()
           
           
dt_hora_out=dt_hora_out()

#graficos


#Graficos
#grafico XR media
#tamanho figsize=(18,8
plt.style.use('dark_background')
fig,ax =plt.subplots(figsize=(18,8))
ax.plot(df_carta['medSub'],color='yellow', marker='o')
ax.axhline(x_bar, color='green', linestyle='dashed')
ax.axhline(Lcsx, color='red', linestyle='--')
ax.axhline(Licx,color='red', linestyle='dashed')
ax.set_xlabel('Subgrupo')
ax.set_ylabel('Media')
ax.set_title('Carta XRS Média')
#plt.show



#grafico processo real 

fig1,ax = plt.subplots(figsize=(14,6))
plt.plot(x_proces,y_proces, color='red')
plt.plot(x_process_cp, y_process_cp, color= 'green', linestyle = '--')
plt.vlines(x_bar, ymin = 0 , ymax = ygmd, color='red', linestyle='-', linewidth=1,
            label= f'Media {x_bar}') 
plt.axvline(Lie, color='k', linestyle='-', linewidth=1, label = f'LIE {Lie:.2f}')
plt.axvline(Lse, color='k', linestyle='-', linewidth=1, 
            label = 'LSE {:.2f}'.format(Lse))
plt.axvline(Nominal, color='yellow', linestyle='--', linewidth=1, 
            label = 'Meta {:.2f}'.format(Nominal))

#plt.yticks([]) #desativa os valores do eixo y
plt.yticks() #desativa os valores do eixo y
plt.legend()
plt.xlabel('Valores')
plt.ylabel('Probabilidade')
plt.title('Distribuição do Processo Real')


#grafico controle frequencia
fig3 = go.Figure(go.Indicator(
    mode = "gauge+number",
    #value=freq,
    value =val_1,
    title = {'text': " Frequencia para coleta"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {'axis': {'range': [None, 5]}},
    #gauge = {'axis': {'range': [None, 60]}}
))
fig3.update_layout(width=400, height=250, margin_b=30,margin_l=30)

#fig3.show()

#nao usado
#Tabela header
fig4 = go.Figure(data=[go.Table(
    header=dict(values=list(sql_header.columns),
                fill_color='black',
                align='center',
                font=dict(color='white', size=15)
                ),
    cells=dict(values=sql_header.transpose().values.tolist(),
               fill_color='black',
               font=dict(color='white', size=15),
               align= 'center'))
])
fig4.update_layout(width=900, height=900)
#fig4.show





#criar  array de ptos de medicao

pto=np.arange(start=1, stop=ptos_medicao+1)
print(pto.dtype)

#lista ptos medicao
pto_range=list(range(1,ptos_medicao+1))
               
               
for n in pto_range:
    print(n)

xx=df.loc[df['MeasuringPoint']==5,'Value']









#grafico boxplot por ponto de medições
pto1=go.Box(y=df.loc[df['MeasuringPoint']==1,'Value'],
            name='Pto1')
pto2=go.Box(y=df.loc[df['MeasuringPoint']==2,'Value'],
            name='Pto2')
pto3=go.Box(y=df.loc[df['MeasuringPoint']==3,'Value'],
            name='Pto3')
pto4=go.Box(y=df.loc[df['MeasuringPoint']==4,'Value'],
            name='Pto4')
pto5=go.Box(y=df.loc[df['MeasuringPoint']==5,'Value'],
            name='Pto5')

data=[pto1,pto2,pto3,pto4,pto5]
layout=go.Layout(title='Dispersão por Ponto de Medicao',
                 xaxis={'title':' Pontos de medicao'},
                 yaxis={'title':'Valores'})
fig5=go.Figure(data=data,layout=layout)

fig5.add_hline(y=x_bar,line_width=3, line_dash="dash", line_color="red",
               annotation_text='Media')  
fig5.add_hline(y=Lse,line_width=2, line_dash="dash", line_color="green",
               annotation_text='LSE', annotation_position="top left", annotation_font_size=15)
fig5.add_hline(y=Lie,line_width=2, line_dash="dash", line_color="green",
               annotation_text='LIE', annotation_position="top left",annotation_font_size=15)
fig5.update_traces(boxmean=True, showlegend=False) 
fig5.update_layout(width=800, height=700)    
#fig5.show

#py.iplot(fig5)
import matplotlib.pyplot as plt
import seaborn as sns

#df_pto_ind.boxplot(column=['MeasuringPoint'], by='Value',ax=ax)
sns.boxplot(x='MeasuringPoint', y='Value', data=df)





#grafico não usado
import plotly.express as px
fig6=px.box(df,x='MeasuringPoint',y='Value')
fig6.add_hline(y=x_bar,line_width=3, line_dash="dash", line_color="red",
               annotation_text='Media')       
    
#fig6.show
fig6.update_layout(width=800, height=800)





#dashboard

st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title="Controle de Coleta",  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)
img = Image.open(r"C:/Users/Duecon\Documents/Python_Scripts/scheduler/python/egalogo_new.png")
#im = Image.open(r"C:/Users/Duecon\Documents/Python_Scripts/Spc_Dashboard/egalogo1.png")
#st.image(im,caption='',use_column_width=False)
#row0_spacer1, .1  row0_spacer3 .1   row0_spacer2
row0_1, row0_2, row0_3,  = st.columns(( 1.1, 4.0, 1.4 ))
with row0_1:
    st.title('SPC CONTROLE ')
with row0_2:
     st.table(sql_header)
 #   st.plotly_chart(fig4)
with row0_3:
    
    st.write('*Valor descartado alem dos limites Permitidos* ',vlr_extr) 
    st.write('*SubGgrupo*', subgrupo_mx+1)
    st.write('*Data e Hora da Coleta* ', dt_hora_out)

#st.title ('SPC CONTROLE COLETA')

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)   


col1, col2, col3 = st.columns((1.2,2,1))

with col1:
    #img = Image.open(r"C:/Users/Duecon\Documents/Python_Scripts/Spc_Dashboard/logo_ega.png")
    img = Image.open(r"C:/Users/Duecon\Documents/Python_Scripts/scheduler/python/egalogo_new.png")
    st.image(img,caption='',use_column_width=False)
   # st.write('',sql_header['Maquina'])
    st.plotly_chart(fig3)
    st.write('Inicio da Coleta ', inicio_coleta)
    st.write('Coleta Atual ', coleta_atual)
    st.write('Proxima Coleta ', proxima_coleta)
    st.write('Frequencia ', freq_minutos)
    st.write('Tamanho do Subgrupo   ',subgrupo_n)
   # st.subheader('PONTOS DE MEDICAO')
    
    
    #st.write('Ponto de medicao/ Coletas ' , df_r.loc[(7,0):(7,4)].astype(int))
    #st.dataframe(df_r.loc[(7,0):(7,4)].astype(int))
    #st.write('Ponto de medicao/ Coletas ', df_r.iloc[-1:].astype(int))
    st.write('Ponto de medicao/ Coletas',df[['MeasuringPoint','SubGroup','Value']].tail(x1))

with col2:
   
    st.write('Grafico Carta Media XRS')
    st.pyplot(fig)
    ##st.write('Dispersão por Pontos de Medicao')
    st.plotly_chart(fig5)
    #st.write('Histograma do Processo')
    #st.pyplot(fig1)
    #st.plotly_chart(fig4)
    #st.plotly_chart(fig6)
    #gb= GridOptionsBuilder.from_dataframe(sql_header)
    #AgGrid(sql_header, 
           #ob.build(),
            # theme='dark',
    #fit_columns_on_grid_load=True,        
    #gridOptions=gb,
   # height=500,
   # allow_unsafe_jscode=True,
   # enable_enterprise_modules=True,)
    
with col3:
    
    st.text('')   
    st.text('') 
    st.text('')   
      
    st.write('Limite Superior de Controle LSCx = ',Lcsx)
    st.write('Linite Centtral de Controle Lcx = ', x_bar)
    st.write('Linite Inferior de Controle LICx =', Licx)
    st.write('Média do Processo = ',x_bar)
    st.write(' Total de amostras = ', statis['count'])
    st.write('Limite Superior de Especificação', Lse)
    st.write('Valor maximo  coletado = ', maximo_entrada)
    st.write('Limite de Especificação Inferior', Lie)
    st.write('Valor minimo coletado =', minima_entrada)
    
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')  
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')   
    st.text('')   
   
    st.text('')   
    st.text('')   
   
     
   
    
    
   
    st.write(' CPK = ', Cpk)
    st.write(' PPK = ',Ppk)
    st.write(' Desvio Padrão Longo Curto =',Dspw)
    st.write('Desvio Padrão Lono Prazo = ',DspO)
    #st.write('Limite Superior de Especificação', Lse)
    st.write('Nominal',Nominal)
    #st.write('Limite de Especificação Inferior', Lie)
    
    #st.write(' PPM LSE Curto Prazo = ',ppm_lse)
    #st.write('PPM LIE Curto Prazo =',ppm_lie)
    #st.write(' PPM TOTAL Curto Prazo = ',ppm_total)
   
    #st.write(' PPM LIE Longo Prazo = ',ppm_lie)
    #st.write(' PPM LSE Longo Prazo = ',ppm_lse)
    #st.write(' PPM TOTAL Longo Prazo = ',ppm_total)
    
    