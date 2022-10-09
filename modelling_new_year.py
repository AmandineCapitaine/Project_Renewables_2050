import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
from data_formatting import Formater_17ans 
from scipy.stats import kstest
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go

ans = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]

def Profils_vre_datetime():
    profils_off, profils_on, profils_pv = Formater_17ans()
    profils_off.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_off.index = pd.to_datetime(profils_off.index, unit='h')
    profils_on.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_on.index = pd.to_datetime(profils_on.index, unit='h')
    profils_pv.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_pv.index = pd.to_datetime(profils_pv.index, unit='h')
    
    return profils_off, profils_on, profils_pv

def Mean_std_profil_18(profil, verbose_mean=True):
    #Création d'un DataFrame avec les 18 années de production 
    df = np.zeros(18*8760)

    for i in range(18):
        s = profil[ans[i]]
        df[8760*i:8760*(i+1)] = s

    df = pd.DataFrame(df)
    time = '2000-01-31'
    df.index = pd.to_datetime(df.index, unit='h', origin=pd.Timestamp(time))

    if verbose_mean:
        df_mean = df.resample('D').mean()  #moyennes de production journalières sur 18 ans
    else:
        df_mean = df.resample('D').std()  #ecart-types de production journalières sur 18 ans

    return df_mean, df 


from statsmodels.tsa.seasonal import seasonal_decompose

def Decomposition(df, verbose_plot=True):
    decomp = seasonal_decompose(df)

    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid
    
    if verbose_plot:

        plt.figure(figsize=(15,11))
        plt.subplot(411)
        plt.plot(df, 'cornflowerblue', label='original')
        plt.legend()
        plt.subplot(412)
        plt.plot(trend, 'salmon', label='trend')
        plt.legend()
        plt.subplot(413)
        plt.plot(seasonal[:100], 'palevioletred', label='seasonal')
        plt.legend()
        plt.subplot(414)
        plt.plot(residual, 'yellow', label='residuals')
        plt.legend()
        plt.suptitle('Profile decomposition', y=0.9)
        plt.show()

    return trend, seasonal, residual


def Simulate_Noise(trend, seasonal, residual, df, verbose_plot=True):
    
    mean = round(np.mean(residual),2)
    std =  round(np.std(residual),2)  
    num_samples = 365*18
    samples = np.random.normal(mean, std, size=num_samples)
    
    new = trend + seasonal + samples
    
    min_vre = df.min()[0]
    for i in range(len(new)):
        if new[i]<min_vre:
            new[i] = min_vre
    
    if verbose_plot:
        print("ecart type du bruit : ", np.std(residual))
        print("moyenne du bruit : ", np.mean(residual))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=samples, name='sample'))
        fig.add_trace(go.Scatter(y=residual, name='residual'))
        fig.update_layout(title="Comparaison du bruit simulé et du bruit de la série décomposée", yaxis_zeroline=False, xaxis_zeroline=False)
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=new, name='sample'))
        fig.add_trace(go.Scatter(y=df[0], name='residual'))
        fig.update_layout(title="Comparaison du vrai profil et du profil reconstruit à partir d'un bruit simulé", yaxis_zeroline=False, xaxis_zeroline=False)
        fig.show()
    
    return new






