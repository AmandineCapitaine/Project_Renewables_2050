import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
from data_formatting import Formater_17ans, Import_data, Profils_vre_datetime
from scipy.stats import kstest
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids

colors = ["palevioletred", "mediumaquamarine", "cornflowerblue", "royalblue", "slategray", "mediumseagreen", "salmon", "yellow", "khaki", "0.2"]

ans = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]


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

def Array_mean_std(day_mean_18, day_std_18):
    df = np.zeros((18*365,2))
    for i in range(18*365):
        df[i,0] = day_mean_18[0].iloc[i]
        df[i,1] = day_std_18[0].iloc[i]
    return df

def New_Array_mean_std(daily_mean, daily_std):
    daily_mean[:3] = daily_mean.mean() #pour remplacer les NAN
    daily_std[:3] = daily_std.std()
    new_mean_std = np.zeros((365,2))
    new_mean_std[:,0] = daily_mean
    new_mean_std[:,1] = daily_std
    return new_mean_std

def distanceMetric(x, y, alpha=1.5):
    return np.sqrt((x[0]-y[0])**2 + (x[1]*alpha-y[1]*alpha)**2)

def Clustering_days(df, n_clusters, metric='euclidean', verbose_plot=False, verbose_return=True):

    # — — — — — — -Fit KMedoids clustering — — — — — — — — — — — -
    KMobj = KMedoids(n_clusters=n_clusters, metric=metric).fit(df)

    # — — — — — — -Assigning Cluster Labels — — — — — — — — — — — -
    labels = KMobj.labels_
    
    # — — — — — — -Extracting Unique Labels — — — — — — — — — — — -
    unq_lab = set(labels)

    # — — — — — — -Setting Up Color Codes — — — — — — — — — — — -
    colors_plot = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unq_lab))]

    for k, col in zip(unq_lab, colors_plot):
        class_member_mask = labels == k

         # — — — — — — -Setting datapoint Feature X and Feature Y — — — — — — — — — — — -
        xy = df[class_member_mask]
        
        if verbose_plot:
             # — — — — — — -Plotting Feature X and Feature Y for each cluster labels — — — — — — — — — — — -
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = tuple(col), markeredgecolor = 'white', markersize=10)
    
    if verbose_plot:
        # — — — — — — -Annotate Centroids — — — — — — — — — — — -
        plt.plot(KMobj.cluster_centers_[:, 0], KMobj.cluster_centers_[:, 1], 'o', markerfacecolor='orange', markeredgecolor='k',
                 markersize=10)

        # — — — — — — -Add title to the plot — — — — — — — — — — — -
        plt.title("KMedoids clustering (centers in orange)", fontsize = 14)
        plt.show()
    
    if verbose_return:
        return KMobj.cluster_centers_, labels
    
    
def Index_centers(n_clusters, centers, labels, array):    
    index_centers = np.zeros((n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(len(np.where(array == centers[i]))):
            if np.where(array == centers[i])[j][0] == np.where(array == centers[i])[j][1]:
                index_centers[i] = np.where(array == centers[i])[j][0]
    return index_centers

def Typical_days(n_clusters, centers, labels, array, p_18): 
    typical_days = np.zeros((24,len(centers)))
    index_centers = Index_centers(n_clusters, centers, labels, array)
    for ind in range(len(index_centers)):
        typical_days[:,ind] = p_18[index_centers[ind]*24:(index_centers[ind]+1)*24][0]
    return typical_days

def Typical_year(n_clusters, centers, labels, array, p_18, an): 
    typical_days = Typical_days(n_clusters, centers, labels, array, p_18)
    typical_year = np.zeros(8760)

    for i in range(365):
        typical_year[i*24:(i+1)*24] = typical_days[:,labels[i+an*365]]
    return typical_year


def Labels_new_year(array, new_array, n_clusters, metric='euclidean', verbose_plot=True):
    array2 = np.copy(array)
    new_array2 = np.copy(new_array)
    
    #on fit le modèle sur les données existantes pour avoir des typical days existants
    KMobj = KMedoids(n_clusters=n_clusters, metric=metric).fit(array2) 
    
    #on prédit la classe des nouveaux jours ie des productions moyennes journalières simulées
    new_labels = KMobj.predict(new_array2)
    unq_lab = set(new_labels)
    
    # — — — — — — -Setting Up Color Codes — — — — — — — — — — — -
    colors_plot = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unq_lab))]

    for k, col in zip(unq_lab, colors_plot):
        class_member_mask = new_labels == k

         # — — — — — — -Setting datapoint Feature X and Feature Y — — — — — — — — — — — -
        xy = new_array2[class_member_mask]
        
        if verbose_plot:
             # — — — — — — -Plotting Feature X and Feature Y for each cluster labels — — — — — — — — — — — -
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = tuple(col), markeredgecolor = 'white', markersize=10)
            # — — — — — — -Annotate Centroids — — — — — — — — — — — -
            plt.plot(KMobj.cluster_centers_[k, 0], KMobj.cluster_centers_[k, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
                     markersize=10)
    
    if verbose_plot:
        # — — — — — — -Add title to the plot — — — — — — — — — — — -
        plt.title("KMedoids clustering", fontsize = 14)
        plt.show()
    
    return KMobj.cluster_centers_, KMobj.labels_, new_labels

def Typical_new_year(n_clusters, centers, labels, new_labels, array, p_18): 
    typical_days = Typical_days(n_clusters, centers, labels, array, p_18)
    typical_year = np.zeros(8760)

    for i in range(365):
        typical_year[i*24:(i+1)*24] = typical_days[:,new_labels[i]]  #typical days sur la nouvelle année
    return typical_year
    
def Test_serie(profils, new_profil, time='1970-01-01'):
    signal = profils['2006']
    profils = profils.index = pd.to_datetime(profils.index, unit='h', origin=pd.Timestamp(time))
                     
    #Statistiques des séries
    print("Moyenne annuelle de production en 2006 ", signal.mean())
    print("Moyenne annuelle de production du nouveau profil ", new_profil.mean())
    print("Ecart-type annuelle de production en 2006 ", signal.std())
    print("Ecart-type annuelle de production du nouveau profil ", new_profil.std())
       
    #Affichage moyennes mensuelles           
    monthly_mean_18ans = profils.resample('M').mean()
    monthly_mean_new = new_profil.resample('M').mean()

    plt.plot(monthly_mean_18ans, c=colors[0])
    plt.plot(monthly_mean_18ans['2006'], label="2006", c=colors[5])
    plt.plot(monthly_mean_new, label="New", c=colors[-1])
    plt.title("Comparaison des profils annuels avec moyenne mensuelle")
    plt.xlabel("Année")
    plt.ylabel("Production solaire")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()                
    
    #Histogramme
    bins= 10
    plt.hist(profils, bins=bins, ls='dashed', alpha = 0.5, lw=3)
    plt.hist(new_profil[0], bins=bins, ls='dotted', alpha = 0.5, lw=3, color= 'r', label='new')
    plt.hist(profils['2006'], bins=bins, ls='dotted', alpha = 0.2, lw=3, color= 'b', label='2006')
    plt.title('Histogramme des 2 profils')
    plt.legend()
    plt.show()
    
    #Plot ACF
    nb_lags = 100000
    acf_2006 = acf(signal, nlags=nb_lags, fft=False)
    acf_new = acf(new_profil[0], nlags=nb_lags, fft=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=acf_new, name='ACF nouveau profil'))
    fig.add_trace(go.Scatter(y=acf_2006, name='ACF 2006'))
    fig.update_layout(title="Comparaison des fonction d'autocorrélation des  profils solaires",
                        yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
    
    #Kolmogorov-Smirnov test
    print('Kolmogorov-Smirnov test ', kstest(signal, new_profil[0]))
    
