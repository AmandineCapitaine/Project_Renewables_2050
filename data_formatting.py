# Imports 
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import csv
import os


def Import_data():
    vre2017 = pd.read_csv("inputs/vre_profiles2017.csv", header=None)
    vre2016 = pd.read_csv("inputs/vre_profiles2016.csv", header=None)
    vre2015 = pd.read_csv("inputs/vre_profiles2015.csv", header=None)
    vre2014 = pd.read_csv("inputs/vre_profiles2014.csv", header=None)
    vre2013 = pd.read_csv("inputs/vre_profiles2013.csv", header=None)
    vre2012 = pd.read_csv("inputs/vre_profiles2012.csv", header=None)
    vre2011 = pd.read_csv("inputs/vre_profiles2011.csv", header=None)
    vre2010 = pd.read_csv("inputs/vre_profiles2010.csv", header=None)
    vre2009 = pd.read_csv("inputs/vre_profiles2009.csv", header=None)
    vre2008 = pd.read_csv("inputs/vre_profiles2008.csv", header=None)
    vre2007 = pd.read_csv("inputs/vre_profiles2007.csv", header=None)
    vre2006 = pd.read_csv("inputs/vre_profiles2006.csv", header=None)
    vre2005 = pd.read_csv("inputs/vre_profiles2005.csv", header=None)
    vre2004 = pd.read_csv("inputs/vre_profiles2004.csv", header=None)
    vre2003 = pd.read_csv("inputs/vre_profiles2003.csv", header=None)
    vre2002 = pd.read_csv("inputs/vre_profiles2002.csv", header=None)
    vre2001 = pd.read_csv("inputs/vre_profiles2001.csv", header=None)
    vre2000 = pd.read_csv("inputs/vre_profiles2000.csv", header=None)
    return vre2000, vre2001, vre2002, vre2003, vre2004, vre2005, vre2006, vre2007, vre2008, vre2009, vre2010, vre2011, vre2012, vre2013,vre2014,vre2015,vre2016,vre2017

def Formater(profil, time):
    
    #on découpe le profil entre les différentes technologies génératrices
    profil.columns = ["vre", "heure", "prod"]
    profil_offshore = profil.head(8760)
    profil_onshore = profil.truncate(before = 8760, after = 17519)
    profil_pv = profil.tail(8760)
    
    profil_offshore.columns = ["vre", "heure", "prod"]
    profil_offshore['heure'] = pd.to_datetime(profil_offshore['heure'], unit='h', origin=pd.Timestamp(time))
    profil_offshore = profil_offshore.set_index('heure')
    
    profil_onshore.columns = ["vre", "heure", "prod"]
    profil_onshore['heure'] = pd.to_datetime(profil_onshore['heure'], unit='h', origin=pd.Timestamp(time))
    profil_onshore = profil_onshore.set_index('heure')
    
    profil_pv.columns = ["vre", "heure", "prod"]
    profil_pv['heure'] = pd.to_datetime(profil_pv['heure'], unit='h', origin=pd.Timestamp(time))
    profil_pv = profil_pv.set_index('heure')
    
    profils = pd.DataFrame(profil_offshore["prod"])
    profils.columns = ['offshore']
    profils.insert(1, "onshore", profil_onshore['prod'])
    profils.insert(2, "pv", profil_pv['prod'])

    return profils

def Decouper(profil):
    #on découpe le profil entre les différentes technologies génératrices
    profil.columns = ["vre", "heure", "prod"]
    profil_offshore = profil.head(8760)
    profil_onshore = profil.truncate(before = 8760, after = 17519)
    profil_pv = profil.tail(8760)
    return profil_offshore, profil_onshore, profil_pv

def Formater_5ans(profil1, profil2, profil3, profil4, profil5):
    profil_offshore1, profil_onshore1, profil_pv1 = Decouper(profil1)
    profil_offshore2, profil_onshore2, profil_pv2 = Decouper(profil2)
    profil_offshore3, profil_onshore3, profil_pv3 = Decouper(profil3)
    profil_offshore4, profil_onshore4, profil_pv4 = Decouper(profil4)
    profil_offshore5, profil_onshore5, profil_pv5 = Decouper(profil5)
    
    profils_off = pd.DataFrame([profil_offshore1["prod"], profil_offshore2["prod"], profil_offshore3["prod"], profil_offshore4["prod"], profil_offshore5["prod"]])
    profils_on = pd.DataFrame([profil_onshore1["prod"], profil_onshore2["prod"], profil_onshore3["prod"], profil_onshore4["prod"], profil_onshore5["prod"]])
    profils_pv = pd.DataFrame([profil_pv1["prod"], profil_pv2["prod"], profil_pv3["prod"], profil_pv4["prod"], profil_pv5["prod"]])

    return profils_off.T, profils_on.T, profils_pv.T

def Formater_17ans():
    vre2000, vre2001, vre2002, vre2003, vre2004, vre2005, vre2006, vre2007, vre2008, vre2009, vre2010, vre2011, vre2012,vre2013,vre2014,vre2015,vre2016,vre2017 = Import_data()

    profil_offshore0, profil_onshore0, profil_pv0 = Decouper(vre2000)
    profil_offshore1, profil_onshore1, profil_pv1 = Decouper(vre2001)
    profil_offshore2, profil_onshore2, profil_pv2 = Decouper(vre2002)
    profil_offshore3, profil_onshore3, profil_pv3 = Decouper(vre2003)
    profil_offshore4, profil_onshore4, profil_pv4 = Decouper(vre2004)
    profil_offshore5, profil_onshore5, profil_pv5 = Decouper(vre2005)
    profil_offshore6, profil_onshore6, profil_pv6 = Decouper(vre2006)
    profil_offshore7, profil_onshore7, profil_pv7 = Decouper(vre2007)
    profil_offshore8, profil_onshore8, profil_pv8 = Decouper(vre2008)
    profil_offshore9, profil_onshore9, profil_pv9 = Decouper(vre2009)
    profil_offshore10, profil_onshore10, profil_pv10 = Decouper(vre2010)
    profil_offshore11, profil_onshore11, profil_pv11 = Decouper(vre2011)
    profil_offshore12, profil_onshore12, profil_pv12 = Decouper(vre2012)
    profil_offshore13, profil_onshore13, profil_pv13 = Decouper(vre2013)
    profil_offshore14, profil_onshore14, profil_pv14 = Decouper(vre2014)
    profil_offshore15, profil_onshore15, profil_pv15 = Decouper(vre2015)
    profil_offshore16, profil_onshore16, profil_pv16 = Decouper(vre2016)
    profil_offshore17, profil_onshore17, profil_pv17 = Decouper(vre2017)
    
    
    profils_off = pd.DataFrame([profil_offshore0["prod"], profil_offshore1["prod"], profil_offshore2["prod"], 
                                profil_offshore3["prod"], profil_offshore4["prod"], profil_offshore5["prod"], 
                                profil_offshore6["prod"], profil_offshore7["prod"], profil_offshore8["prod"],
                                profil_offshore9["prod"], profil_offshore10["prod"], profil_offshore11["prod"], 
                                profil_offshore12["prod"], profil_offshore13["prod"], profil_offshore14["prod"], 
                                profil_offshore15["prod"], profil_offshore16["prod"], profil_offshore17["prod"]])
    
    profils_on = pd.DataFrame([profil_onshore0["prod"], profil_onshore1["prod"], profil_onshore2["prod"], 
                               profil_onshore3["prod"], profil_onshore4["prod"], profil_onshore5["prod"], 
                               profil_onshore6["prod"], profil_onshore7["prod"], profil_onshore8["prod"], 
                               profil_onshore9["prod"], profil_onshore10["prod"], profil_onshore11["prod"], 
                               profil_onshore12["prod"], profil_onshore13["prod"], profil_onshore14["prod"], 
                               profil_onshore15["prod"], profil_onshore16["prod"], profil_onshore17["prod"]])
    
    
    profils_pv = pd.DataFrame([profil_pv0["prod"], profil_pv1["prod"], profil_pv2["prod"], profil_pv3["prod"], 
                               profil_pv4["prod"], profil_pv5["prod"], profil_pv6["prod"], profil_pv7["prod"],
                               profil_pv8["prod"], profil_pv9["prod"], profil_pv10["prod"], profil_pv11["prod"], 
                               profil_pv12["prod"], profil_pv13["prod"], profil_pv14["prod"], profil_pv15["prod"],
                               profil_pv16["prod"], profil_pv17["prod"]])

    return profils_off.T, profils_on.T, profils_pv.T


def Formater_NP(profil, time):
    
    #on découpe le profil entre les différentes technologies génératrices
    profil.columns = ["vre", "heure", "prod"]
    profil_offshore = profil.head(8760)
    profil_onshore = profil.truncate(before = 8760, after = 17519)
    profil_pv = profil.tail(8760)
    
    profil_offshore.columns = ["vre", "heure", "prod"]
    
    profil_onshore.columns = ["vre", "heure", "prod"]
    
    profil_pv.columns = ["vre", "heure", "prod"]
    
    profils = pd.DataFrame(profil_offshore["prod"])
    profils.columns = ['offshore']
    profils.insert(1, "onshore", profil_onshore['prod'])
    profils.insert(2, "pv", profil_pv['prod'])

    return profils

def Profils_vre_datetime():
    profils_off, profils_on, profils_pv = Formater_17ans()
    profils_off.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_off.index = pd.to_datetime(profils_off.index, unit='h')
    profils_on.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_on.index = pd.to_datetime(profils_on.index, unit='h')
    profils_pv.columns = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013","2014", "2015", "2016", "2017"]
    profils_pv.index = pd.to_datetime(profils_pv.index, unit='h')
    
    return profils_off, profils_on, profils_pv
