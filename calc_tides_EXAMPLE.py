
#%% import modules
import os
import pandas as pd
import numpy as np
import utide
from datetime import datetime, timedelta, timezone
import pytz
import matplotlib as mpl
mpl.rcParams["date.epoch"] = '0000-12-31T00:00:00'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import signal 
import scipy.fftpack
import csv
#%% imoprt tide data
filepath = r'\\drive.irds.uwa.edu.au\SEE-PNP-001\HANSEN_UWA-UNSW_Linkage\Data\Waves_WaterLevels\DOT_water_levels\ManOceanMarina_Tides'
files = os.listdir(filepath)
for i, file in enumerate(files):
    if i==0:
        dot = pd.read_csv(os.path.join(filepath,file),names=['H','date'],skiprows=18,na_values=-9999, encoding = 'ISO-8859-1')
    else:      
        dot = dot.append(pd.read_csv(os.path.join(filepath,file),names=['H','date'],skiprows=18,na_values=-9999, encoding = 'ISO-8859-1'))

# for i, file in enumerate(files):
#     if i==14: #JUST TRYING 2020 DATA
#         dot = pd.read_csv(os.path.join(filepath,file),names=['H','date'],skiprows=18,na_values=-9999, encoding = 'ISO-8859-1')

dot = dot.reset_index().drop(columns='index')


#convert to AHD 
dot['hAHD'] = (dot['H']/100) - 0.54

#create local time
dot['date'] = dot['date'].str.strip('/')
dot['date'] = dot['date'].str.strip('/ ')
dot['timelocal'] = pd.to_datetime(dot['date'],format='%Y%m%d.%H%M')


dot= dot.drop(columns = ['date','H'])
dot = dot.set_index('timelocal')
dot = dot.tz_localize(timezone(timedelta(hours=8)))
dot = dot.tz_convert('UTC')
dot = dot.reset_index().rename(columns={'timelocal':'timeutc'})

#create floating number with base date 0000-12-31 so that 1 = np.datetime64(1,1,1)
# time_ordinal = []
# for i,row in dot.iterrows():  
#     time_ordinal.append(abs(mdates.date2num(np.datetime64('0000-12-31')))+mdates.date2num(row['timeutc']))

# dot['time_ordinal'] = time_ordinal

#%% use Ryan's approach 
   
# Tidal variability (eta_tide) - ignore seasonal constituents 
constituents=['K1', 'O1', 'P1', 'M2', 'S2', 'S1', 'Q1', 'N2', 'K2',
            'J1', 'NO1', 'MU2', 'OO1', 'SIG1', 'RHO1','M4',
            '2Q1', 'MSM', '2N2', 'SK3', 'L2', 'MS4', 'T2', 'PHI1', 'PI1', 'M3',
            'SO1', 'NU2', 'EPS2', 'THE1', 'LDA2', 'MN4', 'CHI1', 'ALP1', 'MK3',
            'TAU1', 'MK4', 'M6', 'MO3', 'PSI1', '2MN6', 'M8', '2MS6', 'BET1',
            'S4', 'H1', 'ETA2', 'SO3', 'UPS1', 'H2', 'OQ2', 'SN4', 'SK4',
            'MKS2', '2SK5', '2MK6', 'MSN2', 'R2', '2MK5', '3MK7', 'GAM2',
            'MSK6', '2SM6']

#!! Important - time needs to be in epoch days since Jan 1 0001
# coef_tide = utide.solve(dot.time_ordinal, dot.hAHD, lat = -32.522347, epoch = 'python',method='ols', conf_int='MC',constit=constituents,trend=False)
coef_tide = utide.solve(dot.timeutc, dot.hAHD, lat = -32.522347,
                        method='ols', conf_int='MC',constit='auto')

tide = utide.reconstruct(dot.timeutc, coef_tide)

dot['tide']=tide.h

# #write csv
# csv_path = r'D:\UNSW_Project\Tide_prediction\data'
# fname = 'Mandurah_Marina_combined.csv'

# dot.to_csv(os.path.join(csv_path,fname), sep=',',index=False, date_format='%Y-%m-%d %H:%M:%S%z')
#%% plot the rawdata and visualise
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharey=True, sharex=True, figsize=(14, 7))
ax0.grid(True)
ax0.plot(dot.timeutc, dot.hAHD, alpha=0.8, label=u'Observations')
ax0.legend(numpoints=1, loc='lower right')
# ax0.xlabel('Frequency [Hz]')
ax0.set_ylabel('[m]')

ax1.plot(dot.timeutc, dot.tide, alpha=0.5, label=u'Prediction')
ax1.legend(numpoints=1, loc='lower right')
ax1.set_ylabel('[m]')
ax1.grid('on')
ax2.plot(dot.timeutc, dot.hAHD-dot.tide, alpha=0.5, label=u'Residue')
_ = ax2.legend(numpoints=1, loc='lower right')
ax2.set_xlabel('year')
ax2.set_ylabel('[m]')
ax2.grid('on')
#%% check current time
t_check = pd.date_range(start='2021-07-05',end='2021-07-09',freq = '15T')
t_check = t_check.tz_localize('UTC')
time_check_ordinal = []
for i,t in enumerate(t_check):
    time_check_ordinal.append(abs(mdates.date2num(np.datetime64('0000-12-31')))+mdates.date2num(t))

tide_check = utide.reconstruct(t_check, coef_tide)
t_check_local = t_check.tz_convert('Australia/Perth')
#%% plot check 
plt.figure(figsize=(10,6))
plt.plot(t_check_local, tide_check.h)
plt.grid('on')
tz = timezone(timedelta(hours=8))
# bomtime = np.array([datetime(2021,7,5,18,45,0,tzinfo=tz), datetime(2021,7,6,7,45,0,tzinfo=tz),
#            datetime(2021,7,6,18,45,0,tzinfo=tz), datetime(2021,7,7,8,8,0,tzinfo=tz),
#            datetime(2021,7,7,18,45,0,tzinfo=tz),datetime(2021,7,8,8,35,0,tzinfo=tz),
#            datetime(2021,7,8,19,1,0,tzinfo=tz)])
bomtime = pd.to_datetime(['2021-07-05 18:45:00+08:00', '2021-07-06 07:45:00+08:00','2021-07-06 18:45:00+08:00',
                          '2021-07-07 08:00:00+08:00','2021-07-07 18:45:00+08:00','2021-07-08 08:35:00+08:00',
                          '2021-07-08 19:01:00+08:00'])

bomtime2 = bomtime.tz_convert('Australia/Perth')
bomh = np.array([0.55,0.86, 0.54, 0.88, 0.52, 0.90, 0.51])

plt.plot(bomtime2, bomh-0.55,'r*')
plt.xlabel('Date')
plt.ylabel('Tide level [mAHD]')
plt.show()
#%% tide prediction for 2019-2022
t_man = pd.date_range(start='2019-01-01',end='2032-01-01',freq = '60T')
t_man = t_man.tz_localize('UTC')
# time_man_ordinal = []
# for i,t in enumerate(t_man):
#     time_man_ordinal.append(abs(mdates.date2num(np.datetime64('0000-12-31')))+mdates.date2num(t))

tide_man = utide.reconstruct(t_man, coef_tide)
#%%
# t_man_local = t_man.tz_convert('Australia/Perth')
# plt.plot(t_man_local, tide_man.h)

#df = pd.DataFrame(columns = ['datetime_utc', 'z_tide_ahd'])

df = pd.DataFrame({
    'Time (dd/mm/yyyy HH:MM GMT': t_man,
    'Elevation (m AHD)': tide_man.h
})
    
#write csv
csv_path = r'D:\UNSW_Project\Tide_prediction\output'
fname = 'PerthTidesGMT.csv'

# df.to_csv(os.path.join(csv_path,fname), sep=',',index=False, date_format='%d-%m-%Y %H:%M:%S', float_format='%.3f')
# df.to_csv(os.path.join(csv_path,fname), sep=',',index=False, date_format='%d-%m-%Y %H:%M:%S')

df.to_csv(os.path.join(csv_path, fname), sep=',', index=False, date_format='%d-%m-%Y %H:%M')


# fname = 'Mandurah_Marina_2019-2022_15min.csv'
# dataout = pd.DataFrame()
# dataout['time_utc'] = t_man
# dataout['tide_AHD'] = tide_man.h
# dataout.to_csv(os.path.join(csv_path,fname), sep=',',index=False, date_format='%Y-%m-%d %H:%M:%S%z')


#%%
# Seasonal variability (eta_SE)
constituents_seasonal=['SA', 'SSA']
#!! Important - time needs to be in epoch days since Jan 1 0001
# coef_seasonal = utide.solve(time_ordinal, H['Sea Level Detrended'].values, lat=WA_site_lats[ind], method='ols', conf_int='MC',constit=constituents_seasonal,trend=True,nodal=False)
# Seasonal = utide.reconstruct(time_ordinal, coef_seasonal)
# H['Seasonal']=Seasonal.h
    
# # Interannual variability (eta_MMSLA)
# Interannual_dum=H['Sea Level Detrended']-H['Seasonal']
# H['Interannual']=Interannual_dum.rolling(90*24,win_type='hamming',center=True).mean()
    
# H['Seasonal + Interannual']=H['Seasonal']+H['Interannual']

dft = pd.read_csv('PerthTidesGMT.csv', names=['datetime_gmt','tide_m'], header=0)
dft['datetime_gmt'] = pd.to_datetime(dft['datetime_gmt'], format="%d-%m-%Y %H:%M", utc=True)
    
    
