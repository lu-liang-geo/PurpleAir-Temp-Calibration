#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import calendar
import math
import statsmodels.api as sm
import libpysal
import scikit_posthocs as sp


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import linregress, t, friedmanchisquare, kruskal, gaussian_kde
from datetime import datetime, timezone
from scipy.stats import kruskal, kstest, levene
from statsmodels.stats.stattools import durbin_watson


# # Preprocessing
# - merge HOBO and PA based on same location and timestamp
# - remove outliners based on z-score
# - add solar radiation and wind variables

# In[3]:


# merge HOBO and sensor location ID
sensor_info = pd.read_csv("/Users/justintse/Desktop/Thesis/DT_sensor_export.csv",low_memory=False)
sensor_info = sensor_info.dropna(subset=['HOBO_Device_ID'])
sensor_info.rename(columns={'Location_ID__used_as_Key_for_all_tables_': 'Location ID'}, inplace=True)
sensor_info = sensor_info[['Location ID', 'StrataSamp', 'HOBO_Device_ID']]
sensor_info['HOBO_Device_ID'] = sensor_info['HOBO_Device_ID'].astype(int)

HOBO = pd.read_csv("/Users/justintse/Desktop/Thesis/FinalHOBODataset.csv",low_memory=False)
HOBO['Timestamp'] = pd.to_datetime(HOBO['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
HOBO.rename(columns={'Temperature': 'HOBO_Tem',
                             'Sensor_ID': 'HOBO_ID',
                             'RH': 'HOBO_RH'}, inplace=True)

HOBO_df = pd.merge(HOBO, sensor_info, left_on='HOBO_ID', right_on='HOBO_Device_ID', how='inner')


# In[4]:


# merge PA and HOBO data
PA = pd.read_csv("/Users/justintse/Desktop/Thesis/PA_geotagged_hourly_gentle_copy.csv",low_memory=False)
PA = PA.drop(columns=['PM2.5 (CF=1)','Pressure','0.3um','0.5um','1.0um','2.5um','5.0um','10.0um','PM1.0 (CF=1)','PM10.0 (CF=1)'])

#remove the timezone offset value
PA['Timestamp']= PA['Timestamp'].replace('-05:00$', '', regex=True)
PA['Timestamp']= PA['Timestamp'].replace('-06:00$', '', regex=True)
PA.rename(columns={'Sensor_ID': 'PA_ID', 'Temperature': 'PA_Tem', 'Humidity': 'PA_RH'}, inplace=True)

HOBO_df['Timestamp'] = pd.to_datetime(HOBO_df['Timestamp'], errors='coerce')
PA['Timestamp'] = pd.to_datetime(PA['Timestamp'], errors='coerce')
PA = PA.drop(columns=['Latitude', 'Longitude'])
HOBO_PA = pd.merge(HOBO_df, PA, on= ['Timestamp','Location ID'], how='inner')
HOBO_PA = HOBO_PA.dropna(subset=['StrataSamp']) # remove irrelevant location
HOBO_PA.describe()


# In[5]:


uniq_count = HOBO_PA['Location ID'].nunique()
print(f"Total locations: {uniq_count}")

uniqu_stra_count = HOBO_PA.groupby('StrataSamp')['Location ID'].nunique()
print(uniqu_stra_count)


# In[6]:


z_scores_hobo = np.abs((HOBO_PA['HOBO_Tem'] - HOBO_PA['HOBO_Tem'].mean()) / HOBO_PA['HOBO_Tem'].std())
z_scores_pa = np.abs((HOBO_PA['PA_Tem'] - HOBO_PA['PA_Tem'].mean()) / HOBO_PA['PA_Tem'].std())
threshold = 2
df = HOBO_PA[(z_scores_hobo <= threshold) & (z_scores_pa <= threshold)]


# In[7]:


location_counts = df.groupby('Location ID').size()
print(location_counts)


# In[8]:


strat_counts = df.groupby('StrataSamp').size()
strat_percent = (strat_counts / strat_counts.sum()) * 100
print(strat_percent)


# In[9]:


# remove sites with small sample sizes
df = df[(df['Location ID'] != '1081') & (df['Location ID'] != '3411-1')]
df.describe()


# In[10]:


uniq_count = df['Location ID'].nunique()
print(f"Unique location count: {uniq_count}")


# In[11]:


final_locations = df[['Location ID', 'Latitude', 'Longitude']].drop_duplicates()
final_locations.to_csv("/Users/justintse/Desktop/hobo_T_sensor.csv", index=False)


# In[12]:


# check the timeframe
print(df['Timestamp'].min())
print(df['Timestamp'].max())


# # Calculate summary stats

# In[13]:


# overall performance
slope, intercept, r_value, _, _ = linregress(df['HOBO_Tem'], df['PA_Tem'])
mae = mean_absolute_error(df['HOBO_Tem'], df['PA_Tem'])
rmse = np.sqrt(mean_squared_error(df['HOBO_Tem'], df['PA_Tem']))
mbe = np.mean(df['PA_Tem'] - df['HOBO_Tem'])
print(f"Overall r value: {r_value:.2f}")
print(f"Overall mae: {mae:.2f}")
print(f"Overall rmse: {rmse:.2f}")
print(f"Overall mbe: {mbe:.2f}")


# In[14]:


df['Date'] = pd.to_datetime(df['Timestamp'])
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour


# In[15]:


def calculate_statistics(df, group_by_field, target_field, predictor_field, location_id_field):
    def compute_metrics(group):
        y_true = group[target_field]
        y_pred = group[predictor_field]
        
        # Calculate statistics
        counts = len(group)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mbe = np.mean(y_pred - y_true)
        mae = mean_absolute_error(y_true, y_pred)
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
        unique_location_ids = group[location_id_field].nunique()
        
        return pd.Series({
            'Locations': unique_location_ids,
            'n': counts,
            'RMSE': rmse,
            'MBE': mbe,
            'MAE': mae,
            'R': r_value,
            'SE': std_err
        })
    
    return df.groupby(group_by_field).apply(compute_metrics)


# In[16]:


monthly_stats = calculate_statistics(df, group_by_field='Month', target_field='HOBO_Tem', 
                                predictor_field='PA_Tem', location_id_field='Location ID')
print(monthly_stats)


# In[17]:


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
font_name = "Arial"
month_acronyms = [calendar.month_abbr[m] for m in range(1, 13)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=600, gridspec_kw={'width_ratios': [7, 3]})

# --- First Plot: RMSE, MAE, and r ---
ax1.plot(month_acronyms, monthly_stats['RMSE'], marker='o', markersize=10, color='#8EA8C3', label='RMSE', linewidth=2)
ax1.plot(month_acronyms, monthly_stats['MAE'], marker='s', markersize=10, color='#406E8E', label='MAE', linewidth=2)
ax1.plot(month_acronyms, monthly_stats['MBE'], marker='x', markersize=10, color='#797B84', label='MBE', linewidth=2)

# Set labels
ax1.set_xlabel('Month', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylabel('Error (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylim(2, 9)
ax1.set_yticks(np.arange(2, 9, 1))

# Customize ticks
ax1.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax1.get_xticklabels(), fontweight='bold', fontname=font_name)
plt.setp(ax1.get_yticklabels(), fontweight='bold', fontname=font_name)

# Second y-axis for correlation coefficient (r)
ax1_secondary = ax1.twinx()
ax1_secondary.plot(month_acronyms, monthly_stats['R'], marker='^', markersize=10, color='#161925', label=r'$r$', linewidth=2)
ax1_secondary.set_ylabel(r'$r$', rotation=270, labelpad=20, fontsize=20, fontweight='bold', fontname=font_name)
ax1_secondary.set_ylim(0, 1.2)
ax1_secondary.set_yticks(np.arange(0, 1.2, 0.3))
ax1_secondary.tick_params(axis='y', labelsize=20, width=2)
plt.setp(ax1_secondary.get_yticklabels(), fontweight='bold', fontname=font_name)

# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_secondary.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper left',
           prop={'size': 20, 'weight': 'bold', 'family': font_name},
           frameon=False,
           ncol=2)

# --- Second Plot: HOBO Temperature Trend ---

monthly_data = [df[df['Month'] == m]['HOBO_Tem'] for m in range(1, 13)] 

for i, data in enumerate(monthly_data):
    # Calculate the quartiles
    mean_val = np.mean(data)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    min_val, max_val = np.min(data), np.max(data)
    ax2.plot([i+1, i+1], [q25, q75], color='#3066be', lw=3)
    ax2.scatter(i + 1, mean_val, color='#3066be', s=50, marker='o')

ax2.set_xlabel('Month', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax2.set_ylabel('HOBO Temperature (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax2.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax2.get_yticklabels(), fontweight='bold', fontname=font_name)
visible_months = np.arange(1, 13, 2)
ax2.set_xticks(visible_months)
ax2.set_xticklabels([month_acronyms[m-1] for m in visible_months], fontsize=20, fontweight='bold', fontname=font_name)

plt.tight_layout()
plt.show()


# In[18]:


hourly_stats = calculate_statistics(df, group_by_field='Hour', target_field='HOBO_Tem', 
                                predictor_field='PA_Tem', location_id_field='Location ID')
print(hourly_stats)


# In[19]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=600, gridspec_kw={'width_ratios': [7, 3]})

# --- First Plot: RMSE, MAE, and r ---
ax1.plot(hourly_stats['RMSE'], marker='o', markersize=10, color='#8EA8C3', label='RMSE', linewidth=2)
ax1.plot(hourly_stats['MAE'], marker='s', markersize=10, color='#406E8E', label='MAE', linewidth=2)
ax1.plot(hourly_stats['MBE'], marker='x', markersize=10, color='#797B84', label='MBE', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=2)

# Set labels
ax1.set_xlabel('Hour', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylabel('Error (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)

# Customize ticks
ax1.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax1.get_yticklabels(), fontweight='bold', fontname=font_name)

visible_hours = np.arange(0, 24, 3)
ax1.set_xticks(visible_hours)
ax1.set_xticklabels([f'{h}:00' for h in visible_hours], fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylim(-2, 15)
ax1.set_yticks(np.arange(-2, 15, 2))

# Second y-axis for correlation coefficient (r)
ax1_secondary = ax1.twinx()
ax1_secondary.plot(hourly_stats['R'], marker='^', markersize=10, color='#161925', label=r'$r$', linewidth=2)
ax1_secondary.set_ylabel(r'$r$', rotation=270, labelpad=20, fontsize=20, fontweight='bold', fontname=font_name)
ax1_secondary.set_ylim(0, 1.2)
ax1_secondary.set_yticks(np.arange(0, 1.2, 0.3))
ax1_secondary.tick_params(axis='y', labelsize=20, width=2)
plt.setp(ax1_secondary.get_yticklabels(), fontweight='bold', fontname=font_name)

# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_secondary.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper left',
           prop={'size': 20, 'weight': 'bold', 'family': font_name},
           frameon=False,
           ncol=2)

# --- Second Plot: HOBO Temperature Trend ---

hourly_data = [df[df['Hour'] == h]['HOBO_Tem'] for h in range(24)]

for i, data in enumerate(hourly_data):
    # Calculate the quartiles
    mean_val = np.mean(data)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    min_val, max_val = np.min(data), np.max(data)
    ax2.plot([i+1, i+1], [q25, q75], color='#3066be', lw=3)
    ax2.scatter(i + 1, mean_val, color='#3066be', s=50, marker='o')

# Set labels
ax2.set_xlabel('Hour', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax2.set_ylabel('HOBO Temperature (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax2.tick_params(axis='both', labelsize=16, width=2)
plt.setp(ax2.get_yticklabels(), fontweight='bold', fontname=font_name)

visible_hours = np.arange(0, 24, 6)
ax2.set_xticks(visible_hours + 1)
ax2.set_xticklabels([f'{h}:00' for h in visible_hours], fontsize=20, fontweight='bold', fontname=font_name)

plt.tight_layout()
plt.show()


# # Temperature Time Series

# In[20]:


# check the variables
TCEQ = pd.read_csv("/Users/justintse/Desktop/HOBO PA Tem/TCEQ_denton_site_Wind_Radiation_data.csv", low_memory=False)
unique_cd = TCEQ['Parameter Cd'].unique()
print(unique_cd)


# In[21]:


TCEQ['Date'] = TCEQ['Date'].astype(str)
TCEQ['Time'] = TCEQ['Time'].astype(str).str.zfill(5)
TCEQ['Timestamp'] = pd.to_datetime(TCEQ['Date'] + ' ' + TCEQ['Time'], format='%Y%m%d %H:%M', errors='coerce')
TCEQ_pivoted = TCEQ.pivot_table(index='Timestamp', columns='Parameter Cd', values='Value')
TCEQ_pivoted = TCEQ_pivoted.reset_index()


# In[22]:


TCEQ_solarWind = TCEQ_pivoted[['Timestamp',61101, 62101]]
TCEQ_solarWind = TCEQ_solarWind.rename(columns={61101: 'WNDS', 62101: 'DTO_Tem'})

start_date = '2022-03-06 00:00:00'
end_date = '2023-08-16 12:00:00'
TCEQ_solarWind = TCEQ_solarWind[(TCEQ_solarWind['Timestamp'] >= start_date) & (TCEQ_solarWind['Timestamp'] <= end_date)]

TCEQ_WNDS = TCEQ_solarWind[['Timestamp', 'WNDS']].dropna(subset=['WNDS'])
TCEQ_Tem = TCEQ_solarWind[['Timestamp', 'DTO_Tem']].dropna(subset=['DTO_Tem'])
TCEQ_Tem['DTO_Tem'] = (TCEQ_Tem['DTO_Tem'] - 32) * (5/9)


# In[23]:


# calculate lowess 
TCEQ_mean = TCEQ_Tem.groupby('Timestamp').agg({'DTO_Tem': 'mean'}).reset_index()
TCEQ_mean['Date'] = pd.to_datetime(TCEQ_mean['Timestamp'])

TCEQ_mean['Date_num'] = TCEQ_mean['Date'].astype('int64') // 10**9  # Convert to seconds since epoch

lowess = sm.nonparametric.lowess(TCEQ_mean['DTO_Tem'], TCEQ_mean['Date_num'], frac=0.3)
lowess_df = pd.DataFrame(lowess, columns=['Date_num', 'LOESS'])

# Convert back to datetime
lowess_df['Date'] = pd.to_datetime(lowess_df['Date_num'], unit='s')
lowess_df.drop(columns=['Date_num'], inplace=True)


# In[24]:


# Calculate average temperatures
average_pa = df['PA_Tem'].mean()
average_hobo = df['HOBO_Tem'].mean()
average_dto = TCEQ_Tem['DTO_Tem'].mean()
print(f"PA mean: {average_pa}")
print(f"HOBO mean: {average_hobo}")
print(f"DTO mean: {average_dto}")


# In[25]:


df_mean = df.groupby('Timestamp').agg({'HOBO_Tem': 'mean', 'PA_Tem': 'mean'}).reset_index()
df_mean['Date'] = pd.to_datetime(df_mean['Timestamp'])

font_settings= {'fontsize': 16, 'fontweight': 'bold'}

plt.figure(figsize=(12, 4), dpi=600)

# Plot PA_Tem and HOBO_Tem
plt.plot(df_mean['Date'], df_mean['PA_Tem'], label='PA', color='#1a659e', linewidth=0.5)
plt.plot(df_mean['Date'], df_mean['HOBO_Tem'], label='HOBO', color='#20bf55', linewidth=0.5)

# Plot the LOESS trend for DTO
plt.plot(lowess_df['Date'], lowess_df['LOESS'], label='DTO', color='grey', linewidth=3)

plt.axhline(y=21.58, color='#20bf55', linestyle='--', linewidth=3, label='HOBO mean')
plt.axhline(y=25.35, color='#1a659e', linestyle='--', linewidth=3, label='PA mean')

font_settings = {'fontsize': 16, 'fontweight': 'bold'}
plt.xlabel('Date', labelpad=14, **font_settings)
plt.ylabel('Temperature (°C)', labelpad=14, **font_settings)

plt.xlim(df_mean['Date'].min(), df_mean['Date'].max())
plt.tick_params(axis='both', which='major', labelsize=16)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(plt.gca().get_xticklabels(), fontweight='bold')
plt.setp(plt.gca().get_yticklabels(), fontweight='bold')
plt.legend(loc='lower right', prop={'size': 12, 'weight': 'bold'}, frameon=False)

plt.tight_layout()
plt.show()


# # Percentile Comparison

# In[26]:


pa_tem = df['PA_Tem'].values
hobo_tem = df['HOBO_Tem'].values
errors = pa_tem - hobo_tem  

sorted_indices = np.argsort(pa_tem)
sorted_pa = pa_tem[sorted_indices]
sorted_errors = errors[sorted_indices]

# Define percentiles
N = len(sorted_pa)
percentiles = np.linspace(0, 100, N)

plt.figure(figsize=(8, 6), dpi=600)
scatter = plt.scatter(percentiles, sorted_errors, 
                      c=sorted_pa,
                      cmap='viridis',
                      alpha=0.3,
                      s=3)

cbar = plt.colorbar(scatter)
cbar.set_label('PA Temperature (°C)', fontsize=14, fontweight='bold', labelpad=20)
cbar.ax.tick_params(labelsize=12) 
plt.axhline(0, color='k', linestyle='--', linewidth=2, label='Reference Line')
plt.xlabel('Percentile of PA Temperature', fontsize=16, fontweight='bold', labelpad=14)
plt.ylabel('Temperature Anomaly (°C)', fontsize=16, fontweight='bold', labelpad=14)
plt.tick_params(axis='both', which='major', labelsize=16, width=2)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()


# In[27]:


mask = np.logical_and(sorted_errors >= -2, sorted_errors <= 10)
percentage_in_range = (np.sum(mask) / len(sorted_errors)) * 100

print(f"Percentage of values between -2°C and 10°C: {percentage_in_range:.2f}%")


# # Inter- & Intra-group analysis
# Perform Kruskal-Wallis H-test

# In[28]:


# Calculate the absolute error
df['biasr'] = df['PA_Tem'] - df['HOBO_Tem']


# In[29]:


plt.figure(figsize=(8, 8), dpi=300)

ax = sns.boxplot(
    data=df,
    x="StrataSamp",
    y="biasr",
    color='black',
    showfliers=True,
    linewidth=1.5,
    boxprops=dict(facecolor='none', edgecolor='grey'),
    whiskerprops=dict(color='grey'),
    capprops=dict(color='grey'),
    medianprops=dict(color='#E18335'),
    flierprops=dict(marker='o', markersize=5, markerfacecolor='k', markeredgecolor='k', alpha=0.05)
)

ymax = ax.get_ylim()[1]
y_offset = 0.07 * (ymax) 
medians = df.groupby("StrataSamp")["biasr"].median()
xticks = ax.get_xticks()

for tick, label in zip(xticks, medians):
    ax.text(tick, ymax - y_offset, f"{label:.2f}",
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')


ax.set_xlabel("Strata Group", fontsize=16, fontweight='bold', labelpad=14)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=16, fontweight='bold', labelpad=14)
plt.xticks(rotation=45, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()


# In[30]:


# Kolmogorov-Smirnov Test
ks_biasr = stats.kstest(df['biasr'], 'norm', args=(df['biasr'].mean(), df['biasr'].std()))
print(ks_biasr)

# Levene's Test
strata_groups = [group['biasr'].dropna() for _, group in df.groupby('StrataSamp')]
levene_test = stats.levene(*strata_groups)
print(levene_test)


# In[31]:


# Perform Kruskal-Wallis H-test for inter group
stat, pvalue = kruskal(
    df['biasr'][df['StrataSamp'] == 'Rural High'],
    df['biasr'][df['StrataSamp'] == 'Rural Low'],
    df['biasr'][df['StrataSamp'] == 'Suburban High'],
    df['biasr'][df['StrataSamp'] == 'Suburban Low']
)

print(f"Kruskal-Wallis Test p-value: {pvalue}")
print(f"Kruskal-Wallis Test stat: {stat}")


# In[32]:


dunn_results = sp.posthoc_dunn(df, val_col='biasr', group_col='StrataSamp', p_adjust='bonferroni')
print(dunn_results)


# In[33]:


def within_group_test(group_data, group_name):
    sensor_groups = group_data.groupby('Location ID')['biasr'].apply(list)
    
    if len(sensor_groups) < 2:
        print(f"Group '{group_name}' has fewer than two sensors. Skipping Kruskal-Wallis test.")
        return None, None
    
    stat, pvalue = kruskal(*sensor_groups)
    return stat, pvalue

# Perform analysis for each group
for group in df['StrataSamp'].unique():  # Fixed: Added colon here
    group_data = df[df['StrataSamp'] == group]
    stat, pvalue = within_group_test(group_data, group)
    if stat is not None:
        print(f"Kruskal-Wallis Test p-value within '{group}': {pvalue}")


# # Sensor Sensitivity

# wind_df: wind data
# df_solar: filled shortwave

# In[34]:


solar = pd.read_csv('/Users/justintse/Desktop/HOBO PA Tem/Copy of solar_irradiance.csv', low_memory=False)
solar['timestamp_central'] = pd.to_datetime(solar[['year', 'month', 'day', 'hour']], utc=True).dt.tz_convert('America/Chicago')
solar = solar.drop(columns=['year', 'month', 'day', 'hour'])

longwave_columns = [col for col in solar.columns if col.startswith('lw')]
shortwave_columns = [col for col in solar.columns if col.startswith('sw')]

data = [
    {
        'Timestamp': row['timestamp_central'].tz_localize(None),
        'Location ID': lw_col[2:],
        'longwave': row[lw_col],
        'shortwave': row[sw_col]
    }
    for _, row in solar.iterrows()
    for lw_col, sw_col in zip(longwave_columns, shortwave_columns)
]

solar = pd.DataFrame(data)
solar = solar.drop_duplicates(subset=['Timestamp', 'Location ID'])
solar['longwave'] = pd.to_numeric(solar['longwave'], errors='coerce')
solar['shortwave'] = solar['shortwave'].fillna(0)


# In[35]:


solar['Timestamp'] = pd.to_datetime(solar['Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
solar_df = pd.merge(solar, df, on=['Timestamp', 'Location ID'], how='inner')
wind_df = pd.merge(TCEQ_WNDS, df, on=['Timestamp'], how='inner')
lw_df = solar_df.dropna(subset=['longwave'])
final_df = pd.merge(solar_df, TCEQ_WNDS, on=['Timestamp'], how='inner').dropna()


# In[36]:


solar_df_clean = solar_df.dropna(subset=['longwave'])


# In[37]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True, dpi=600)

for month in range(1, 13):
    ax = axes[(month - 1) // 4, (month - 1) % 4]
    month_data = df[df['Month'] == month]['biasr']

    sns.histplot(month_data, bins=30, kde=True, color="k", ax=ax, stat="percent")
    ax.axvline(0, color="black", linestyle="--", linewidth=2)

    month_abbr = calendar.month_abbr[month]
    ax.text(0.95, 0.95, month_abbr, transform=ax.transAxes, ha='right', va='top', 
            fontsize=16, fontweight='bold', fontname="Arial")

    if (month - 1) // 4 == 2:  
        ax.set_xlabel("Temperature Anomaly (°C)", fontsize=14, fontname="Arial")
    else:
        ax.set_xlabel("")

    if (month - 1) % 4 == 0:  
        ax.set_ylabel("Percentage (%)", fontsize=14, fontname="Arial")
    else:
        ax.set_ylabel("")

    ax.tick_params(axis='both', labelsize=12)
    plt.setp(ax.get_xticklabels(), fontname="Arial")
    plt.setp(ax.get_yticklabels(), fontname="Arial")

plt.tight_layout()
plt.show()


# In[38]:


# temperature vs temperature anomaly
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for month in range(1, 13):
    ax = axes[(month - 1) // 4, (month - 1) % 4]
    month_data = df[df['Month'] == month]

    ax.scatter(month_data['HOBO_Tem'], month_data['PA_Tem'], color="k", s=3, alpha=0.05)
    slope, intercept, r_value, p_value,_ = linregress(month_data['HOBO_Tem'], month_data['PA_Tem'])
    r_squared = r_value**2
    
    min_val = min(month_data['HOBO_Tem'].min(), month_data['PA_Tem'].min())
    max_val = max(month_data['HOBO_Tem'].max(), month_data['PA_Tem'].max())

    # 1:1 Line (Reference)
    ax.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='--', linewidth=2, label='1:1 Line')


    # Best fit line
    z = np.polyfit(month_data['HOBO_Tem'], month_data['PA_Tem'], 1)
    p = np.poly1d(z)
    ax.plot(month_data['HOBO_Tem'], p(month_data['HOBO_Tem']), color='#E18335', linestyle='-', linewidth=2, label='Best Fit Line')
    
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = ''
                
    line_eq = f"y = {slope:.2f}x + {intercept:.2f}"
    r_text = rf"$\mathit{{r}}$ = {r_value:.2f}{significance}"

    ax.text(0.05, 0.92, line_eq, transform=ax.transAxes,fontsize=16)
    ax.text(0.05, 0.84, r_text, transform=ax.transAxes,fontsize=16)
    month_abbr = calendar.month_abbr[month]
    ax.text(0.88, 0.95, month_abbr, transform=ax.transAxes, ha='right', va='top', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    if (month - 1) // 4 == 2:  # Last row
        ax.set_xlabel('HOBO Temperature (°C)', fontsize=16)
    else:
        ax.set_xlabel('')

    if (month - 1) % 4 == 0:  # First column
        ax.set_ylabel('Temperature Anomaly(°C)', fontsize=16)
    else:
        ax.set_ylabel('')

plt.legend(loc='lower right',fontsize=14, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[39]:


def plot_monthly_scatter(df, x, y, x_label, y_label):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=600)

    for month in range(1, 13):
        ax = axes[(month - 1) // 4, (month - 1) % 4]
        month_data = df[df['Month'] == month]

        ax.scatter(month_data[x], month_data[y], color="k", s=3, alpha=0.05)
        
        # Regression line
        if len(month_data) > 1:
            slope, intercept, r_value, p_value, _ = linregress(month_data[x], month_data[y])
            z = np.polyfit(month_data[x], month_data[y], 1)
            p = np.poly1d(z)
            ax.plot(month_data[x], p(month_data[x]), color='#E18335', linestyle='-', linewidth=2, label='Best Fit Line')
            
            # Set up r_text with significance based on p-value
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = ''
                
            line_eq = f"y = {slope:.2f}x + {intercept:.2f}"
            r_text = rf"$\mathit{{r}}$ = {r_value:.2f}{significance}"
            
            ax.text(0.05, 0.92, line_eq, transform=ax.transAxes, fontsize=16)
            ax.text(0.05, 0.84, r_text, transform=ax.transAxes, fontsize=16)
        
        ax.axhline(y=0, color='k', linewidth=2, linestyle='--', label='Reference')

        # Month label
        month_abbr = calendar.month_abbr[month]
        ax.text(0.95, 0.95, month_abbr, transform=ax.transAxes, ha='right', va='top', fontsize=16, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=16)

        # Label x-axis only for the last row
        if (month - 1) // 4 == 2:  
            ax.set_xlabel(x_label, fontsize=16)
        else:
            ax.set_xlabel('')

        # Label y-axis only for the first column
        if (month - 1) % 4 == 0:  
            ax.set_ylabel(y_label, fontsize=16)
        else:
            ax.set_ylabel('')
            
    plt.legend(loc='lower right',fontsize=14, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/Users/justintse/Desktop/sample.png', dpi=600, bbox_inches='tight')
    plt.show()


# In[40]:


plot_monthly_scatter(df, x='HOBO_Tem', y='biasr', x_label='Temperature (°C)', y_label='Temperature Anomaly (°C)')


# In[41]:


plot_monthly_scatter(df, x='HOBO_RH', y='biasr', x_label='Relative Humidity (%)', y_label='Temperature Anomaly (°C)')


# In[42]:


plot_monthly_scatter(lw_df, x='longwave', y='biasr', x_label='Longwave Irradiance (W/m²)', y_label='Temperature Anomaly (°C)')


# In[43]:


plot_monthly_scatter(solar_df, x='shortwave', y='biasr', x_label='Shortwave Irradiance (W/m²)', y_label='Temperature Anomaly (°C)')


# In[44]:


plot_monthly_scatter(wind_df, x='WNDS', y='biasr', x_label='Wind Speed (m/s)', y_label='Temperature Anomaly (°C)')


# In[64]:


def add_regression_with_annotation(ax, x, y, title):
    
    valid_data = pd.DataFrame({ 'x': x, 'y': y }).dropna()
    x_valid = valid_data['x']
    y_valid = valid_data['y']
    
    # Generate a 2D histogram (heatmap)
    hist = ax.hist2d(x_valid, y_valid, bins=80, cmap='Greys', vmin=1, vmax=250, alpha=0.8)
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    ax.plot(x, slope * x + intercept, color='#E18335', linewidth=2, label='Best Fit Line')
    ax.axhline(y=0, color='black', linestyle='--', label='Reference Line')
    minus = '\u2212'
    slope_sign = minus if slope < 0 else ''
    slope_str = f"{abs(slope):.2f}"
    intercept_sign = f" + {abs(intercept):.2f}" if intercept >= 0 else f" {minus} {abs(intercept):.2f}"
    equation = f"y = {slope_sign}{slope_str}x{intercept_sign}"

    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = ''
    r_text = rf"$\mathit{{r}}$ = {r_value:.2f}{significance}".replace('-', '\u2212')
    ax.annotate(equation, xy=(0.95, 0.15), xycoords='axes fraction', fontsize=16,
                ha='right', va='bottom', color='black')
    ax.annotate(r_text, xy=(0.95, 0.06), xycoords='axes fraction', fontsize=16,
                ha='right', va='bottom', color='black')
    
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=20, ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'), fontweight='normal')
    return hist

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  # 2 rows, 3 columns
axs = axs.flatten()  # Flatten for easy indexing

# (a) HOBO_Tem
x = df['HOBO_Tem']
y = df['biasr']
add_regression_with_annotation(axs[0], x, y, title='(a)')
axs[0].set_xlabel('HOBO Temperature (°C)', fontsize=14, fontweight='bold')
axs[0].set_ylabel('Temperature Anomaly (°C)', fontsize=14, fontweight='bold')

# (b) HOBO_RH
x = df['HOBO_RH']
y = df['biasr']
add_regression_with_annotation(axs[1], x, y, title='(b)')
axs[1].set_xlabel('HOBO RH (%)', fontsize=14, fontweight='bold')
axs[1].set_ylabel('')
axs[1].set_yticks([])

# (c) Longwave Radiation
x = lw_df['longwave']
y = lw_df['biasr']
add_regression_with_annotation(axs[2], x, y, title='(c)')
axs[2].set_xlabel('Longwave Radiation (W/m²)', fontsize=14, fontweight='bold')
axs[2].set_ylabel('')
axs[2].set_yticks([])

# (d) Shortwave Radiation
x = solar_df['shortwave']
y = solar_df['biasr']
add_regression_with_annotation(axs[3], x, y, title='(d)')
axs[3].set_xlabel('Shortwave Radiation (W/m²)', fontsize=14, fontweight='bold')
axs[3].set_ylabel('Temperature Anomaly (°C)', fontsize=14, fontweight='bold')

# (e) Wind Speed
x = wind_df['WNDS']
y = wind_df['biasr']
add_regression_with_annotation(axs[4], x, y, title='(e)')
axs[4].set_xlabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
axs[4].set_ylabel('')
axs[4].set_yticks([])
axs[4].legend(loc='upper right', prop={'size': 12, 'weight': 'bold'}, frameon=False)

# Remove the empty 6th subplot
fig.delaxes(axs[5])

# Formatting: Bold ticks and borders
for ax in axs[:5]:  # Only the first 5 plots
    ax.tick_params(axis='both', labelsize=14)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(1)

plt.tight_layout()
plt.savefig('/Users/justintse/Desktop/sample.png', dpi=600, bbox_inches='tight')
plt.show()


# # Diurnal cycle anomalies model

# In[46]:


tem_cycle = final_df.groupby(['Month', 'Hour'])['biasr'].mean().reset_index()
tem_cycle.rename(columns={'biasr': 'mean_diurnal_anomaly'}, inplace=True)
tem_amly = final_df.merge(tem_cycle, on=['Month', 'Hour'], how='left')
tem_amly['dT_amly'] = tem_amly['biasr'] - tem_amly['mean_diurnal_anomaly']


# In[47]:


def calculate_amly(df, column_name):

    cycle = df.groupby(['Month', 'Hour'])[column_name].mean().reset_index()
    cycle.rename(columns={column_name: 'mean_diurnal'}, inplace=True)

    df_amly = df.merge(cycle, on=['Month', 'Hour'], how='left')
    df_amly[f'{column_name}_amly'] = df_amly[column_name] - df_amly['mean_diurnal']
    
    return df_amly

lw_amly = calculate_amly(lw_df, 'longwave')
sw_amly = calculate_amly(solar_df, 'shortwave')
wnds_amly = calculate_amly(wind_df, 'WNDS')
hobo_amly = calculate_amly(df, 'HOBO_Tem')
rh_amly = calculate_amly(df, 'HOBO_RH')


# In[48]:


merged_amly = tem_amly

# List of DataFrames to merge
df_list = [lw_amly[['Location ID', 'Timestamp', 'longwave_amly']],
           sw_amly[['Location ID', 'Timestamp', 'shortwave_amly']],
           wnds_amly[['Location ID', 'Timestamp', 'WNDS_amly']],
           hobo_amly[['Location ID', 'Timestamp', 'HOBO_Tem_amly']],
           rh_amly[['Location ID', 'Timestamp', 'HOBO_RH_amly']]]

for dataframe in df_list:
    merged_amly = merged_amly.merge(dataframe, on=['Location ID', 'Timestamp'], how='inner')


# In[49]:


X = merged_amly[['WNDS_amly', 'shortwave_amly', 'longwave_amly', 'HOBO_Tem_amly', 'HOBO_RH_amly']]
y = merged_amly['dT_amly']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# # Calibration Model Testing

# ## Additive Terms Model
# 
# - Model 1: PA 
# - Model 2: PA + RH
# - Model 3: PA + SW
# - Model 4: PA + LW
# - Model 5: PA + LW + SW
# - Model 6: PA + RH + LW + SW
# - Model 7: PA + RH + SW + LW + WNDS
# - Model 8: PA + RH + PA x RH

# In[50]:


def run_reg(df, predictors, target):
    # Split the dataset
    X = df[predictors]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add constant for intercept
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit the regression model using statsmodels
    model = sm.OLS(y_train, X_train).fit()
    summary = model.summary()

    # Predict on test set
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mbe = np.mean(y_pred.values - y_test.values)
    r2 = model.rsquared
    aic = model.aic
    
    errors = y_pred.values - y_test.values
    print(f"Mean: {np.mean(errors):.4f}, Min: {np.min(errors):.2f}, Max: {np.max(errors):.2f}")


    # Extract coefficients and intercept
    intercept = model.params.iloc[0]
    coefficients = model.params[1:].values

    equation_terms = " + ".join([f"{coefficients[i]:.4f}*{predictor}" for i, predictor in enumerate(predictors)])
    model_equation = f"{target} = {intercept:.4f} + {equation_terms}"

    print(f"Model equation: {model_equation}")
    print(f'R-squared: {r2:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MBE: {mbe:.2f}, AIC: {aic:.2f}')
    print(f'Number of training samples: {len(X_train)}, Number of testing samples: {len(X_test)}')

    return {
        "equation": model_equation,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "aic": aic,
        "coefficients": coefficients,
        "intercept": intercept,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "summary": summary
    }


# In[51]:


# model 1
predictors = ['PA_Tem']
target = 'HOBO_Tem'
results = run_reg(df, predictors, target)


# In[52]:


# model 2
predictors = ['PA_Tem', 'PA_RH']
target = 'HOBO_Tem'
results = run_reg(df, predictors, target)


# In[53]:


# model 3
predictors = ['PA_Tem', 'shortwave']
target = 'HOBO_Tem'
results = run_reg(solar_df, predictors, target)


# In[54]:


# model 4
predictors = ['PA_Tem','longwave']
target = 'HOBO_Tem'
results = run_reg(lw_df, predictors, target)


# In[55]:


# model 5
predictors = ['PA_Tem','longwave','shortwave']
target = 'HOBO_Tem'
results = run_reg(solar_df_clean, predictors, target)


# In[56]:


# model 6
predictors = ['PA_Tem','longwave','shortwave', 'PA_RH']
target = 'HOBO_Tem'
results = run_reg(solar_df_clean, predictors, target)


# In[57]:


# model 7
predictors = ['PA_Tem','longwave','shortwave', 'PA_RH', 'WNDS']
target = 'HOBO_Tem'
results = run_reg(final_df, predictors, target)


# ## Additive + Multiplicative Terms Model

# In[58]:


def run_reg_with_interaction(df, predictors, interaction_terms, target):
    # Create interaction terms
    for term1, term2 in interaction_terms:
        df[f"{term1}:{term2}"] = df[term1] * df[term2]

    interaction_columns = [f"{term1}:{term2}" for term1, term2 in interaction_terms]
    all_predictors = predictors + interaction_columns

    # Split the data
    X = df[all_predictors]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add constant for intercept
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit the regression model using statsmodels
    model = sm.OLS(y_train, X_train).fit()
    summary = model.summary()

    # Predict on test set
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mbe = np.mean(y_pred.values - y_test.values)
    r2 = model.rsquared
    aic = model.aic
    
    print(f"Mean: {np.mean(errors):.4f}, Min: {np.min(errors):.2f}, Max: {np.max(errors):.2f}")

    # Extract coefficients and intercept
    intercept = model.params.iloc[0]
    coefficients = model.params[1:].values

    equation_terms = " + ".join([f"{coefficients[i]:.4f}*{predictor}" for i, predictor in enumerate(predictors)])
    model_equation = f"{target} = {intercept:.4f} + {equation_terms}"

    print(f"Model equation: {model_equation}")
    print(f'R-squared: {r2:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f},MBE: {mbe:.2f}, AIC: {aic:.2f}')
    print(f'Number of training samples: {len(X_train)}, Number of testing samples: {len(X_test)}')

    return {
        "equation": model_equation,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "aic": aic,
        "coefficients": coefficients,
        "intercept": intercept,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "summary": summary
    }


# In[59]:


# Model 8
predictors = ['PA_Tem', 'PA_RH']
interaction_terms = [('PA_Tem','PA_RH')]
target = 'HOBO_Tem'
results = run_reg_with_interaction(df, predictors, interaction_terms, target)


# # Best model with variable anomaly

# In[60]:


lw_sw_amly = pd.merge(lw_amly[['Location ID', 'Timestamp', 'longwave_amly']],
                     sw_amly[['Location ID', 'Timestamp', 'shortwave_amly']],
                     on=['Location ID', 'Timestamp'],
                     how='inner')

lw_sw_amly_df = pd.merge(solar_df_clean, lw_sw_amly, on=['Location ID', 'Timestamp'], how='inner')


# In[61]:


# Model 9
predictors = ['PA_Tem','longwave','shortwave', 'longwave_amly', 'shortwave_amly']
target = 'HOBO_Tem'
results = run_reg(lw_sw_amly_df, predictors, target)


# ## Multicollinearity test

# In[132]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[183]:


features = lw_sw_amly_df[['PA_Tem','longwave','shortwave', 'longwave_amly', 'shortwave_amly']]
features = pd.DataFrame(sm.add_constant(features))
vif = pd.DataFrame()
vif['Feature'] = features.columns
vif['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

print(vif)


# ## Comparing corrected and uncorrected scenario

# In[105]:


model_df = lw_sw_amly_df.copy()

# model 5
predictors = ['PA_Tem','longwave','shortwave', 'longwave_amly', 'shortwave_amly']
target = 'HOBO_Tem'
results = run_reg(model_df, predictors, target)


# In[106]:


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi = 600)
axes = axs

# Plotting Function
def plot_regression(ax, x, y, xlabel, ylabel, title, show_xlabel=True, show_ylabel=True, show_yticks=True, show_legend=False):
    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color='#E18335', linewidth=3, label='Best Fit Line')
    ax.hist2d(x, y, bins=80, cmap='Greys', vmin=1, vmax=250, alpha=0.8)

    r_squared = 1 - (np.sum((y - p(x))**2) / np.sum((y - np.mean(y))**2))
    ax.text(0.95, 0.17, f"R² = {r_squared:.2f}", transform=ax.transAxes, fontsize=22, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'), fontweight='bold')
    ax.text(0.95, 0.10, f"RMSE = {rmse:.2f}", transform=ax.transAxes, fontsize=22, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'), fontweight='bold')
    ax.text(0.95, 0.03, f"MAE = {mae:.2f}", transform=ax.transAxes, fontsize=22, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'), fontweight='bold')

    ax.plot([x.min(), x.max()], [x.min(), x.max()],
            color='k', linestyle='--', linewidth=3, label='Perfect Fit Line')

    if show_legend:
        ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.4, 1), frameon=False)

    # Apply bold font to labels and ticks
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad = 15)
    else:
        ax.set_xlabel('')

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=20, fontweight='bold', labelpad = 10)
    else:
        ax.set_ylabel('')
        if not show_yticks:
            ax.set_yticks([])

    ax.set_ylim(-4, 52)
    ax.set_xlim(0, 42)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=28, ha='left', va='top',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'), fontweight='normal')
    ax.tick_params(axis='both', labelsize=20, width=2, length=6)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

# Plot 1: Before correction
x = model_df['HOBO_Tem']
y = model_df['PA_Tem']
plot_regression(axes[0], x, y, '', 'PA Temperature (°C)', '(a)', show_xlabel=False, show_ylabel=True, show_yticks=True)

# Plot 2: After PA correction
correction_C = -8 / 1.8
model_df.loc[:, 'PA_Tem_corrected'] = model_df['PA_Tem'] + correction_C
x = model_df['HOBO_Tem']
y = model_df['PA_Tem_corrected']
plot_regression(axes[1], x, y, 'HOBO Temperature (°C)', '', '(b)', show_xlabel=True, show_ylabel=False, show_yticks=False)

# Plot 3: After our correction
model_df['PA_corrected'] = (
    results["coefficients"][0] * model_df['PA_Tem'] +
    results["coefficients"][1] * model_df['longwave'] +
    results["coefficients"][2] * model_df['shortwave'] +
    results["coefficients"][3] * model_df['longwave_amly'] +
    results["coefficients"][4] * model_df['shortwave_amly'] +
    results["intercept"]
)
x = model_df['HOBO_Tem']
y = model_df['PA_corrected']
plot_regression(axes[2], x, y, '', '', '(c)', show_xlabel=False, show_ylabel=False, show_yticks=False, show_legend=True)

plt.tight_layout()
plt.show()


# In[163]:


def location_errors(group):
    HOBO = group['HOBO_Tem'].values

    # Uncalibrated
    uncalibrated = group['PA_Tem'].values
    mae_uncal = mean_absolute_error(HOBO, uncalibrated)
    rmse_uncal = np.sqrt(mean_squared_error(HOBO, uncalibrated))
    mbe_uncal = np.mean(uncalibrated - HOBO)

    # Calibrated
    calibrated = group['PA_corrected'].values
    mae_cal = mean_absolute_error(HOBO, calibrated)
    rmse_cal = np.sqrt(mean_squared_error(HOBO, calibrated))
    mbe_cal = np.mean(calibrated - HOBO)

    # % changes
    mae_change = ((mae_cal - mae_uncal) / mae_uncal) * 100 if mae_uncal != 0 else np.nan
    rmse_change = ((rmse_cal - rmse_uncal) / rmse_uncal) * 100 if rmse_uncal != 0 else np.nan
    mbe_change = ((mbe_cal - mbe_uncal) / mbe_uncal) * 100 if mbe_uncal != 0 else np.nan

    return pd.Series({
        'MAE_Uncal': mae_uncal,
        'RMSE_Uncal': rmse_uncal,
        'MBE_Uncal': mbe_uncal,
        'MAE_Cal': mae_cal,
        'RMSE_Cal': rmse_cal,
        'MBE_Cal': mbe_cal,
        'MAE_Change': mae_change,
        'RMSE_Change': rmse_change,
        'MBE_Change': mbe_change
    })

location_errors = model_df.groupby('Location ID').apply(location_errors).reset_index()
location_errors


# In[117]:


import geopandas as gpd
from shapely.geometry import Point

location_coords = model_df.groupby('Location ID')[['Latitude', 'Longitude']].first().reset_index()
location_errors = location_errors.merge(location_coords, on='Location ID')
gdf = gpd.GeoDataFrame(location_errors, 
                       geometry=gpd.points_from_xy(location_errors['Longitude'], location_errors['Latitude']),
                       crs="EPSG:4326")


# In[130]:


tracts = gpd.read_file('/Users/justintse/Desktop/HOBO PA Tem/Census_Tract_Planning_Database_2019-shp.zip')

if tracts.crs != gdf.crs:
    tracts = tracts.to_crs(gdf.crs)


# In[172]:


cmap = plt.colormaps['viridis']
norm = colors.Normalize(vmin=-100, vmax=-20)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []

fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=300, constrained_layout=True)

# Plot MAE
tracts.plot(ax=axes[0], facecolor='none', edgecolor='grey', linewidth=1)
gdf.plot(column='RMSE_Change', cmap=cmap, ax=axes[0], markersize=150, edgecolor='black', norm=norm)

# Plot RMSE
tracts.plot(ax=axes[1], facecolor='none', edgecolor='grey', linewidth=1)
gdf.plot(column='MAE_Change', cmap=cmap, ax=axes[1], markersize=150, edgecolor='black', norm=norm)

# Plot MBE
tracts.plot(ax=axes[2], facecolor='none', edgecolor='grey', linewidth=1)
gdf.plot(column='MBE_Change', cmap=cmap, ax=axes[2], markersize=150, edgecolor='black', norm=norm)

# colorbar
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.01)
cbar.set_label('Change (%)', fontsize=16, fontweight='bold', labelpad = 15)
tick_interval = 20
ticks = np.arange(-100, -20 + tick_interval, tick_interval)
cbar.set_ticks(ticks)
cbar.ax.tick_params(labelsize=14)

# Add subplot annotations
annotations = ['(a)', '(b)', '(c)']
for i, label in enumerate(annotations):
    axes[i].text(
        0.05, 0.95, label,
        transform=axes[i].transAxes,
        fontsize=28,
        fontweight='regular',
        verticalalignment='top',
        horizontalalignment='left'
    )

for ax in axes:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.show()


# # Extended Heat Index Comparison
# 
# @article{20heatindex,
# Title   = {Extending the Heat Index},
# Author  = {Yi-Chuan Lu and David M. Romps},
# Journal = {Journal of Applied Meteorology and Climatology},
# Volume  = {61},
# Number  = {10},
# Pages   = {1367--1383},
# Year    = {2022},
# }
# 
# This headindex function returns the Heat Index in Kelvin. The inputs are:
# - T, the temperature in Kelvin
# - RH, the relative humidity, which is a value from 0 to 1
# - show_info is an optional logical flag. If true, the function returns the physiological state.

# In[96]:


import math


# In[480]:


# Thermodynamic parameters
Ttrip = 273.16       # K
ptrip = 611.65       # Pa
E0v   = 2.3740e6     # J/kg
E0s   = 0.3337e6     # J/kg
rgasa = 287.04       # J/kg/K 
rgasv = 461.         # J/kg/K 
cva   = 719.         # J/kg/K
cvv   = 1418.        # J/kg/K 
cvl   = 4119.        # J/kg/K
cvs   = 1861.        # J/kg/K
cpa   = cva + rgasa
cpv   = cvv + rgasv

# The saturation vapor pressure
def pvstar(T):
    if T == 0.0:
        return 0.0
    elif T<Ttrip:
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * math.exp( (E0v + E0s -(cvv-cvs)*Ttrip)/rgasv * (1./Ttrip - 1./T) )
    else:
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * math.exp( (E0v       -(cvv-cvl)*Ttrip)/rgasv * (1./Ttrip - 1./T) )

# The latent heat of vaporization of water
def Le(T):
    return (E0v + (cvv-cvl)*(T-Ttrip) + rgasv*T)

# Thermoregulatory parameters
sigma       = 5.67e-8                     # W/m^2/K^4 , Stefan-Boltzmann constant
epsilon     = 0.97                        #           , emissivity of surface, steadman1979
M           = 83.6                        # kg        , mass of average US adults, fryar2018
H           = 1.69                        # m         , height of average US adults, fryar2018
A           = 0.202*(M**0.425)*(H**0.725) # m^2       , DuBois formula, parson2014
cpc         = 3492.                       # J/kg/K    , specific heat capacity of core, gagge1972
C           = M*cpc/A                     #           , heat capacity of core
r           = 124.                        # Pa/K      , Zf/Rf, steadman1979
Q           = 180.                        # W/m^2     , metabolic rate per skin area, steadman1979
phi_salt    = 0.9                         #           , vapor saturation pressure level of saline solution, steadman1979
Tc          = 310.                        # K         , core temperature, steadman1979
Pc          = phi_salt * pvstar(Tc)       #           , core vapor pressure
L           = Le(310.)                    #           , latent heat of vaporization at 310 K
p           = 1.013e5                     # Pa        , atmospheric pressure
eta         = 1.43e-6                     # kg/J      , "inhaled mass" / "metabolic rate", steadman1979
Pa0         = 1.6e3                       # Pa        , reference air vapor pressure in regions III, IV, V, VI, steadman1979

# Thermoregulatory functions
def Qv(Ta,Pa): # respiratory heat loss, W/m^2
    return  eta * Q *(cpa*(Tc-Ta)+L*rgasa/(p*rgasv) * ( Pc-Pa ) )
def Zs(Rs): # mass transfer resistance through skin, Pa m^2/W
    return (52.1 if Rs == 0.0387 else 6.0e8 * Rs**5)
def Ra(Ts,Ta): # heat transfer resistance through air, exposed part of skin, K m^2/W
    hc      = 17.4
    phi_rad = 0.85
    hr      = epsilon * phi_rad * sigma* (Ts**2 + Ta**2)*(Ts + Ta)
    return 1./(hc+hr)
def Ra_bar(Tf,Ta): # heat transfer resistance through air, clothed part of skin, K m^2/W
    hc      = 11.6
    phi_rad = 0.79
    hr      = epsilon * phi_rad * sigma* (Tf**2 + Ta**2)*(Tf + Ta)
    return 1./(hc+hr)
def Ra_un(Ts,Ta): # heat transfer resistance through air, when being naked, K m^2/W
    hc      = 12.3
    phi_rad = 0.80
    hr      = epsilon * phi_rad * sigma* (Ts**2 + Ta**2)*(Ts + Ta)
    return 1./(hc+hr)

Za     = 60.6/17.4  # Pa m^2/W, mass transfer resistance through air, exposed part of skin
Za_bar = 60.6/11.6  # Pa m^2/W, mass transfer resistance through air, clothed part of skin
Za_un  = 60.6/12.3  # Pa m^2/W, mass transfer resistance through air, when being naked

# tolerance and maximum iteration for the root solver 
tol     = 1e-8
tolT    = 1e-8
maxIter = 100

# Given air temperature and relative humidity, returns the equivalent variables 
def find_eqvar(Ta,RH):
    Pa    = RH*pvstar(Ta) #         , air vapor pressure
    Rs    = 0.0387        # m^2K/W  , heat transfer resistance through skin
    phi   = 0.84          #         , covering fraction
    dTcdt = 0.            # K/s     , rate of change in Tc
    m     = (Pc-Pa)/(Zs(Rs)+Za)
    m_bar = (Pc-Pa)/(Zs(Rs)+Za_bar)
    Ts = solve(lambda Ts: (Ts-Ta)/Ra(Ts,Ta)     + (Pc-Pa)/(Zs(Rs)+Za)     - (Tc-Ts)/Rs, max(0.,min(Tc,Ta)-Rs*abs(m)),     max(Tc,Ta)+Rs*abs(m),    tol,maxIter)
    Tf = solve(lambda Tf: (Tf-Ta)/Ra_bar(Tf,Ta) + (Pc-Pa)/(Zs(Rs)+Za_bar) - (Tc-Tf)/Rs, max(0.,min(Tc,Ta)-Rs*abs(m_bar)), max(Tc,Ta)+Rs*abs(m_bar),tol,maxIter)
    flux1 = Q-Qv(Ta,Pa)-(1.-phi)*(Tc-Ts)/Rs                   # C*dTc/dt when Rf=Zf=\inf
    flux2 = Q-Qv(Ta,Pa)-(1.-phi)*(Tc-Ts)/Rs - phi*(Tc-Tf)/Rs  # C*dTc/dt when Rf=Zf=0
    if (flux1 <= 0.) : # region I
        eqvar_name = "phi"
        phi = 1.-(Q-Qv(Ta,Pa))*Rs/(Tc-Ts)
        Rf  = float('inf')
    elif (flux2 <=0.) : # region II&III
        eqvar_name = "Rf"
        Ts_bar = Tc - (Q-Qv(Ta,Pa))*Rs/phi + (1./phi -1.)*(Tc-Ts)
        Tf = solve(lambda Tf: (Tf-Ta)/Ra_bar(Tf,Ta) + (Pc-Pa)*(Tf-Ta)/((Zs(Rs)+Za_bar)*(Tf-Ta)+r*Ra_bar(Tf,Ta)*(Ts_bar-Tf)) - (Tc-Ts_bar)/Rs, Ta,Ts_bar,tol,maxIter)
        Rf = Ra_bar(Tf,Ta)*(Ts_bar-Tf)/(Tf-Ta)
    else: # region IV,V,VI
        Rf = 0.
        flux3 =  Q-Qv(Ta,Pa)-(Tc-Ta)/Ra_un(Tc,Ta)-(phi_salt*pvstar(Tc)-Pa)/Za_un
        if (flux3 < 0.) : # region IV,V
            Ts = solve(lambda Ts: (Ts-Ta)/Ra_un(Ts,Ta)+(Pc-Pa)/(Zs((Tc-Ts)/(Q-Qv(Ta,Pa)))+Za_un)-(Q-Qv(Ta,Pa)),0.,Tc,tol,maxIter)
            Rs = (Tc-Ts)/(Q-Qv(Ta,Pa))
            eqvar_name = "Rs"
            Ps = Pc - (Pc-Pa)* Zs(Rs)/( Zs(Rs)+Za_un)
            if (Ps > phi_salt * pvstar(Ts)):  # region V
                Ts = solve( lambda Ts : (Ts-Ta)/Ra_un(Ts,Ta) + (phi_salt*pvstar(Ts)-Pa)/Za_un -(Q-Qv(Ta,Pa)), 0.,Tc,tol,maxIter)
                Rs = (Tc-Ts)/(Q-Qv(Ta,Pa))
                eqvar_name = "Rs*"
        else: # region VI
            Rs = 0.
            eqvar_name = "dTcdt"
            dTcdt = (1./C)* flux3
    return [eqvar_name,phi,Rf,Rs,dTcdt]

# given the equivalent variable, find the Heat Index
def find_T(eqvar_name,eqvar):
    if (eqvar_name == "phi"):
        T = solve(lambda T: find_eqvar(T,1.)[1]-eqvar,0.,240.,tolT,maxIter)
        region = 'I'
    elif (eqvar_name == "Rf"):
        T = solve(lambda T: find_eqvar(T,min(1.,Pa0/pvstar(T)))[2]-eqvar,230.,300.,tolT,maxIter)
        region = ('II' if Pa0>pvstar(T) else 'III')
    elif (eqvar_name == "Rs" or eqvar_name == "Rs*"):
        T = solve(lambda T: find_eqvar(T,Pa0/pvstar(T))[3]-eqvar,295.,350.,tolT,maxIter)
        region = ('IV' if eqvar_name == "Rs" else 'V')
    else:
        T = solve(lambda T: find_eqvar(T,Pa0/pvstar(T))[4]-eqvar,340.,1000.,tolT,maxIter)
        region = 'VI'
    return T, region

# combining the two functions find_eqvar and find_T
def heatindex(Ta,RH,show_info=False):
    dic = {"phi":1,"Rf":2,"Rs":3,"Rs*":3,"dTcdt":4}
    eqvars = find_eqvar(Ta,RH)
    T, region = find_T(eqvars[0],eqvars[dic[eqvars[0]]])
    if (Ta == 0.): T = 0.
    if (show_info==True):
        if region=='I':
            print("Region I, covering (variable phi)")
            print("Clothing fraction is "+ str(round(eqvars[1],3)))
        elif region=='II':
            print("Region II, clothed (variable Rf, pa = pvstar)")
            print("Clothing thickness is "+ str(round((eqvars[2]/16.7)*100.,3))+" cm")
        elif region=='III':
            print("Region III, clothed (variable Rf, pa = pref)")
            print("Clothing thickness is "+ str(round((eqvars[2]/16.7)*100.,3))+" cm")
        elif region=='IV':
            kmin = 5.28               # W/K/m^2     , conductance of tissue
            rho  = 1.0e3              # kg/m^3      , density of blood
            c    = 4184.              # J/kg/K      , specific heat of blood
            print("Region IV, naked (variable Rs, ps < phisalt*pvstar)")
            print("Blood flow is " + str(round(( (1./eqvars[3] - kmin)*A/(rho*c) ) *1000.*60.,3))+" l/min")
        elif region=='V':
            kmin = 5.28               # W/K/m^2     , conductance of tissue
            rho  = 1.0e3              # kg/m^3      , density of blood
            c    = 4184.              # J/kg/K      , specific heat of blood
            print("Region V, naked dripping sweat (variable Rs, ps = phisalt*pvstar)")
            print("Blood flow is " + str(round(( (1./eqvars[3] - kmin)*A/(rho*c) ) *1000.*60.,3))+" l/min")
        else:
            print("Region VI, warming up (dTc/dt > 0)")
            print("dTc/dt = "+ str(round(eqvars[4]*3600.,6))+ " K/hour")
    return T

def solve(f,x1,x2,tol,maxIter):
    a  = x1
    b  = x2
    fa = f(a)
    fb = f(b)
    if fa*fb>0.:
        raise SystemExit('wrong initial interval in the root solver')
        return None
    else:
        for i in range(maxIter):
            c  = (a+b)/2.
            fc = f(c)
            if fb*fc > 0. :
                b  = c
                fb = fc
            else:
                a  = c
                fa = fc   
            if abs(a-b) < tol:
                return c
            if i == maxIter-1:
                raise SystemExit('reaching maximum iteration in the root solver')
                return None


# In[481]:


df['HOBO_Tem_K'] = df['HOBO_Tem'] + 273.15
df['PA_Tem_K'] = df['PA_Tem'] + 273.15

df['HOBO_RH_decimal'] = df['HOBO_RH'] / 100.0
df['PA_RH_decimal'] = df['PA_RH'] / 100.0

# Apply the heatindex function
df['HOBO_EHI'] = df.apply(lambda row: heatindex(row['HOBO_Tem_K'], row['HOBO_RH_decimal']), axis=1)
df['PA_EHI'] = df.apply(lambda row: heatindex(row['PA_Tem_K'], row['PA_RH_decimal']), axis=1)
df = df.drop(['HOBO_Tem_K', 'PA_Tem_K','PA_RH_decimal', 'HOBO_RH_decimal'], axis=1)
df.describe()


# In[482]:


# connvert from K to F
df['HOBO_EHI'] = (df['HOBO_EHI'] - 273.15) * 9/5 + 32
df['PA_EHI'] = (df['PA_EHI'] - 273.15) * 9/5 + 32
df.to_csv('/Users/justintse/Desktop/HOBO PA Tem/HOBO_PA_EHI.csv', index=False)


# In[90]:


EHI_df = pd.read_csv('/Users/justintse/Desktop/HOBO PA Tem/HOBO_PA_EHI.csv',low_memory=False)
EHI_df['HOBO_EHI_C'] = (EHI_df['HOBO_EHI'] - 32) * 5/9
EHI_df['PA_EHI_C'] = (EHI_df['PA_EHI'] - 32) * 5/9
EHI_df.describe()


# In[98]:


EHI_monthly_stats = calculate_statistics(EHI_df, group_by_field='Month', target_field='HOBO_EHI_C', 
                                predictor_field='PA_EHI_C', location_id_field='Location ID')
print(EHI_monthly_stats)


# In[99]:


EHI_hourly_stats = calculate_statistics(EHI_df, group_by_field='Hour', target_field='HOBO_EHI_C', 
                                predictor_field='PA_EHI_C', location_id_field='Location ID')
print(EHI_hourly_stats)


# In[101]:


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
font_name = "Arial"
month_acronyms = [calendar.month_abbr[m] for m in range(1, 13)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=600, gridspec_kw={'width_ratios': [7, 3]})

# --- First Plot: RMSE, MAE, and r ---
ax1.plot(month_acronyms, EHI_monthly_stats['RMSE'], marker='o', markersize=10, color='#C2D8B9', label='RMSE', linewidth=2)
ax1.plot(month_acronyms, EHI_monthly_stats['MAE'], marker='s', markersize=10, color='#4A7856', label='MAE', linewidth=2)
ax1.plot(month_acronyms, EHI_monthly_stats['MBE'], marker='x', markersize=10, color='#797B84', label='MBE', linewidth=2)

# Set labels
ax1.set_xlabel('Month', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylabel('Error (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylim(2, 9)
ax1.set_yticks(np.arange(2, 9, 1))

# Customize ticks
ax1.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax1.get_xticklabels(), fontweight='bold', fontname=font_name)
plt.setp(ax1.get_yticklabels(), fontweight='bold', fontname=font_name)

# Second y-axis for correlation coefficient (r)
ax1_secondary = ax1.twinx()
ax1_secondary.plot(month_acronyms, EHI_monthly_stats['R'], marker='^', markersize=10, color='#435E44', label=r'$r$', linewidth=2)
ax1_secondary.set_ylabel(r'$r$', rotation=270, labelpad=20, fontsize=20, fontweight='bold', fontname=font_name)
ax1_secondary.set_ylim(0, 1.2)
ax1_secondary.set_yticks(np.arange(0, 1.2, 0.3))
ax1_secondary.tick_params(axis='y', labelsize=20, width=2)
plt.setp(ax1_secondary.get_yticklabels(), fontweight='bold', fontname=font_name)

# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_secondary.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper left',
           prop={'size': 20, 'weight': 'bold', 'family': font_name},
           frameon=False,
           ncol=2)

# --- Second Plot: HOBO Temperature Trend ---

monthly_data = [EHI_df[EHI_df['Month'] == m]['HOBO_EHI_C'] for m in range(1, 13)] 

for i, data in enumerate(monthly_data):
    # Calculate the quartiles
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    mean_val = np.mean(data)
    min_val, max_val = np.min(data), np.max(data)
    ax2.plot([i+1, i+1], [q25, q75], color='#81b29a', lw=3)
    ax2.scatter(i + 1, mean_val, color='#81b29a', s=50, marker='o')

ax2.set_xlabel('Month', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax2.set_ylabel('HOBO Extended Heat Index (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax2.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax2.get_yticklabels(), fontweight='bold', fontname=font_name)
visible_months = np.arange(1, 13, 2)
ax2.set_xticks(visible_months)
ax2.set_xticklabels([month_acronyms[m-1] for m in visible_months], fontsize=20, fontweight='bold', fontname=font_name)

plt.tight_layout()
plt.show()


# In[102]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=600, gridspec_kw={'width_ratios': [7, 3]})

# --- First Plot: RMSE, MAE, and r ---
ax1.plot(EHI_hourly_stats['RMSE'], marker='o', markersize=10, color='#C2D8B9', label='RMSE', linewidth=2)
ax1.plot(EHI_hourly_stats['MAE'], marker='s', markersize=10, color='#4A7856', label='MAE', linewidth=2)
ax1.plot(EHI_hourly_stats['MBE'], marker='x', markersize=10, color='#797B84', label='MBE', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=2)

# Set labels
ax1.set_xlabel('Hour', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylabel('Error (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)

# Customize ticks
ax1.tick_params(axis='both', labelsize=20, width=2)
plt.setp(ax1.get_yticklabels(), fontweight='bold', fontname=font_name)

visible_hours = np.arange(0, 24, 3)
ax1.set_xticks(visible_hours)
ax1.set_xticklabels([f'{h}:00' for h in visible_hours], fontsize=20, fontweight='bold', fontname=font_name)
ax1.set_ylim(-2, 15)
ax1.set_yticks(np.arange(-2, 15, 2))

# Second y-axis for correlation coefficient (r)
ax1_secondary = ax1.twinx()
ax1_secondary.plot(EHI_hourly_stats['R'], marker='^', markersize=10, color='#435E44', label=r'$r$', linewidth=2)
ax1_secondary.set_ylabel(r'$r$', rotation=270, labelpad=20, fontsize=20, fontweight='bold', fontname=font_name)
ax1_secondary.set_ylim(0, 1.2)
ax1_secondary.set_yticks(np.arange(0, 1.2, 0.3))
ax1_secondary.tick_params(axis='y', labelsize=20, width=2)
plt.setp(ax1_secondary.get_yticklabels(), fontweight='bold', fontname=font_name)

# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_secondary.get_legend_handles_labels()
# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_secondary.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper left',
           prop={'size': 20, 'weight': 'bold', 'family': font_name},
           frameon=False,
           ncol=2)

# --- Second Plot: HOBO Temperature Trend ---

hourly_data = [EHI_df[EHI_df['Hour'] == h]['HOBO_EHI_C'] for h in range(24)]

for i, data in enumerate(hourly_data):
    # Calculate the quartiles
    mean_val = np.mean(data)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    min_val, max_val = np.min(data), np.max(data)
    ax2.plot([i+1, i+1], [q25, q75], color='#81b29a', lw=3)
    ax2.scatter(i + 1, mean_val, color='#81b29a', s=50, marker='o')

# Set labels
ax2.set_xlabel('Hour', labelpad=12, fontsize=20, fontweight='bold', fontname=font_name)
ax2.set_ylabel('HOBO Extended Heat Index (°C)', labelpad=15, fontsize=20, fontweight='bold', fontname=font_name)
ax2.tick_params(axis='both', labelsize=16, width=2)
plt.setp(ax2.get_yticklabels(), fontweight='bold', fontname=font_name)

visible_hours = np.arange(0, 24, 6)
ax2.set_xticks(visible_hours + 1)
ax2.set_xticklabels([f'{h}:00' for h in visible_hours], fontsize=20, fontweight='bold', fontname=font_name)

plt.tight_layout()
plt.show()


# In[194]:


# Define the heat class
def classify_heat_index(ehi_value):
    if ehi_value >= 125:
        return 'Extreme Danger'
    elif ehi_value >= 103:
        return 'Danger'
    elif ehi_value >= 90:
        return 'Extreme Caution'
    elif ehi_value >= 80:
        return 'Caution'
    else:
        return 'None'

EHI_df['HOBO_Heat_Class'] = EHI_df['HOBO_EHI'].apply(classify_heat_index)
EHI_df['PA_Heat_Class'] = EHI_df['PA_EHI'].apply(classify_heat_index)

hobo_counts = EHI_df['HOBO_Heat_Class'].value_counts()
pa_counts = EHI_df['PA_Heat_Class'].value_counts()
total_count = len(EHI_df)

hobo_percentages = (hobo_counts / total_count) * 100 
pa_percentages = (pa_counts / total_count) * 100


heat_class_per = pd.DataFrame({
    'Heat Class': ['Caution', 'Extreme Caution', 'Danger', 'Extreme Danger'],
    'HOBO Percentage': [
        hobo_percentages.get('Caution', 0), 
        hobo_percentages.get('Extreme Caution', 0), 
        hobo_percentages.get('Danger', 0), 
        hobo_percentages.get('Extreme Danger', 0)
    ],
    'PA Percentage': [
        pa_percentages.get('Caution', 0), 
        pa_percentages.get('Extreme Caution', 0), 
        pa_percentages.get('Danger', 0), 
        pa_percentages.get('Extreme Danger', 0)
    ]
})

print(heat_class_per)


# # Master Dataset

# In[77]:


master_df = pd.merge(solar_df, TCEQ_WNDS, on=['Timestamp'], how='left')
master_df = pd.merge(master_df, TCEQ_Tem, on=['Timestamp'], how='left')
master_df = master_df.drop(columns=['HOBO_ID', 'hour', 'day', 'PM1.0 (ATM)', 'PM2.5 (ATM)',
                       'PM10.0 (ATM)', 'Date', 'Month', 'Hour', 'HOBO_Device_ID', 'PA_ID'])
master_df.columns


# In[81]:


master_df.to_csv('/Users/justintse/Desktop/HOBO PA Tem/master_df.csv', index=False)

