# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:47:57 2020

@author: LMurphy
"""

# Import packages and libraries for latter usage.
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing



# ### Import the data files
sample = pd.read_excel(r'C:\Users\lmurphy\OneDrive - Three\Desktop\Data Science MSc\Dissertation and Project\Project\Data\SIM_TO_HS_20210108.xlsx')
sample.head(4)

sample.info()


###############################################################################

### Check for cardinality

max_cardinality = 100
high_cardinality = [col for col in sample.select_dtypes(exclude=np.number)
                   if sample[col].nunique() > max_cardinality]
sample2 = sample.drop(columns=high_cardinality)
sample2.info()

### All fine

###############################################################################

### Summary stats on numerics
stats = sample.describe().astype('int64')

### Correlation matrix
df = sample[['age',
    'ALLOC_AYCE_DATA',
    'ALLOC_DATA',
    'ALLOC_VOICE',
    'Amortisation',
    'BEST_CUSTOMER_SCORE',
    'bill_prod_MRC',
    'bill_prod_MRC_rpi',
    'CONGESTED_CALL_BAND_4G_1800',
    'CONGESTED_CALL_BAND_4G_800',
    'CONGESTED_DATA_BAND_4G_1800',
    'CONGESTED_DATA_BAND_4G_800',
    'COVERAGE_BAND_3G',
    'COVERAGE_BAND_4G_1800',
    'COVERAGE_BAND_4G_800',
    'CRM_MRC',
    'CRM_MRC_rpi',
    'DATA_ALLOC_50PC_DAYS_MIN',
    'Dev_1_3yr_LT_500',
    'Dev_1yr_LT_500',
    'Dev_GT_3yr_GT_100',
    'Dev_LT_3yr_GT_500',
    'EOCN_MRC',
    'estimated_cogs',
    'final_MRC',
    'Handset_cost',
    'hs_age_mths',
    'ifrs_ubd',
    'INFLUENCE_SCORE',
    'INVESTMENT_SCORE',
    'LM_call_10',
    'LM_call_30',
    'LM_call_drop',
    'LM_COV_3G_BEST',
    'LM_COV_4G_1800_BEST',
    'LM_COV_4G_800_BEST',
    'LM_data_10',
    'LM_everday_call_10',
    'LM_everday_call_30',
    'LM_everday_data_10',
    'LM_everday_data_30',
    'LM_sms_10',
    'LM_sms_30',
    'LM_sms_drop',
    'LM_top1_ftg',
    'LM_top1_inl',
    'LM_top1_ret',
    'LM_top1_upg',
    'LM_top2_ftg',
    'LM_top2_inl',
    'LOYALITY_SCORE',
    'M3_TOT_DATA_MB',
    'MAX_DL_PEAK_THROUGPUT_3G',
    'MAX_DL_PEAK_THROUGPUT_4G_1800',
    'MAX_DL_PEAK_THROUGPUT_4G_800',
    'MINS_TO_VMAIL_BAND',
    'MINS_TO_VMAIL_RATE',
    'net_total_cost',
    'NMRC',
    'NMRC_90',
    'OOCR_3G',
    'SIZE_OF_INFLUENCE',
    'std_voice_time30',
    'tenure',
    'total_voice_time30'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.5)
sns.heatmap(corrMatrix, annot=False, cmap='Reds')



###############################################################################



### Get dummies for categorical inputs

#sample = pd.concat([sample, pd.get_dummies(sample['adv_segment'],prefix='adv_seg',prefix_sep='_')], axis=1)
#sample = pd.concat([sample, pd.get_dummies(sample['basic_segment'],prefix='basic_seg',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['BEST_CUSTOMER_FLAG'],prefix='bcust',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['change_in_device'],prefix='dev_chg',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['device_segment'],prefix='dev_seg',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['FIRST_CONTRACT'],prefix='first_cont',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['grouped_seg'],prefix='xxxxxxx',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['hset_age'],prefix='hest_age',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['hset_cogs'],prefix='hset_cogs',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['INFLUENCE_GROUP'],prefix='infl_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['INVESTMENT_GROUP'],prefix='inv_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['LM_acq_channel'],prefix='acq_chn',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['LM_netw_dev_grp'],prefix='dev_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['LM_upg_channel'],prefix='upg_chn',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['LOYALITY_GROUP'],prefix='loyl_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['Mosaic_UK6_GROUP_char'],prefix='mosaic_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['new_segment'],prefix='nseseg',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['pre_alloc'],prefix='pre_all',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['prev_data_tier'],prefix='prev_tier',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['prev_device_type'],prefix='prev_dev',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['useg'],prefix='useg',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['VALUE_GROUP'],prefix='val_grp',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['dev_make'],prefix='dev_make',prefix_sep='_')], axis=1)
sample = pd.concat([sample, pd.get_dummies(sample['device_seg'],prefix='device_seg',prefix_sep='_')], axis=1)

sample.head(4)
sample.info()


###############################################################################



# ### Data Normalisation:

sample['age_NML'] = preprocessing.scale(sample[['age']])
sample['total_voice_time30_NML'] = preprocessing.scale(sample[['total_voice_time30']])
sample['std_voice_time30_NML'] = preprocessing.scale(sample[['std_voice_time30']])
sample['M3_TOT_DATA_MB_NML'] = preprocessing.scale(sample[['M3_TOT_DATA_MB']])
sample['bill_prod_MRC_NML'] = preprocessing.scale(sample[['bill_prod_MRC']])
sample['CRM_MRC_NML'] = preprocessing.scale(sample[['CRM_MRC']])
#sample['CRM_MRC_rpi_NML'] = preprocessing.scale(sample[['CRM_MRC_rpi']])
sample['bill_prod_MRC_rpi_NML'] = preprocessing.scale(sample[['bill_prod_MRC_rpi']])
sample['EOCN_MRC_NML'] = preprocessing.scale(sample[['EOCN_MRC']])
sample['final_MRC_NML'] = preprocessing.scale(sample[['final_MRC']])
sample['net_total_cost_NML'] = preprocessing.scale(sample[['net_total_cost']])
sample['ifrs_ubd_NML'] = preprocessing.scale(sample[['ifrs_ubd']])
sample['Handset_cost_NML'] = preprocessing.scale(sample[['Handset_cost']])
sample['NMRC_NML'] = preprocessing.scale(sample[['NMRC']])
sample['NMRC_90_NML'] = preprocessing.scale(sample[['NMRC_90']])
sample['OOCR_3G_NML'] = preprocessing.scale(sample[['OOCR_3G']])
sample['MAX_DL_PEAK_THROUGPUT_3G_NML'] = preprocessing.scale(sample[['MAX_DL_PEAK_THROUGPUT_3G']])
sample['MAX_DL_PEAK_THROUGPUT_4G_1800_NML'] = preprocessing.scale(sample[['MAX_DL_PEAK_THROUGPUT_4G_1800']])
sample['MAX_DL_PEAK_THROUGPUT_4G_800_NML'] = preprocessing.scale(sample[['MAX_DL_PEAK_THROUGPUT_4G_800']])
sample['MINS_TO_VMAIL_RATE_NML'] = preprocessing.scale(sample[['MINS_TO_VMAIL_RATE']])
sample['SIZE_OF_INFLUENCE_NML'] = preprocessing.scale(sample[['SIZE_OF_INFLUENCE']])
sample['estimated_cogs_NML'] = preprocessing.scale(sample[['estimated_cogs']])
sample['hs_age_mths_NML'] = preprocessing.scale(sample[['hs_age_mths']])

sample.head(4)
sample.info()



###############################################################################



### Drop data section

sample = sample.drop([
    'ban',
#    'adv_segment',
#    'basic_segment',
    'BEST_CUSTOMER_FLAG',
    'change_in_device',
    'data_tier',
    'device_segment',
    'FIRST_CONTRACT',
    'grouped_seg',
    'hset_age',
    'hset_cogs',
    'INFLUENCE_GROUP',
    'INVESTMENT_GROUP',
    'latest_device',
    'LM_acq_channel',
    'LM_netw_dev_grp',
    'LM_upg_channel',
    'LOYALITY_GROUP',
    'Mosaic_UK6_GROUP_char',
    'new_segment',
    'pre_alloc',
    'prev_data_tier',
    'prev_device_type',
    'useg',
    'VALUE_GROUP',
    'dev_make',
    'age',
    'total_voice_time30',
    'std_voice_time30',
    'M3_TOT_DATA_MB',
    'bill_prod_MRC',
    'CRM_MRC',
#    'CRM_MRC_rpi',
    'bill_prod_MRC_rpi',
    'EOCN_MRC',
    'final_MRC',
    'net_total_cost',
    'ifrs_ubd',
    'Handset_cost',
    'NMRC',
    'NMRC_90',
    'OOCR_3G',
    'MAX_DL_PEAK_THROUGPUT_3G',
    'MAX_DL_PEAK_THROUGPUT_4G_1800',
    'MAX_DL_PEAK_THROUGPUT_4G_800',
    'MINS_TO_VMAIL_RATE',
    'SIZE_OF_INFLUENCE',
    'estimated_cogs',
    'hs_age_mths',
    'device_seg',
    'Zero_use',
    'Test_Control'
    ], axis=1)




# create summary statistics for the numeric attributes (age, BMI, children and charges).
sample.describe()



sample.to_csv('C:/Users/lmurphy/OneDrive - Three/Desktop/Data Science MSc/Dissertation and Project/Project/Data/Sample_prepped.csv', index=False)

    
    
#############################################################################
#############################################################################
#########################  E-N-D  O-F  P-R-O-G-R-A-M ########################
#############################################################################
#############################################################################
