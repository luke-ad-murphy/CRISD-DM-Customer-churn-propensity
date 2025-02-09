# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:28:32 2020

@author: LMurphy
"""

### Libraries
import numpy as np
import pandas as pd


import sys
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#########################################################################


### Set some output options
# options to limit the number of decimal places displayed to 3.
pd.options.display.float_format = "{:.3f}".format
# Set iPython's max row display to 1000.
pd.set_option('display.max_row', 1000)
# Set iPython's max column width to 50.
pd.set_option('display.max_columns', 50)


#########################################################################

### Import the data files
sample = pd.read_excel(r'C:\Users\lmurphy\OneDrive - Three\Desktop\Data Science MSc\Dissertation and Project\Project\Data\SIM_TO_HS_SAMP.xlsx')
sample.head(4)


#########################################################################

#######################
### VISUALISATIONS  ###
#######################

#########################################################################

### BAR CHARTS

# example
#df_suic.groupby('generation').agg({'pop_1m' : 'sum'}).plot(title='Population by generation',kind='bar',legend=False)
#plt.ylabel('Population in millions')

#sample['target'].value_counts().plot(kind='bar')
#plt.legend(loc='upper right')


def bar_chart(var):
    sample[var].value_counts().plot(kind='bar')
    plt.legend(loc='upper right')

bar_chart('target')

bar_chart('adv_segment')
bar_chart('ALLOC_AYCE_DATA')
bar_chart('ALLOC_AYCE_VOICE')
bar_chart('ALLOC_DATA')
bar_chart('ALLOC_VOICE')
bar_chart('B_CONTRACT_LENGTH')
bar_chart('B_pre_upge_tariff')
bar_chart('basic_segment')
bar_chart('BEST_CUSTOMER_FLAG')
bar_chart('BEST_CUSTOMER_SCORE')
bar_chart('C_CONTRACT_LENGTH')
bar_chart('C_post_upge_tariff')
bar_chart('C_pre_upge_tariff')
bar_chart('change_in_device')
bar_chart('CONGESTED_CALL_BAND_3G')
bar_chart('CONGESTED_CALL_BAND_4G_1800')
bar_chart('CONGESTED_CALL_BAND_4G_800')
bar_chart('CONGESTED_DATA_BAND_3G')
bar_chart('CONGESTED_DATA_BAND_4G_1800')
bar_chart('CONGESTED_DATA_BAND_4G_800')
bar_chart('COVERAGE_BAND_3G')
bar_chart('COVERAGE_BAND_4G_1800')
bar_chart('COVERAGE_BAND_4G_800')
bar_chart('credit_checked_sim_customer')
bar_chart('data_tier')
bar_chart('DCR_BAND_3G')
bar_chart('DCR_BAND_4G_1800')
bar_chart('DCR_BAND_4G_800')
bar_chart('Dev_1_3yr_LT_500')
bar_chart('Dev_1yr_LT_500')
bar_chart('Dev_GT_3yr_GT_100')
bar_chart('Dev_LT_3yr_GT_500')
bar_chart('device_segment')
bar_chart('DL_70PCNT_THRESHOLD_BAND_3G')
bar_chart('DL_70PCNT_THRESHOLD_BAND_4G_1800')
bar_chart('DL_70PCNT_THRESHOLD_BAND_4G_800')
bar_chart('DL_PEAK_THROUGPUT_BAND_3G')
bar_chart('DL_PEAK_THROUGPUT_BAND_4G_1800')
bar_chart('DL_PEAK_THROUGPUT_BAND_4G_800')
bar_chart('DL_RETRANS_BAND_3G')
bar_chart('DL_RETRANS_BAND_4G_1800')
bar_chart('DL_RETRANS_BAND_4G_800')
bar_chart('FIRST_CONTRACT')
bar_chart('grouped_seg')
bar_chart('homesignal_flag')
bar_chart('hset_age')
bar_chart('hset_cogs')
bar_chart('INFLUENCE_GROUP')
bar_chart('INFLUENCE_SCORE')
bar_chart('INSURANCE_PRODUCT')
bar_chart('INSURANCE_PRODUCT_NAME')
bar_chart('INVESTMENT_GROUP')
bar_chart('INVESTMENT_SCORE')
bar_chart('ivr_cancel')
bar_chart('ivr_upgrade')
bar_chart('LM_acq_channel')
bar_chart('LM_call_10')
bar_chart('LM_call_30')
bar_chart('LM_call_drop')
bar_chart('LM_COV_3G_BEST')
bar_chart('LM_COV_4G_1800_BEST')
bar_chart('LM_COV_4G_800_BEST')
bar_chart('LM_data_10')
bar_chart('LM_data_30')
bar_chart('LM_data_drop')
bar_chart('LM_everday_call_10')
bar_chart('LM_everday_call_30')
bar_chart('LM_everday_data_10')
bar_chart('LM_everday_data_30')
bar_chart('LM_everday_sms_10')
bar_chart('LM_everday_sms_30')
bar_chart('LM_netw_dev')
bar_chart('LM_netw_dev_grp')
bar_chart('LM_sms_10')
bar_chart('LM_sms_30')
bar_chart('LM_sms_drop')
bar_chart('LM_top1_acq_m1')
bar_chart('LM_top1_acq_m2')
bar_chart('LM_top1_acq_m3')
bar_chart('LM_top1_chn')
bar_chart('LM_top1_chn_m1')
bar_chart('LM_top1_chn_m2')
bar_chart('LM_top1_chn_m3')
bar_chart('LM_top1_cpp')
bar_chart('LM_top1_ftg')
bar_chart('LM_top1_inl')
bar_chart('LM_top1_ret')
bar_chart('LM_top1_upg')
bar_chart('LM_top1_upg_m1')
bar_chart('LM_top1_upg_m2')
bar_chart('LM_top1_upg_m3')
bar_chart('LM_top2_acq_m1')
bar_chart('LM_top2_acq_m2')
bar_chart('LM_top2_acq_m3')
bar_chart('LM_top2_chn')
bar_chart('LM_top2_chn_m1')
bar_chart('LM_top2_chn_m2')
bar_chart('LM_top2_chn_m3')
bar_chart('LM_top2_cpp')
bar_chart('LM_top2_ftg')
bar_chart('LM_top2_inl')
bar_chart('LM_top2_ret')
bar_chart('LM_top2_upg')
bar_chart('LM_top2_upg_m1')
bar_chart('LM_top2_upg_m2')
bar_chart('LM_top2_upg_m3')
bar_chart('LM_upg_channel')
bar_chart('LOYALITY_GROUP')
bar_chart('LOYALITY_SCORE')
bar_chart('M1_USAGE_FLAG')
bar_chart('M2_USAGE_FLAG')
bar_chart('M3_USAGE_FLAG')
bar_chart('MINS_TO_VMAIL_BAND')
bar_chart('Mosaic_UK6_GROUP_char')
bar_chart('new_segment')
bar_chart('OOC_BAND_3G')
bar_chart('OOC_BAND_4G_1800')
bar_chart('OOC_BAND_4G_800')
bar_chart('PAGE_LOAD_ERROR_BAND_3G')
bar_chart('PAGE_LOAD_ERROR_BAND_4G_1800')
bar_chart('PAGE_LOAD_ERROR_BAND_4G_800')
bar_chart('pre_alloc')
bar_chart('prev_data_tier')
bar_chart('prev_device_type')
bar_chart('RADIO_FAILURE_BAND_3G')
bar_chart('RADIO_FAILURE_BAND_4G_1800')
bar_chart('RADIO_FAILURE_BAND_4G_800')
bar_chart('REDIAL_BAND_3G')
bar_chart('REDIAL_BAND_4G_1800')
bar_chart('REDIAL_BAND_4G_800')
bar_chart('REDIALS_BAND_VOLTE')
bar_chart('s2h_upgrade_flag')
bar_chart('SIZE_OF_INFLUENCE')
bar_chart('THREE_MTH_USAGE_FLAG')
bar_chart('UL_AVG_RTT_BAND_3G')
bar_chart('UL_AVG_RTT_BAND_4G_1800')
bar_chart('UL_AVG_RTT_BAND_4G_800')
bar_chart('useg')
bar_chart('VALUE_GROUP')
bar_chart('Zero_use')
bar_chart('CONGESTED_CALL_RATE_3G')
bar_chart('CONGESTED_CALL_RATE_4G_1800')
bar_chart('CONGESTED_CALL_RATE_4G_800')
bar_chart('CONGESTED_DATA_RATE_3G')
bar_chart('CPP_6M_COUNT')
bar_chart('DATA_ALLOC_50PC_DAYS_count')
bar_chart('DATA_ALLOC_75PC_DAYS_count')
bar_chart('DATA_ALLOC_90PC_DAYS_count')
bar_chart('DATA_ALLOC_FULL_CONS_DAYS_count')
bar_chart('DCR_4G_1800')
bar_chart('DCR_4G_800')
bar_chart('DL_70PCNT_THRESHOLD_3G')
bar_chart('DL_70PCNT_THRESHOLD_4G_1800')
bar_chart('DL_70PCNT_THRESHOLD_4G_800')
bar_chart('DL_RETRANS_DAYS_3G')
bar_chart('DL_RETRANS_DAYS_4G_1800')
bar_chart('DL_RETRANS_DAYS_4G_800')
bar_chart('DL_RETRANS_RATE_3G')
bar_chart('DL_RETRANS_RATE_4G_1800')
bar_chart('DL_RETRANS_RATE_4G_800')
bar_chart('DL_RETRANS_RATE_5PCT_3G')
bar_chart('DL_RETRANS_RATE_5PCT_4G_1800')
bar_chart('DL_RETRANS_RATE_5PCT_4G_800')
bar_chart('hh_accs_closed')
bar_chart('hh_accs_opened')
bar_chart('hh_accs_upgrade')
bar_chart('hh_cpp_acc')
bar_chart('HH_MBB')
bar_chart('hh_mbb_acc_closed_m1')
bar_chart('hh_mbb_acc_closed_m2')
bar_chart('hh_mbb_acc_closed_m3')
bar_chart('HH_MBB_acc_FTG')
bar_chart('HH_MBB_acc_inlife')
bar_chart('HH_MBB_acc_ret_window')
bar_chart('hh_mbb_acc_upgrade_m1')
bar_chart('hh_mbb_acc_upgrade_m2')
bar_chart('hh_mbb_acc_upgrade_m3')
bar_chart('hh_MBB_accs_closed')
bar_chart('hh_MBB_accs_opened')
bar_chart('hh_MBB_accs_upgrade')
bar_chart('hh_new_MBB_acc_m1')
bar_chart('hh_new_MBB_acc_m2')
bar_chart('hh_new_MBB_acc_m3')
bar_chart('hh_new_voi_acc_m1')
bar_chart('hh_new_voi_acc_m2')
bar_chart('hh_new_voi_acc_m3')
bar_chart('hh_voi_acc_closed_m1')
bar_chart('hh_voi_acc_closed_m2')
bar_chart('hh_voi_acc_closed_m3')
bar_chart('HH_Voi_acc_FTG')
bar_chart('HH_Voi_acc_inlife')
bar_chart('HH_Voi_acc_ret_window')
bar_chart('hh_voi_acc_upgrade_m1')
bar_chart('hh_voi_acc_upgrade_m2')
bar_chart('hh_voi_acc_upgrade_m3')
bar_chart('hh_voi_accs_closed')
bar_chart('hh_voi_accs_opened')
bar_chart('hh_voi_accs_upgrade')
bar_chart('HH_Voice')
bar_chart('HOMESIGNAL_DATE_LREC')
bar_chart('int_bill_enq_m1')
bar_chart('int_bill_enq_m2')
bar_chart('int_bill_enq_m3')
bar_chart('int_cant_pay_m1')
bar_chart('int_cant_pay_m2')
bar_chart('int_cant_pay_m3')
bar_chart('int_cc_m1')
bar_chart('int_cc_m2')
bar_chart('int_cc_m3')
bar_chart('int_comp_better_offer_m1')
bar_chart('int_comp_better_offer_m2')
bar_chart('int_comp_better_offer_m3')
bar_chart('int_complaint_m1')
bar_chart('int_complaint_m2')
bar_chart('int_complaint_m3')
bar_chart('int_device_loststol_m1')
bar_chart('int_device_loststol_m2')
bar_chart('int_device_loststol_m3')
bar_chart('int_make_payment_m1')
bar_chart('int_make_payment_m2')
bar_chart('int_make_payment_m3')
bar_chart('int_my3_m1')
bar_chart('int_my3_m2')
bar_chart('int_my3_m3')
bar_chart('int_network_m1')
bar_chart('int_network_m2')
bar_chart('int_network_m3')
bar_chart('int_new_ph_con_m1')
bar_chart('int_new_ph_con_m2')
bar_chart('int_new_ph_con_m3')
bar_chart('int_offers_enq_m1')
bar_chart('int_offers_enq_m2')
bar_chart('int_offers_enq_m3')
bar_chart('int_online_m1')
bar_chart('int_online_m2')
bar_chart('int_online_m3')
bar_chart('int_outcm_case_raised_m1')
bar_chart('int_outcm_case_raised_m2')
bar_chart('int_outcm_case_raised_m3')
bar_chart('int_outcm_consid_offer_m1')
bar_chart('int_outcm_consid_offer_m2')
bar_chart('int_outcm_consid_offer_m3')
bar_chart('int_outcm_pac_given_m1')
bar_chart('int_outcm_pac_given_m2')
bar_chart('int_outcm_pac_given_m3')
bar_chart('int_outcm_resolved_m1')
bar_chart('int_outcm_resolved_m2')
bar_chart('int_outcm_resolved_m3')
bar_chart('int_refund_enq_m1')
bar_chart('int_refund_enq_m2')
bar_chart('int_refund_enq_m3')
bar_chart('int_tariff_enq_m1')
bar_chart('int_tariff_enq_m2')
bar_chart('int_tariff_enq_m3')
bar_chart('int_tech_enq_m1')
bar_chart('int_tech_enq_m2')
bar_chart('int_tech_enq_m3')
bar_chart('int_want_better_deal_m1')
bar_chart('int_want_better_deal_m2')
bar_chart('int_want_better_deal_m3')
bar_chart('int_want_cancel_m1')
bar_chart('int_want_cancel_m2')
bar_chart('int_want_cancel_m3')
bar_chart('int_want_dat_addon_m1')
bar_chart('int_want_dat_addon_m2')
bar_chart('int_want_dat_addon_m3')
bar_chart('int_want_pac_m1')
bar_chart('int_want_pac_m2')
bar_chart('int_want_pac_m3')
bar_chart('int_want_upgrade_m1')
bar_chart('int_want_upgrade_m2')
bar_chart('int_want_upgrade_m3')
bar_chart('int_wont_pay_m1')
bar_chart('int_wont_pay_m2')
bar_chart('int_wont_pay_m3')
bar_chart('ml_ACCOUNTS')
bar_chart('ml_accs_closed')
bar_chart('ml_accs_opened')
bar_chart('ml_accs_upgrade')
bar_chart('ml_cpp_acc')
bar_chart('ml_MBB')
bar_chart('ml_mbb_acc_closed_m1')
bar_chart('ml_mbb_acc_closed_m2')
bar_chart('ml_mbb_acc_closed_m3')
bar_chart('ml_MBB_acc_FTG')
bar_chart('ml_MBB_acc_inlife')
bar_chart('ml_MBB_acc_ret_window')
bar_chart('ml_mbb_acc_upgrade_m1')
bar_chart('ml_mbb_acc_upgrade_m2')
bar_chart('ml_mbb_acc_upgrade_m3')
bar_chart('ml_MBB_accs_closed')
bar_chart('ml_MBB_accs_opened')
bar_chart('ml_MBB_accs_upgrade')
bar_chart('ml_new_MBB_acc_m1')
bar_chart('ml_new_MBB_acc_m2')
bar_chart('ml_new_MBB_acc_m3')
bar_chart('ml_new_voi_acc_m1')
bar_chart('ml_new_voi_acc_m2')
bar_chart('ml_new_voi_acc_m3')
bar_chart('ml_voi_acc_closed_m1')
bar_chart('ml_voi_acc_closed_m2')
bar_chart('ml_voi_acc_closed_m3')
bar_chart('ml_Voi_acc_FTG')
bar_chart('ml_Voi_acc_inlife')
bar_chart('ml_Voi_acc_ret_window')
bar_chart('ml_voi_acc_upgrade_m1')
bar_chart('ml_voi_acc_upgrade_m2')
bar_chart('ml_voi_acc_upgrade_m3')
bar_chart('ml_voi_accs_closed')
bar_chart('ml_voi_accs_opened')
bar_chart('ml_voi_accs_upgrade')
bar_chart('ml_Voice')
bar_chart('no_of_renewals')
bar_chart('OOCR_4G_1800')
bar_chart('OOCR_4G_800')
bar_chart('REDIALS_4G_1800')
bar_chart('REDIALS_4G_800')
bar_chart('renewal_12m')
bar_chart('renewal_1m')
bar_chart('renewal_24m')
bar_chart('VOICE_ALLOC_50PC_DAYS_count')
bar_chart('VOICE_ALLOC_50PC_DAYS_MIN')
bar_chart('VOICE_ALLOC_75PC_DAYS_count')
bar_chart('VOICE_ALLOC_75PC_DAYS_MIN')
bar_chart('VOICE_ALLOC_90PC_DAYS_count')
bar_chart('VOICE_ALLOC_90PC_DAYS_MIN')
bar_chart('VOICE_ALLOC_FULL_CONS_DAYS_count')
bar_chart('VOICE_ALLOC_FULL_CONS_DAYS_MIN')



#########################################################################

# CORRELATION MATRIX FOR SELECTED INPUTS

# example
#df = insur2[['charges', 'age', 'bmi', 'children', 'male_yes', 'smoker_yes',
#             'children_yes', 'east_yes', 'northeast', 'northwest',
#             'southeast', 'southwest', 'Obese', 'Chronic_obese',
#             'age_cat_ord', 'weight_ord']]
#corrMatrix = df.corr()
#sns.set(font_scale=0.8)
#sns.heatmap(corrMatrix, annot=True, cmap='Blues', fmt='d')
#plt.show()


df = sample[['age',
             'Amortisation',
             'amortisation_90',
             'avg_of_premium_calls30',
             'avg_perday_amount_of_data30',
             'avg_perday_voice_time30',
             'avg_unique_mob_num_dial30',
             'avg_unique_num_dial30',
             'bill_prod_MRC',
             'bill_prod_MRC_rpi',
             'cal_mon_usage_M1',
             'cal_mon_usage_M2',
             'cal_mon_usage_M3',
             'count_total_1m',
             'count_total_3m',
             'CRM_MRC',
             'CRM_MRC_rpi',
             'EOCN_DATA_USAGE_MB',
             'EOCN_MRC',
             'final_MRC',
             'Handset_cost',
             'ifrs_ubd',
             'invoice_MRC',
             'M1_GOBINGE_DATA_MB',
             'M1_MAIN_DATA_MB',
             'M1_TOT_DATA_MB',
             'M2_GOBINGE_DATA_MB',
             'M2_MAIN_DATA_MB',
             'M2_TOT_DATA_MB',
             'M3_GOBINGE_DATA_MB',
             'M3_MAIN_DATA_MB',
             'M3_TOT_DATA_MB',
             'msf',
             'net_total_cost',
             'NMRC',
             'NMRC_90',
             'nw_speed_months',
             'std_data_used30',
             'std_voice_time30',
             'tenure',
             'THREE_MTH_AVE_DATA_MB',
             'Three_mth_GoBinge_data_MB',
             'Three_MTH_MAIN_DATA_MB',
             'THREE_MTH_TOT_DATA_MB',
             'tot_16mb_dlspeed_1m',
             'tot_16mb_dlspeed_3m',
             'tot_1kb_dlspeed_1m',
             'tot_1kb_dlspeed_3m',
             'tot_1mb_dlspeed_1m',
             'tot_1mb_dlspeed_3m',
             'tot_200kb_dlspeed_1m',
             'tot_200kb_dlspeed_3m',
             'tot_2mb_dlspeed_1m',
             'tot_2mb_dlspeed_3m',
             'tot_400kb_dlspeed_1m',
             'tot_400kb_dlspeed_3m',
             'tot_4mb_dlspeed_1m',
             'tot_4mb_dlspeed_3m',
             'tot_600kb_dlspeed_1m',
             'tot_600kb_dlspeed_3m',
             'tot_800kb_dlspeed_1m',
             'tot_800kb_dlspeed_3m',
             'tot_8mb_dlspeed_1m',
             'tot_8mb_dlspeed_3m',
             'total_amnt_of_fah_roam_data30',
             'total_amount_of_data30',
             'total_cnt_of_fah_calls30',
             'total_cnt_of_molo_calls30',
             'total_cnt_of_on_net_calls30',
             'total_cnt_of_prem_calls30',
             'total_cnt_of_voice_calls30',
             'total_fah_sms30',
             'total_fah_time30',
             'total_internat_roaming_sms30',
             'total_international_sms30',
             'total_molo_time30',
             'total_out_of_bundle_time30',
             'total_sms30',
             'total_time_to_onnet30',
             'total_time_to_prem_numb30',
             'total_upfront_charge',
             'total_voice_time30',
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.8)
sns.heatmap(corrMatrix, annot=False, cmap='Blues')



#########################################################################

### BOXPLOTS

# example
smokers = insur.loc[insur.smoker == 'yes']

#def smokers_boxplot(cat):
#    sns.boxplot(x=cat, y='charges',  data=smokers, palette=pal, fliersize=0)
#    sns.stripplot(x=cat, y='charges', data=smokers,
#                  jitter=True, dodge=True, linewidth=0.5, palette=pal)
#    plt.legend(loc='upper left')

#smokers_boxplot('age_cat')

sns.set_style('white')
pal = sns.color_palette('Paired')

def sample_boxplot(cat):
    sns.boxplot(x='target', y=cat,  data=sample, palette=pal, fliersize=0)
    sns.stripplot(x='target', y=cat, data=sample,
                  jitter=True, dodge=True, linewidth=0.5, palette=pal)

sample_boxplot('age')
sample_boxplot('Amortisation')
sample_boxplot('amortisation_90')
sample_boxplot('avg_of_premium_calls30')
sample_boxplot('avg_perday_amount_of_data30')
sample_boxplot('avg_perday_voice_time30')
sample_boxplot('avg_unique_mob_num_dial30')
sample_boxplot('avg_unique_num_dial30')
sample_boxplot('bill_prod_MRC')
sample_boxplot('bill_prod_MRC_rpi')
sample_boxplot('cal_mon_usage_M1')
sample_boxplot('cal_mon_usage_M2')
sample_boxplot('cal_mon_usage_M3')
sample_boxplot('count_total_1m')
sample_boxplot('count_total_3m')
sample_boxplot('CRM_MRC')
sample_boxplot('CRM_MRC_rpi')
sample_boxplot('EOCN_DATA_USAGE_MB')
sample_boxplot('EOCN_MRC')
sample_boxplot('final_MRC')
sample_boxplot('Handset_cost')
sample_boxplot('ifrs_ubd')
sample_boxplot('invoice_MRC')
sample_boxplot('M1_GOBINGE_DATA_MB')
sample_boxplot('M1_MAIN_DATA_MB')
sample_boxplot('M1_TOT_DATA_MB')
sample_boxplot('M2_GOBINGE_DATA_MB')
sample_boxplot('M2_MAIN_DATA_MB')
sample_boxplot('M2_TOT_DATA_MB')
sample_boxplot('M3_GOBINGE_DATA_MB')
sample_boxplot('M3_MAIN_DATA_MB')
sample_boxplot('M3_TOT_DATA_MB')
sample_boxplot('msf')
sample_boxplot('net_total_cost')
sample_boxplot('NMRC')
sample_boxplot('NMRC_90')
sample_boxplot('nw_speed_months')
sample_boxplot('std_data_used30')
sample_boxplot('std_voice_time30')
sample_boxplot('tenure')
sample_boxplot('THREE_MTH_AVE_DATA_MB')
sample_boxplot('Three_mth_GoBinge_data_MB')
sample_boxplot('Three_MTH_MAIN_DATA_MB')
sample_boxplot('THREE_MTH_TOT_DATA_MB')
sample_boxplot('tot_16mb_dlspeed_1m')
sample_boxplot('tot_16mb_dlspeed_3m')
sample_boxplot('tot_1kb_dlspeed_1m')
sample_boxplot('tot_1kb_dlspeed_3m')
sample_boxplot('tot_1mb_dlspeed_1m')
sample_boxplot('tot_1mb_dlspeed_3m')
sample_boxplot('tot_200kb_dlspeed_1m')
sample_boxplot('tot_200kb_dlspeed_3m')
sample_boxplot('tot_2mb_dlspeed_1m')
sample_boxplot('tot_2mb_dlspeed_3m')
sample_boxplot('tot_400kb_dlspeed_1m')
sample_boxplot('tot_400kb_dlspeed_3m')
sample_boxplot('tot_4mb_dlspeed_1m')
sample_boxplot('tot_4mb_dlspeed_3m')
sample_boxplot('tot_600kb_dlspeed_1m')
sample_boxplot('tot_600kb_dlspeed_3m')
sample_boxplot('tot_800kb_dlspeed_1m')
sample_boxplot('tot_800kb_dlspeed_3m')
sample_boxplot('tot_8mb_dlspeed_1m')
sample_boxplot('tot_8mb_dlspeed_3m')
sample_boxplot('total_amnt_of_fah_roam_data30')
sample_boxplot('total_amount_of_data30')
sample_boxplot('total_cnt_of_fah_calls30')
sample_boxplot('total_cnt_of_molo_calls30')
sample_boxplot('total_cnt_of_on_net_calls30')
sample_boxplot('total_cnt_of_prem_calls30')
sample_boxplot('total_cnt_of_voice_calls30')
sample_boxplot('total_fah_sms30')
sample_boxplot('total_fah_time30')
sample_boxplot('total_internat_roaming_sms30')
sample_boxplot('total_international_sms30')
sample_boxplot('total_molo_time30')
sample_boxplot('total_out_of_bundle_time30')
sample_boxplot('total_sms30')
sample_boxplot('total_time_to_onnet30')
sample_boxplot('total_time_to_prem_numb30')
sample_boxplot('total_upfront_charge')
sample_boxplot('total_voice_time30')



#########################################################################

### HISTOGRAMS

# example
#flights['arr_delay'].describe()

#plt.figure(1)
# Make the histogram using matplotlib, bins must be integer
#plt.hist(flights['arr_delay'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))

# Add labels
#plt.title('Histogram of Arrival Delays')
#plt.xlabel('Delay (min)'); plt.ylabel('Flights');
#plt.show()


def histogram(cat):
    plt.hist(sample[cat], color = 'red', edgecolor = 'black',
         bins = int(50/5))
    plt.title(cat)

histogram('B_basic_MRC')
histogram('B_UPGRADE_DT_LREC')
histogram('C_UPGRADE_DT_LREC')
histogram('CONGESTED_DATA_RATE_4G_1800')
histogram('CONGESTED_DATA_RATE_4G_800')
histogram('CPP_CHANGE_1_LREC')
histogram('CPP_CHANGE_2_LREC')
histogram('CPP_CHANGE_3_LREC')
histogram('CPP_CHANGE_4_LREC')
histogram('CPP_DATE_1_LREC')
histogram('CPP_DATE_2_LREC')
histogram('CPP_DATE_3_LREC')
histogram('CPP_DATE_4_LREC')
histogram('DATA_ALLOC_50PC_DAYS_MIN')
histogram('DATA_ALLOC_75PC_DAYS_MIN')
histogram('DATA_ALLOC_90PC_DAYS_MIN')
histogram('DATA_ALLOC_FULL_CONS_DAYS_MIN')
histogram('DCR_3G')
histogram('estimated_cogs')
histogram('HH_ACCOUNTS')
histogram('HH_REC_MBB_ACQ_DATE_LREC')
histogram('HH_REC_MBB_CHURN_DATE_LREC')
histogram('HH_REC_MBB_UPG_DATE_LREC')
histogram('HH_REC_VOI_ACQ_DATE_LREC')
histogram('HH_REC_VOI_CHURN_DATE_LREC')
histogram('HH_REC_VOI_UPG_DATE_LREC')
histogram('hs_age_mths')
histogram('INT_REC_BILL_ENQ_LREC')
histogram('INT_REC_CANT_PAY_LREC')
histogram('INT_REC_COMP_BETTER_OFFER_LREC')
histogram('INT_REC_COMPLAINT_LREC')
histogram('INT_REC_DEVICE_LOSTSTOL_LREC')
histogram('INT_REC_MAKE_PAYMENT_LREC')
histogram('INT_REC_NETWORK_LREC')
histogram('INT_REC_NEW_PH_CON_LREC')
histogram('INT_REC_OFFERS_ENQ_LREC')
histogram('INT_REC_REFUND_ENQ_LREC')
histogram('INT_REC_TARIFF_ENQ_LREC')
histogram('INT_REC_TECH_ENQ_LREC')
histogram('INT_REC_WANT_BETTER_DEAL_LREC')
histogram('INT_REC_WANT_CANCEL_LREC')
histogram('INT_REC_WANT_DAT_ADDON_LREC')
histogram('INT_REC_WANT_PAC_LREC')
histogram('INT_REC_WANT_UPGRADE_LREC')
histogram('INT_REC_WONT_PAY_LREC')
histogram('IVR_REC_CHN_DATE_LREC')
histogram('IVR_REC_UPG_DATE_LREC')
histogram('last_3_mths_active_days')
histogram('MAX_DL_PEAK_THROUGPUT_3G')
histogram('MAX_DL_PEAK_THROUGPUT_4G_1800')
histogram('MAX_DL_PEAK_THROUGPUT_4G_800')
histogram('MINS_TO_VMAIL_RATE')
histogram('ML_REC_MBB_ACQ_DATE_LREC')
histogram('ML_REC_MBB_CHURN_DATE_LREC')
histogram('ML_REC_MBB_UPG_DATE_LREC')
histogram('ML_REC_VOI_ACQ_DATE_LREC')
histogram('ML_REC_VOI_CHURN_DATE_LREC')
histogram('ML_REC_VOI_UPG_DATE_LREC')
histogram('OOCR_3G')
histogram('PAGE_LOAD_ERRORS_3G')
histogram('PAGE_LOAD_ERRORS_4G_1800')
histogram('PAGE_LOAD_ERRORS_4G_800')
histogram('pct_16mb_dlspeed_1m')
histogram('pct_16mb_dlspeed_3m')
histogram('pct_1kb_dlspeed_1m')
histogram('pct_1kb_dlspeed_3m')
histogram('pct_1mb_dlspeed_1m')
histogram('pct_1mb_dlspeed_3m')
histogram('pct_200kb_dlspeed_1m')
histogram('pct_200kb_dlspeed_3m')
histogram('pct_2mb_dlspeed_1m')
histogram('pct_2mb_dlspeed_3m')
histogram('pct_400kb_dlspeed_1m')
histogram('pct_400kb_dlspeed_3m')
histogram('pct_4mb_dlspeed_1m')
histogram('pct_4mb_dlspeed_3m')
histogram('pct_600kb_dlspeed_1m')
histogram('pct_600kb_dlspeed_3m')
histogram('pct_800kb_dlspeed_1m')
histogram('pct_800kb_dlspeed_3m')
histogram('pct_8mb_dlspeed_1m')
histogram('pct_8mb_dlspeed_3m')
histogram('pre_mrc_excl_vat')
histogram('REDIALS_3G')
histogram('REDIALS_VOLTE')
histogram('TOP5_REC_ACQ_DT_LREC')
histogram('TOP5_REC_CHURN_DT_LREC')
histogram('TOP5_REC_UPG_DT_LREC')
histogram('UL_AVG_RTT_3G')
histogram('UL_AVG_RTT_4G_1800')
histogram('UL_AVG_RTT_4G_800')



#########################################################################

## Corrlations by data category

# Account cat
df = sample[[
'Dev_1_3yr_LT_500',
'Dev_1yr_LT_500',
'Dev_GT_3yr_GT_100',
'Dev_LT_3yr_GT_500',
'HH_ACCOUNTS',
'hh_accs_closed',
'hh_accs_opened',
'hh_accs_upgrade',
'hh_cpp_acc',
'hh_voi_acc_closed_m3',
'HH_Voi_acc_FTG',
'HH_Voi_acc_inlife',
'hh_voi_accs_closed',
'hh_voi_accs_opened',
'hh_voi_accs_upgrade',
'HH_Voice',
'hs_age_mths',
'INFLUENCE_SCORE',
'INVESTMENT_SCORE',
'LM_top1_chn',
'LM_top1_cpp',
'LM_top1_ftg',
'LM_top1_inl',
'LM_top1_ret',
'LM_top1_upg',
'LM_top2_chn',
'LOYALITY_SCORE',
'no_of_renewals',
'pre_alloc',
'pre_mrc_excl_vat',
'SIZE_OF_INFLUENCE',
'tenure'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Blues')


# Billing cat
df = sample[[
'Amortisation',
'amortisation_90',
'bill_prod_MRC',
'bill_prod_MRC_rpi',
'CPP_6M_COUNT',
'CRM_MRC',
'CRM_MRC_rpi',
'EOCN_MRC',
'estimated_cogs',
'final_MRC',
'Handset_cost',
'ifrs_ubd',
'invoice_MRC',
'net_total_cost',
'NMRC',
'NMRC_90'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Greens')


# Demogs cat
df = sample[[
'age',
'BEST_CUSTOMER_SCORE'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Greys')


# Interactions cat
df = sample[[
'int_bill_enq_m1',
'int_bill_enq_m2',
'int_bill_enq_m3',
'int_cc_m1',
'int_cc_m2',
'int_cc_m3',
'int_network_m1',
'int_outcm_case_raised_m1',
'int_outcm_case_raised_m2',
'int_outcm_case_raised_m3',
'int_outcm_consid_offer_m1',
'int_outcm_consid_offer_m2',
'int_outcm_consid_offer_m3',
'int_outcm_pac_given_m1',
'int_outcm_pac_given_m2',
'int_outcm_pac_given_m3',
'int_outcm_resolved_m1',
'int_outcm_resolved_m2',
'int_outcm_resolved_m3',
'int_tech_enq_m1',
'int_tech_enq_m2',
'int_tech_enq_m3',
'int_want_cancel_m1',
'int_want_cancel_m2',
'int_want_cancel_m3',
'int_want_pac_m1',
'ivr_cancel',
'ivr_upgrade'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Purples')



# Network cat
df = sample[[
'CONGESTED_CALL_RATE_4G_1800',
'CONGESTED_CALL_RATE_4G_800',
'CONGESTED_DATA_RATE_3G',
'CONGESTED_DATA_RATE_4G_1800',
'CONGESTED_DATA_RATE_4G_800',
'DCR_3G',
'DCR_BAND_3G',
'DCR_BAND_4G_1800',
'DCR_BAND_4G_800',
'DL_70PCNT_THRESHOLD_3G',
'DL_70PCNT_THRESHOLD_4G_1800',
'DL_70PCNT_THRESHOLD_4G_800',
'DL_RETRANS_DAYS_3G',
'DL_RETRANS_DAYS_4G_1800',
'DL_RETRANS_DAYS_4G_800',
'DL_RETRANS_RATE_5PCT_3G',
'homesignal_flag',
'LM_call_drop',
'LM_COV_3G_BEST',
'LM_COV_4G_1800_BEST',
'LM_COV_4G_800_BEST',
'MAX_DL_PEAK_THROUGPUT_3G',
'MAX_DL_PEAK_THROUGPUT_4G_1800',
'MAX_DL_PEAK_THROUGPUT_4G_800',
'nw_speed_months',
'OOCR_3G',
'OOCR_4G_1800',
'OOCR_4G_800',
'PAGE_LOAD_ERRORS_3G',
'PAGE_LOAD_ERRORS_4G_1800',
'pct_16mb_dlspeed_3m',
'pct_1kb_dlspeed_3m',
'pct_1mb_dlspeed_3m',
'pct_200kb_dlspeed_3m',
'pct_2mb_dlspeed_3m',
'pct_400kb_dlspeed_3m',
'pct_4mb_dlspeed_3m',
'pct_600kb_dlspeed_3m',
'pct_8mb_dlspeed_3m',
'REDIALS_4G_1800',
'REDIALS_4G_800',
'REDIALS_BAND_VOLTE',
'tot_16mb_dlspeed_3m',
'tot_1kb_dlspeed_3m',
'tot_1mb_dlspeed_3m',
'tot_200kb_dlspeed_3m',
'tot_2mb_dlspeed_3m',
'tot_400kb_dlspeed_3m',
'tot_4mb_dlspeed_3m',
'tot_600kb_dlspeed_3m',
'tot_800kb_dlspeed_3m',
'tot_8mb_dlspeed_1m',
'tot_8mb_dlspeed_3m',
'UL_AVG_RTT_3G',
'UL_AVG_RTT_4G_1800',
'UL_AVG_RTT_4G_800'
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Reds')


# Usage cat
df = sample[[
'ALLOC_AYCE_DATA',
'ALLOC_AYCE_VOICE',
'ALLOC_DATA',
'ALLOC_VOICE',
'avg_of_premium_calls30',
'avg_perday_amount_of_data30',
'avg_perday_voice_time30',
'avg_unique_mob_num_dial30',
'avg_unique_num_dial30',
'count_total_3m',
'DATA_ALLOC_50PC_DAYS_count',
'DATA_ALLOC_75PC_DAYS_count',
'DATA_ALLOC_90PC_DAYS_count',
'DATA_ALLOC_FULL_CONS_DAYS_count',
'EOCN_DATA_USAGE_MB',
'LM_call_10',
'LM_call_30',
'LM_data_10',
'LM_data_30',
'LM_data_drop',
'LM_everday_call_10',
'LM_everday_call_30',
'LM_everday_data_10',
'LM_everday_data_30',
'LM_everday_sms_10',
'LM_everday_sms_30',
'LM_sms_10',
'LM_sms_30',
'LM_sms_drop',
'M1_GOBINGE_DATA_MB',
'M1_MAIN_DATA_MB',
'M1_TOT_DATA_MB',
'M1_USAGE_FLAG',
'MINS_TO_VMAIL_BAND',
'MINS_TO_VMAIL_RATE',
'std_data_used30',
'std_voice_time30',
'THREE_MTH_AVE_DATA_MB',
'Three_mth_GoBinge_data_MB',
'Three_MTH_MAIN_DATA_MB',
'THREE_MTH_TOT_DATA_MB',
'THREE_MTH_USAGE_FLAG',
'total_amnt_of_fah_roam_data30',
'total_amount_of_data30',
'total_cnt_of_fah_calls30',
'total_cnt_of_molo_calls30',
'total_cnt_of_on_net_calls30',
'total_cnt_of_prem_calls30',
'total_cnt_of_voice_calls30',
'total_fah_sms30',
'total_fah_time30',
'total_internat_roaming_sms30',
'total_international_sms30',
'total_molo_time30',
'total_out_of_bundle_time30',
'total_sms30',
'total_time_to_onnet30',
'total_time_to_prem_numb30',
'total_upfront_charge',
'total_voice_time30',
'VOICE_ALLOC_50PC_DAYS_count',
'VOICE_ALLOC_75PC_DAYS_count',
'VOICE_ALLOC_90PC_DAYS_count',
'VOICE_ALLOC_FULL_CONS_DAYS_count',
             ]]

corrMatrix = df.corr()
sns.set(font_scale=0.6)
sns.heatmap(corrMatrix, annot=False, cmap='Oranges')



#########################################################################

##  % bar charts

# need to aggregate data by year and gender firstly


## Example
adv_segment = sample.groupby(['adv_segment', 'target']).size().reset_index(name='counts')
event = adv_segment.loc[adv_segment.target == 1]
non_event = adv_segment.loc[adv_segment.target == 0]

# begin with stacked bars for men/women by year
N = len(event)
event_stats = event['counts'].tolist()
non_event_stats = non_event['counts'].tolist()
cats = non_event['adv_segment'].tolist()

ind = np.arange(N) # the x locations for the groups
width = 0.75       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, non_event_stats, width, color='grey')
p2 = plt.bar(ind, event_stats, width, bottom=non_event_stats, color='red')
plt.xlabel('adv_segment')
plt.ylabel('Count')
plt.title('adv_segment')
plt.xticks(ind, (cats), rotation=90)
plt.legend((p1[0], p2[0]), ('Non Event', 'Event'))

plt.show()



def perc_bar_chart(var):
    grp = sample.groupby([var, 'target']).size().reset_index(name='counts')
    event = grp.loc[grp.target == 1]
    non_event = grp.loc[grp.target == 0]
    N = len(event)
    event_stats = event['counts'].tolist()
    non_event_stats = non_event['counts'].tolist()
    cats = non_event[var].tolist()
    ind = np.arange(N) # the x locations for the groups
    width = 0.75       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, non_event_stats, width, color='grey')
    p2 = plt.bar(ind, event_stats, width, bottom=non_event_stats, color='red')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(var)
    plt.xticks(ind, (cats), rotation=90)
    plt.legend((p1[0], p2[0]), ('Non Event', 'Event'))

perc_bar_chart('adv_segment')
perc_bar_chart('ALLOC_AYCE_DATA')
perc_bar_chart('ALLOC_AYCE_VOICE')
perc_bar_chart('ALLOC_DATA')
perc_bar_chart('ALLOC_VOICE')
perc_bar_chart('B_CONTRACT_LENGTH')
perc_bar_chart('B_pre_upge_tariff')
perc_bar_chart('basic_segment')
perc_bar_chart('BEST_CUSTOMER_FLAG')
perc_bar_chart('BEST_CUSTOMER_SCORE')
perc_bar_chart('change_in_device')
perc_bar_chart('CONGESTED_CALL_RATE_4G_1800')
perc_bar_chart('CONGESTED_CALL_RATE_4G_800')
perc_bar_chart('CONGESTED_DATA_BAND_3G')
perc_bar_chart('CONGESTED_DATA_BAND_4G_1800')
perc_bar_chart('CONGESTED_DATA_BAND_4G_800')
perc_bar_chart('CONGESTED_DATA_RATE_3G')
perc_bar_chart('COVERAGE_BAND_3G')
perc_bar_chart('COVERAGE_BAND_4G_1800')
perc_bar_chart('COVERAGE_BAND_4G_800')
perc_bar_chart('CPP_6M_COUNT')
perc_bar_chart('DATA_ALLOC_50PC_DAYS_count')
perc_bar_chart('DATA_ALLOC_75PC_DAYS_count')
perc_bar_chart('DATA_ALLOC_90PC_DAYS_count')
perc_bar_chart('DATA_ALLOC_FULL_CONS_DAYS_count')
perc_bar_chart('data_tier')
perc_bar_chart('DCR_BAND_3G')
perc_bar_chart('DCR_BAND_4G_1800')
perc_bar_chart('DCR_BAND_4G_800')
perc_bar_chart('Dev_1_3yr_LT_500')
perc_bar_chart('Dev_1yr_LT_500')
perc_bar_chart('Dev_GT_3yr_GT_100')
perc_bar_chart('Dev_LT_3yr_GT_500')
perc_bar_chart('device_segment')
perc_bar_chart('DL_70PCNT_THRESHOLD_3G')
perc_bar_chart('DL_70PCNT_THRESHOLD_4G_1800')
perc_bar_chart('DL_70PCNT_THRESHOLD_4G_800')
perc_bar_chart('DL_70PCNT_THRESHOLD_BAND_3G')
perc_bar_chart('DL_70PCNT_THRESHOLD_BAND_4G_1800')
perc_bar_chart('DL_70PCNT_THRESHOLD_BAND_4G_800')
perc_bar_chart('DL_PEAK_THROUGPUT_BAND_3G')
perc_bar_chart('DL_PEAK_THROUGPUT_BAND_4G_1800')
perc_bar_chart('DL_PEAK_THROUGPUT_BAND_4G_800')
perc_bar_chart('DL_RETRANS_DAYS_3G')
perc_bar_chart('DL_RETRANS_DAYS_4G_1800')
perc_bar_chart('DL_RETRANS_DAYS_4G_800')
perc_bar_chart('DL_RETRANS_RATE_5PCT_3G')
perc_bar_chart('FIRST_CONTRACT')
perc_bar_chart('grouped_seg')
perc_bar_chart('hh_accs_closed')
perc_bar_chart('hh_accs_opened')
perc_bar_chart('hh_accs_upgrade')
perc_bar_chart('hh_cpp_acc')
perc_bar_chart('HH_MBB')
perc_bar_chart('hh_mbb_acc_closed_m1')
perc_bar_chart('hh_mbb_acc_closed_m2')
perc_bar_chart('hh_mbb_acc_closed_m3')
perc_bar_chart('HH_MBB_acc_FTG')
perc_bar_chart('HH_MBB_acc_inlife')
perc_bar_chart('HH_MBB_acc_ret_window')
perc_bar_chart('hh_MBB_accs_closed')
perc_bar_chart('hh_MBB_accs_opened')
perc_bar_chart('hh_MBB_accs_upgrade')
perc_bar_chart('hh_new_MBB_acc_m1')
perc_bar_chart('hh_new_MBB_acc_m2')
perc_bar_chart('hh_new_MBB_acc_m3')
perc_bar_chart('hh_new_voi_acc_m1')
perc_bar_chart('hh_new_voi_acc_m2')
perc_bar_chart('hh_new_voi_acc_m3')
perc_bar_chart('hh_voi_acc_closed_m1')
perc_bar_chart('hh_voi_acc_closed_m2')
perc_bar_chart('hh_voi_acc_closed_m3')
perc_bar_chart('HH_Voi_acc_FTG')
perc_bar_chart('HH_Voi_acc_inlife')
perc_bar_chart('HH_Voi_acc_ret_window')
perc_bar_chart('hh_voi_acc_upgrade_m1')
perc_bar_chart('hh_voi_acc_upgrade_m2')
perc_bar_chart('hh_voi_acc_upgrade_m3')
perc_bar_chart('hh_voi_accs_closed')
perc_bar_chart('hh_voi_accs_opened')
perc_bar_chart('hh_voi_accs_upgrade')
perc_bar_chart('HH_Voice')
perc_bar_chart('homesignal_flag')
perc_bar_chart('hset_age')
perc_bar_chart('hset_cogs')



perc_bar_chart('INFLUENCE_GROUP')
perc_bar_chart('INFLUENCE_SCORE')
perc_bar_chart('INSURANCE_PRODUCT')
perc_bar_chart('INSURANCE_PRODUCT_NAME')
perc_bar_chart('int_bill_enq_m1')
perc_bar_chart('int_bill_enq_m2')
perc_bar_chart('int_bill_enq_m3')
perc_bar_chart('int_cc_m1')
perc_bar_chart('int_cc_m2')
perc_bar_chart('int_cc_m3')
perc_bar_chart('int_network_m1')
perc_bar_chart('int_outcm_case_raised_m1')
perc_bar_chart('int_outcm_case_raised_m2')
perc_bar_chart('int_outcm_case_raised_m3')
perc_bar_chart('int_outcm_consid_offer_m1')
perc_bar_chart('int_outcm_consid_offer_m2')
perc_bar_chart('int_outcm_consid_offer_m3')
perc_bar_chart('int_outcm_pac_given_m1')
perc_bar_chart('int_outcm_pac_given_m2')
perc_bar_chart('int_outcm_pac_given_m3')
perc_bar_chart('int_outcm_resolved_m1')
perc_bar_chart('int_outcm_resolved_m2')
perc_bar_chart('int_outcm_resolved_m3')
perc_bar_chart('int_tech_enq_m1')
perc_bar_chart('int_tech_enq_m2')
perc_bar_chart('int_tech_enq_m3')
perc_bar_chart('int_want_cancel_m1')
perc_bar_chart('int_want_cancel_m2')
perc_bar_chart('int_want_cancel_m3')
perc_bar_chart('int_want_pac_m1')
perc_bar_chart('INVESTMENT_GROUP')
perc_bar_chart('INVESTMENT_SCORE')
perc_bar_chart('ivr_cancel')
perc_bar_chart('ivr_upgrade')
perc_bar_chart('LM_acq_channel')
perc_bar_chart('LM_call_10')
perc_bar_chart('LM_call_30')
perc_bar_chart('LM_call_drop')
perc_bar_chart('LM_COV_3G_BEST')
perc_bar_chart('LM_COV_4G_1800_BEST')
perc_bar_chart('LM_COV_4G_800_BEST')
perc_bar_chart('LM_data_10')
perc_bar_chart('LM_data_30')
perc_bar_chart('LM_data_drop')
perc_bar_chart('LM_everday_call_10')
perc_bar_chart('LM_everday_call_30')
perc_bar_chart('LM_everday_data_10')
perc_bar_chart('LM_everday_data_30')
perc_bar_chart('LM_everday_sms_10')
perc_bar_chart('LM_everday_sms_30')
perc_bar_chart('LM_netw_dev')
perc_bar_chart('LM_netw_dev_grp')
perc_bar_chart('LM_sms_10')
perc_bar_chart('LM_sms_30')
perc_bar_chart('LM_sms_drop')
perc_bar_chart('LM_top1_acq_m1')
perc_bar_chart('LM_top1_acq_m2')
perc_bar_chart('LM_top1_acq_m3')
perc_bar_chart('LM_top1_chn')
perc_bar_chart('LM_top1_chn_m1')
perc_bar_chart('LM_top1_chn_m2')
perc_bar_chart('LM_top1_chn_m3')
perc_bar_chart('LM_top1_cpp')
perc_bar_chart('LM_top1_ftg')
perc_bar_chart('LM_top1_inl')
perc_bar_chart('LM_top1_ret')
perc_bar_chart('LM_top1_upg')
perc_bar_chart('LM_top1_upg_m1')
perc_bar_chart('LM_top1_upg_m2')
perc_bar_chart('LM_top1_upg_m3')
perc_bar_chart('LM_top2_acq_m1')
perc_bar_chart('LM_top2_acq_m2')
perc_bar_chart('LM_top2_acq_m3')
perc_bar_chart('LM_top2_chn')
perc_bar_chart('LM_top2_chn_m1')
perc_bar_chart('LM_top2_chn_m2')
perc_bar_chart('LM_top2_chn_m3')
perc_bar_chart('LM_top2_cpp')
perc_bar_chart('LM_top2_ftg')
perc_bar_chart('LM_top2_inl')
perc_bar_chart('LM_top2_ret')
perc_bar_chart('LM_top2_upg')
perc_bar_chart('LM_top2_upg_m1')
perc_bar_chart('LM_top2_upg_m2')
perc_bar_chart('LM_top2_upg_m3')
perc_bar_chart('LM_upg_channel')
perc_bar_chart('LOYALITY_GROUP')
perc_bar_chart('LOYALITY_SCORE')
perc_bar_chart('M1_USAGE_FLAG')
perc_bar_chart('M2_USAGE_FLAG')
perc_bar_chart('M3_USAGE_FLAG')
perc_bar_chart('MINS_TO_VMAIL_BAND')
perc_bar_chart('ml_ACCOUNTS')
perc_bar_chart('ml_accs_closed')
perc_bar_chart('ml_accs_opened')
perc_bar_chart('ml_accs_upgrade')
perc_bar_chart('ml_cpp_acc')
perc_bar_chart('ml_MBB')
perc_bar_chart('ml_mbb_acc_closed_m1')
perc_bar_chart('ml_mbb_acc_closed_m2')
perc_bar_chart('ml_mbb_acc_closed_m3')
perc_bar_chart('ml_MBB_acc_FTG')
perc_bar_chart('ml_MBB_acc_inlife')
perc_bar_chart('ml_MBB_acc_ret_window')
perc_bar_chart('ml_MBB_accs_closed')
perc_bar_chart('ml_MBB_accs_opened')
perc_bar_chart('ml_new_MBB_acc_m1')
perc_bar_chart('ml_new_MBB_acc_m2')
perc_bar_chart('ml_new_MBB_acc_m3')
perc_bar_chart('ml_new_voi_acc_m1')
perc_bar_chart('ml_new_voi_acc_m2')
perc_bar_chart('ml_new_voi_acc_m3')
perc_bar_chart('ml_voi_acc_closed_m1')
perc_bar_chart('ml_voi_acc_closed_m2')
perc_bar_chart('ml_voi_acc_closed_m3')
perc_bar_chart('ml_Voi_acc_FTG')
perc_bar_chart('ml_Voi_acc_inlife')
perc_bar_chart('ml_Voi_acc_ret_window')
perc_bar_chart('ml_voi_acc_upgrade_m1')
perc_bar_chart('ml_voi_acc_upgrade_m2')
perc_bar_chart('ml_voi_acc_upgrade_m3')
perc_bar_chart('ml_voi_accs_closed')
perc_bar_chart('ml_voi_accs_opened')
perc_bar_chart('ml_voi_accs_upgrade')
perc_bar_chart('ml_Voice')
perc_bar_chart('Mosaic_UK6_GROUP_char')
perc_bar_chart('new_segment')
perc_bar_chart('no_of_renewals')
perc_bar_chart('OOC_BAND_3G')
perc_bar_chart('OOC_BAND_4G_1800')
perc_bar_chart('OOC_BAND_4G_800')
perc_bar_chart('OOCR_4G_1800')
perc_bar_chart('OOCR_4G_800')
perc_bar_chart('PAGE_LOAD_ERROR_BAND_3G')
perc_bar_chart('PAGE_LOAD_ERROR_BAND_4G_1800')
perc_bar_chart('PAGE_LOAD_ERROR_BAND_4G_800')
perc_bar_chart('pre_alloc')
perc_bar_chart('prev_data_tier')
perc_bar_chart('prev_device_type')
perc_bar_chart('RADIO_FAILURE_BAND_3G')
perc_bar_chart('RADIO_FAILURE_BAND_4G_1800')
perc_bar_chart('RADIO_FAILURE_BAND_4G_800')
perc_bar_chart('REDIAL_BAND_3G')
perc_bar_chart('REDIALS_4G_1800')
perc_bar_chart('REDIALS_4G_800')
perc_bar_chart('REDIALS_BAND_VOLTE')
perc_bar_chart('renewal_12m')
perc_bar_chart('renewal_1m')
perc_bar_chart('renewal_24m')
perc_bar_chart('SIZE_OF_INFLUENCE')
perc_bar_chart('target')
perc_bar_chart('THREE_MTH_USAGE_FLAG')
perc_bar_chart('UL_AVG_RTT_BAND_3G')
perc_bar_chart('UL_AVG_RTT_BAND_4G_1800')
perc_bar_chart('UL_AVG_RTT_BAND_4G_800')
perc_bar_chart('used_device_make')
perc_bar_chart('useg')
perc_bar_chart('VALUE_GROUP')
perc_bar_chart('VOICE_ALLOC_50PC_DAYS_count')
perc_bar_chart('VOICE_ALLOC_50PC_DAYS_MIN')
perc_bar_chart('VOICE_ALLOC_75PC_DAYS_count')
perc_bar_chart('VOICE_ALLOC_75PC_DAYS_MIN')
perc_bar_chart('VOICE_ALLOC_90PC_DAYS_count')
perc_bar_chart('VOICE_ALLOC_90PC_DAYS_MIN')
perc_bar_chart('VOICE_ALLOC_FULL_CONS_DAYS_count')
perc_bar_chart('VOICE_ALLOC_FULL_CONS_DAYS_MIN')
perc_bar_chart('Zero_use')
perc_bar_chart('dev_make')





#########################################################################
#########################################################################
###------------------------ END OF PROGRAM ---------------------------###
#########################################################################
#########################################################################
