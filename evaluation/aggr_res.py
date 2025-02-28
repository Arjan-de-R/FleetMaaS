### Script for analysing replications and converting to scenarios (for specified scenarios)
# For each replication, we determine additional statistics than the ones already included, e.g. statistics depending on registration status or multihoming type
# We determine indicator values in equilibrium
# We check the difference in equilibrium values between different replications, and conclude on if sufficient replications have been run
# We pickle the aggregated results
import os
path = os.getcwd()
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from utils import create_d2d_df_list, create_attr_df_list, determine_req_repl_indicator, contains_item_from_list, time_string_to_seconds
import itertools
import pickle

### ---------------------- INPUT -----------------------###

# Which scenarios?
# var_dict = {'cmpt_type': ['s','p','ss','pp','sp']}  # assumes full enumeration
# var_dict = {'cmpt_type': ['ss']}  # assumes full enumeration
# var_dict = {'cmpt_type': ['ss'], 'dem_mh_share': [0.5, 1], 'sup_mh_share': [0]}  # assumes full enumeration
# var_dict = {'cmpt_type': ['sp'], 'start_reg_plf_share': ['0.1 0.9', '0.3 0.7', '0.7 0.3', '0.9 0.1']}  # assumes full enumeration
var_dict = {'cmpt_type': ['ss'], 'beta_reg': [0.1, 0.25], 'sup_util_multiplier': [5]}  # assumes full enumeration
# ref_value = {'name': 'num_signals', 'val': 50}

# Required parameter values for statistical significance of equilibria
conv_signif = 0.05
conv_max_error = 0.05
conv_steady_days = 25 # Note: needs to be same as in simulation!
moving_average_days = 125 # Note: needs to be same as in simulation!

# Results path
res_path = os.path.join(path, 'results')

# Experiment name
study_name = 'homing_cmpt' # determines where aggregated results are stored

### --------------------- SCRIPT --------------------- ###

# Determine scenarios - enumerating based on input
keys, values_lists = zip(*var_dict.items())
scenarios = list(itertools.product(*values_lists))
scenario_names = ['-'.join(f"{key}-{value}" for key, value in zip(keys, scenario)) for scenario in scenarios]
scenario_names = [s.replace('cmpt_type-','') for s in scenario_names]  # drop 'cmpt_type' from scn name

# Loop over scenarios
for scn_name in scenario_names:
    # Determine scenario values and save them to dict
    scn_name_split = scn_name.split('-')
    scn_dict = {}
    for i in range(1, len(scn_name_split), 2):
        variable_name = scn_name_split[i]
        variable_value = scn_name_split[i + 1]
        scn_dict[variable_name] = variable_value
    scn_dict['cmpt_type'] = scn_name_split[0]

    # List all files and find those corresponding to (different replications of) the specific scenario
    all_items = os.listdir(res_path)
    # Check if scenario is the reference scenario, find the right file name
    # if float(scn_dict[ref_value['name']]) != ref_value['val']:
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(res_path, item)) and '-'.join(item.split('-')[:-2]) == scn_name] # Filter only the directories
    # else:
        # folder_names = [item for item in all_items if os.path.isdir(os.path.join(res_path, item)) and item.startswith(scn_name.split("-")[0]+'-service_types')] # Filter only the directories
    # folder_names = [item for item in all_items if os.path.isdir(os.path.join(res_path, item)) and item.startswith(scn_name + '-service_types')] # Filter only the directories

    # Open replication-independent (but scenario-specific) files - i.e. agent properties TODO: saving params to be used in analysis
    f = open(os.path.join(res_path, folder_names[0], '0_params.json'))
    params = json.load(f)

    # Check if scenario folder exists in 'evaluation'
    aggr_scn_path = os.path.join('evaluation', 'studies', study_name, 'aggr_repl', scn_name)
    if not os.path.exists(os.path.join('evaluation', 'studies')):
        os.mkdir(os.path.join('evaluation', 'studies'))
    if not os.path.exists(os.path.join('evaluation', 'studies', study_name)):
        os.mkdir(os.path.join('evaluation', 'studies', study_name))
    if not os.path.exists(os.path.join('evaluation', 'studies', study_name, 'aggr_repl')):
        os.mkdir(os.path.join('evaluation', 'studies', study_name, 'aggr_repl'))
    if not os.path.exists(aggr_scn_path):
        os.mkdir(aggr_scn_path)

    # Initialise files
    d2d_pax_list, d2d_veh_list, pax_attr_list, driver_attr_list, plf_attr_list, all_pax_attr_list, conv_list, req_repl = [], [], [], [], [], [], [], pd.DataFrame()

    # Loop over replications (adding all replications of the scenario to a single dataframe)
    for repl_folder in folder_names:
        repl_id = int(repl_folder.split("-")[-1])
        repl_folder_path = os.path.join(res_path, repl_folder)
        for item in os.listdir(repl_folder_path): # loop over items corresponding to a single replication
            if item.startswith('day'):
                day_id = int(item.split("_")[1])
                if item.endswith('travs.csv'):
                    d2d_pax_list = create_d2d_df_list(repl_id, day_id, repl_folder_path, item, agent_type='pax', d2d_df_list=d2d_pax_list)
                else:
                    d2d_veh_list = create_d2d_df_list(repl_id, day_id, repl_folder_path, item, agent_type='veh', d2d_df_list=d2d_veh_list)
            if item.endswith('1_pax-properties.csv'):
                pax_attr_list = create_attr_df_list(repl_id, repl_folder_path, item, index_name='pax_id', attr_df_list=pax_attr_list)
            if item.endswith('2_driver-properties.csv'):
                driver_attr_list = create_attr_df_list(repl_id, repl_folder_path, item, index_name='veh_id', attr_df_list=driver_attr_list)
            if item.endswith('3_platform-properties.csv'):
                plf_attr_list = create_attr_df_list(repl_id, repl_folder_path, item, index_name='id', attr_df_list=plf_attr_list)
            if item.endswith('4_out-filter-pax.csv'):
                all_pax_attr_list = create_attr_df_list(repl_id, repl_folder_path, item, index_name='pax_id', attr_df_list=all_pax_attr_list)
            if item.endswith('5_conv-indicators.csv'):
                conv_list = create_attr_df_list(repl_id, repl_folder_path, item, index_name='day', attr_df_list=conv_list)

    print('All dataframes have been loaded')
    # Concat the list of dataframes into a single dataframe
    d2d_pax = pd.concat(d2d_pax_list)
    d2d_veh = pd.concat(d2d_veh_list)
    pax_attr = pd.concat(pax_attr_list).drop(columns=['Unnamed: 0']).rename_axis(index={'pax_id': 'pax'})
    driver_attr = pd.concat(driver_attr_list).rename_axis(index={'veh_id': 'veh'})
    plf_attr = pd.concat(plf_attr_list)
    all_pax_attr = pd.concat(all_pax_attr_list)
    conv_df = pd.concat(conv_list)

    # Sort multi-index df's
    d2d_pax = d2d_pax.sort_index(level=['repl','day','pax'])
    d2d_veh = d2d_veh.sort_index(level=['repl','day','veh'])
    pax_attr = pax_attr.sort_index(level=['repl','pax'])
    driver_attr = driver_attr.sort_index(level=['repl','veh'])
    plf_attr = plf_attr.sort_index(level=['repl','id'])
    all_pax_attr = all_pax_attr.sort_index(level=['repl','pax_id'])
    conv_df = conv_df.sort_index(level=['repl','day'])

    # Determine winning platform - based on last day (most demand)
    last_day_df = conv_df.groupby('repl').tail(1).copy()
    last_day_df['plf_0_winner'] = True
    last_day_df.loc[last_day_df['ptcp_dem_sh_0'] < last_day_df['ptcp_dem_sh_1'], 'plf_0_winner'] = False
    last_day_df = last_day_df.reset_index(level='day', drop=True)

    # Convert time string to seconds
    pax_attr['ttrav'] = pax_attr.apply(lambda row: time_string_to_seconds(row.ttrav), axis=1)

    # Convert chosen_mode column to column per mode, with boolean values
    alt_mode_list = d2d_pax.chosen_mode.unique().tolist()
    alt_mode_list.remove('rs')
    for mode in alt_mode_list:
        d2d_pax[mode] = (d2d_pax.chosen_mode == mode)
        pax_attr['no_rs_{}'.format(mode)] = (pax_attr.mode_without_rs == mode)
        all_pax_attr['other_trav_{}'.format(mode)] = (all_pax_attr.mode_choice == mode)

    # For each indicator in d2d_pax, determine whether it is a 'count' or 'mean' indicator (in the population of agents)
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "registered", "requests", "offer"] + alt_mode_list 
    for col in np.concatenate((d2d_pax.columns.values, (np.array(['xp_detour_0', 'xp_detour_1', 'expected_detour_0', 'expected_detour_1']) if 'xp_ivt_1' in d2d_pax.columns else np.array(['xp_detour_0','expected_detour_0'])))):
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)

    # First add traveller attributes to d2d dataframes
    common_columns = ['repl', 'pax']
    d2d_pax_reset = d2d_pax.reset_index()
    mh_df_reset = pax_attr.reset_index()
    d2d_pax_reset = d2d_pax_reset.merge(mh_df_reset, on=common_columns, how='left')
    d2d_pax_reset.set_index(['repl', 'day', 'pax'], inplace=True)
    d2d_pax_reset['xp_detour_0'] = (d2d_pax_reset.xp_ivt_0 / d2d_pax_reset.ttrav) - 1
    d2d_pax_reset['expected_detour_0'] = (d2d_pax_reset.expected_ivt_0 / d2d_pax_reset.ttrav) - 1
    if 'xp_ivt_1' in d2d_pax_reset.columns:
        d2d_pax_reset['xp_detour_1'] = (d2d_pax_reset.xp_ivt_1 / d2d_pax_reset.ttrav) - 1
        d2d_pax_reset['expected_detour_1'] = (d2d_pax_reset.expected_ivt_1 / d2d_pax_reset.ttrav) - 1

    # Add addititional indicators - separating multihomers and singlehomers
    grouped_result_count = d2d_pax_reset.groupby(['repl', 'day', 'multihoming'])[count_indicators].sum()
    grouped_result_mean = d2d_pax_reset.groupby(['repl', 'day', 'multihoming'])[mean_indicators].mean()
    grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
    pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                            columns='multihoming', 
                                            values=grouped_result.columns, 
                                            fill_value=0)
    d2d_pax_stats = pd.DataFrame(index=pivot_result.index)
    any_trav_mh = pax_attr.multihoming.any()
    any_trav_sh = not pax_attr.multihoming.all()
    d2d_pax_stats['informed_mh'] = pivot_result['informed'][True] if any_trav_mh else np.nan
    d2d_pax_stats['informed_sh'] = pivot_result['informed'][False] if any_trav_sh else np.nan
    d2d_pax_stats['registered_mh'] = pivot_result['registered_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['registered_sh_0'] = pivot_result['registered_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['registered_sh_1'] = pivot_result['registered_1'][False] if any_trav_sh and 'registered_1' in pivot_result.columns else np.nan
    d2d_pax_stats['requests_mh'] = pivot_result['requests_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['requests_sh_0'] = pivot_result['requests_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['registered_sh_1'] = pivot_result['registered_1'][False] if any_trav_sh and 'registered_1' in pivot_result.columns else np.nan
    d2d_pax_stats['requests_sh_1'] = pivot_result['requests_1'][False] if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['gets_offer_mh_0'] = pivot_result['gets_offer_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['gets_offer_mh_1'] = pivot_result['gets_offer_1'][True] if any_trav_mh and 'gets_offer_1' in pivot_result.columns  else np.nan
    d2d_pax_stats['gets_offer_sh_0'] = pivot_result['gets_offer_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['gets_offer_sh_1'] = pivot_result['gets_offer_1'][False] if any_trav_sh and 'gets_offer_1' in pivot_result.columns else np.nan
    d2d_pax_stats['accepts_offer_mh_0'] = pivot_result['accepts_offer_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['accepts_offer_mh_1'] = pivot_result['accepts_offer_1'][True] if any_trav_mh and 'accepts_offer_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_wait_mh'] = pivot_result['xp_wait_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['exp_wait_sh_0'] = pivot_result['xp_wait_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['exp_wait_sh_1'] = pivot_result['xp_wait_1'][False] if any_trav_sh and 'xp_wait_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_corr_wait_mh'] = pivot_result['corr_xp_wait_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['exp_corr_wait_sh_0'] = pivot_result['corr_xp_wait_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['exp_corr_wait_sh_1'] = pivot_result['corr_xp_wait_1'][False] if any_trav_sh and 'corr_xp_wait' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_detour_mh'] = pivot_result['xp_detour_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['exp_detour_sh_0'] = pivot_result['xp_detour_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['exp_detour_sh_1'] = pivot_result['xp_detour_1'][False] if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_km_fare_mh'] = pivot_result['xp_km_fare_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['exp_km_fare_sh_0'] = pivot_result['xp_km_fare_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['exp_km_fare_sh_1'] = pivot_result['xp_km_fare_1'][False] if any_trav_sh and 'xp_km_fare_1' in pivot_result.columns else np.nan
    d2d_pax_stats['perc_wait_mh'] = pivot_result['expected_wait_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['perc_wait_sh_0'] = pivot_result['expected_wait_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['perc_wait_sh_1'] = pivot_result['expected_wait_1'][False] if any_trav_sh and 'expected_wait_1' in pivot_result.columns else np.nan
    d2d_pax_stats['perc_detour_mh'] = pivot_result['expected_detour_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['perc_detour_sh_0'] = pivot_result['expected_detour_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['perc_detour_sh_1'] = pivot_result['expected_detour_1'][False] if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['perc_km_fare_mh'] = pivot_result['expected_km_fare_0'][True] if any_trav_mh else np.nan
    d2d_pax_stats['perc_km_fare_sh_0'] = pivot_result['expected_km_fare_0'][False] if any_trav_sh else np.nan
    d2d_pax_stats['perc_km_fare_sh_1'] = pivot_result['expected_km_fare_1'][False] if any_trav_sh and 'expected_km_fare_1' in pivot_result.columns else np.nan

    # Statistics for winning platform
    d2d_pax_stats = d2d_pax_stats.reset_index().merge(last_day_df.reset_index(), on='repl', how='left').set_index(['repl', 'day'])      # Merge the d2d_pax_stats with last_day_df to get 'plf_0_winner' for each 'repl'
    d2d_pax_stats['requests_sh_largest'] = d2d_pax_stats.apply(lambda row: row['requests_sh_0'] if row['plf_0_winner'] else row['requests_sh_1'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['gets_offer_mh_largest'] = d2d_pax_stats.apply(lambda row: row['gets_offer_mh_0'] if row['plf_0_winner'] else row['gets_offer_mh_1'], axis=1) if (any_trav_mh and any_trav_sh) and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['accepts_offer_mh_largest'] = d2d_pax_stats.apply(lambda row: row['accepts_offer_mh_0'] if row['plf_0_winner'] else row['accepts_offer_mh_1'], axis=1) if (any_trav_mh and any_trav_sh) and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_wait_sh_largest'] = d2d_pax_stats.apply(lambda row: row['exp_wait_sh_0'] if row['plf_0_winner'] else row['exp_wait_sh_1'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_wait_sh_smallest'] = d2d_pax_stats.apply(lambda row: row['exp_wait_sh_1'] if row['plf_0_winner'] else row['exp_wait_sh_0'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_corr_wait_sh_largest'] = d2d_pax_stats.apply(lambda row: row['exp_corr_wait_sh_0'] if row['plf_0_winner'] else row['exp_corr_wait_sh_1'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['exp_corr_wait_sh_smallest'] = d2d_pax_stats.apply(lambda row: row['exp_corr_wait_sh_1'] if row['plf_0_winner'] else row['exp_corr_wait_sh_0'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['perc_wait_sh_largest'] = d2d_pax_stats.apply(lambda row: row['perc_wait_sh_0'] if row['plf_0_winner'] else row['perc_wait_sh_1'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats['perc_wait_sh_smallest'] = d2d_pax_stats.apply(lambda row: row['perc_wait_sh_1'] if row['plf_0_winner'] else row['perc_wait_sh_0'], axis=1) if any_trav_sh and 'requests_1' in pivot_result.columns else np.nan
    d2d_pax_stats = d2d_pax_stats.drop(columns=['plf_0_winner'])

    with open(os.path.join(aggr_scn_path, 'repl_dem_select.pkl'), 'wb') as f:
        pickle.dump(d2d_pax_stats, f)

    # From hereon, only consider days in equilibrium (and reindex starting from 0)
    def select_eql_days_aggr(df):
        '''filter only days in equilibrium in dataframes with data aggregated for the population (no individual travellers)'''
        df = df.groupby(level='repl').tail(conv_steady_days + moving_average_days).reset_index()
        df['day'] = df.groupby('repl').cumcount()
        df = df.set_index(['repl', 'day'])
        return df
    
    def select_eql_days_indiv(df, agent):
        '''filter only days in equilibrium for dataframes with individual travellers'''
        df = df.groupby(level=['repl', agent]).tail(conv_steady_days + moving_average_days).reset_index()
        df['day'] = df.groupby(['repl', agent]).cumcount()
        df = df.set_index(['repl', 'day', agent])
        return df

    d2d_pax = select_eql_days_indiv(d2d_pax, 'pax')
    d2d_pax_stats = select_eql_days_aggr(d2d_pax_stats)
    d2d_pax_reset = select_eql_days_indiv(d2d_pax_reset, 'pax')

    # Created population-level statistics (aggregate over agents) for each replication - on the demand side
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "registered", "requests", "offer"] + alt_mode_list
    for col in d2d_pax.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)
    
    aggr_orig_sum = d2d_pax[count_indicators].groupby(['repl', 'day']).sum().groupby(['day']).mean() # Aggregating by taking average across 'repl' and summing over 'pax' for each column
    aggr_orig_mean = d2d_pax[mean_indicators].groupby(['repl', 'day']).mean().groupby(['day']).mean() # Aggregating by taking average across 'repl' and mean over 'pax' for each column
    
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "registered", "requests", "offer"] + alt_mode_list
    for col in d2d_pax_stats.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)
    aggr_new_sum = d2d_pax_stats[count_indicators].groupby(['repl', 'day']).sum().groupby(['day']).mean() # Aggregating by taking average across 'repl' and summing over 'pax' for each column
    aggr_new_mean = d2d_pax_stats[mean_indicators].groupby(['repl', 'day']).mean().groupby(['day']).mean() # Aggregating by taking average across 'repl' and mean over 'pax' for each column

    aggr_dem_df = pd.concat([aggr_orig_sum, aggr_orig_mean, aggr_new_sum, aggr_new_mean], axis=1)

    # Calculate effective passenger km per platform
    d2d_pax_reset['eff_pax_dist_0'] = d2d_pax_reset.dist * d2d_pax_reset.accepts_offer_0
    d2d_pax_reset['eff_pax_dist_1'] = d2d_pax_reset.dist * d2d_pax_reset.accepts_offer_1 if 'accepts_offer_1' in d2d_pax_reset.columns else np.nan
    aggr_dem_df['eff_pax_dist_0'] = d2d_pax_reset['eff_pax_dist_0'].groupby(['repl', 'day']).sum().groupby(['day']).mean()
    aggr_dem_df['eff_pax_dist_1'] = d2d_pax_reset['eff_pax_dist_1'].groupby(['repl', 'day']).sum().groupby(['day']).mean()

    # Add modes without ridesourcing, including modes of filtered out pool of travellers
    rel_columns = [item for item in pax_attr.columns if item.startswith("no_rs")]
    other_mode_df = pax_attr[rel_columns].groupby(['repl']).sum().mean()
    rel_columns = [item for item in all_pax_attr.columns if item.startswith("other_trav")]
    other_mode_other_trav = all_pax_attr[rel_columns].groupby(['repl']).sum().mean()
    mode_no_rs = pd.concat([other_mode_df, other_mode_other_trav]) # of those in and out of sample

    # Created population-level statistics (aggregate over agents) for each replication - on the supply side
    # First replace 'out' by inverse: 'ptcp'
    d2d_veh['ptcp_0'] = ~d2d_veh['out_0']
    if 'out_1' in d2d_veh.columns:
        d2d_veh['ptcp_1'] = ~d2d_veh['out_1']

    # First add driver attributes to d2d dataframes
    common_columns = ['repl', 'veh']
    d2d_veh_reset = d2d_veh.reset_index()
    mh_df_reset = driver_attr.reset_index()
    d2d_veh_reset = d2d_veh_reset.merge(mh_df_reset, on=common_columns, how='left')
    d2d_veh_reset.set_index(['repl', 'day', 'veh'], inplace=True)

    # Determine exp_inc per platform (rather than for whole market as in 'exp_inc')
    d2d_veh_reset['exp_inc_0'] = d2d_veh_reset['exp_inc'] * d2d_veh_reset['ptcp_0'].apply(lambda x: 1 if x else np.nan)
    d2d_veh_reset['exp_inc_1'] = d2d_veh_reset['exp_inc'] * d2d_veh_reset['ptcp_1'].apply(lambda x: 1 if x else np.nan) if 'ptcp_1' in d2d_veh_reset.columns else np.nan

    # Determine the number of agents that register and deregister every day
    df_reset = d2d_veh_reset.copy().reset_index()
    df_sorted = df_reset.sort_values(by=['repl', 'day', 'veh'])
    df_sorted['registered_0'].replace({True: 1, False: 0}, inplace=True)
    df_sorted['registered_1'].replace({True: 1, False: 0}, inplace=True) if 'registered_1' in df_sorted.columns else np.nan
    df_sorted['regist_diff'] = df_sorted.groupby(['repl', 'veh'])['registered_0'].diff()

    def get_change_value(regist_diff):
        if pd.isna(regist_diff):
            return np.nan
        elif regist_diff == 0:
            return 'n'
        elif regist_diff == 1:
            return 'r'
        elif regist_diff == -1:
            return 'd'

    df_sorted['regist_outcome_0'] = df_sorted['regist_diff'].apply(get_change_value)
    df_sorted['regist_diff'] = df_sorted.groupby(['repl', 'veh'])['registered_1'].diff() if 'registered_1' in df_sorted.columns else np.nan
    df_sorted['regist_outcome_1'] = df_sorted['regist_diff'].apply(get_change_value)

    df_sorted['new_regist_sh_0'] = (df_sorted['regist_outcome_0'] == 'r') & ~df_sorted.multihoming
    df_sorted['new_deregist_sh_0'] = (df_sorted['regist_outcome_0'] == 'd') & ~df_sorted.multihoming
    df_sorted['new_regist_sh_1'] = (df_sorted['regist_outcome_1'] == 'r') & ~df_sorted.multihoming
    df_sorted['new_deregist_sh_1'] = (df_sorted['regist_outcome_1'] == 'd') & ~df_sorted.multihoming
    df_sorted['new_regist_mh'] = (df_sorted['regist_outcome_0'] == 'r') & df_sorted.multihoming
    df_sorted['new_deregist_mh'] = (df_sorted['regist_outcome_0'] == 'd') & df_sorted.multihoming
    df_sorted = df_sorted.set_index(['repl', 'day', 'veh'])
    d2d_veh_reset[['new_regist_sh_0', 'new_deregist_sh_0', 'new_regist_sh_1', 'new_deregist_sh_1', 'new_regist_mh', 'new_deregist_mh']] = df_sorted[['new_regist_sh_0', 'new_deregist_sh_0', 'new_regist_sh_1', 'new_deregist_sh_1', 'new_regist_mh', 'new_deregist_mh']]

    # Create the pivot results - e.g. income multihoming and income singlehoming
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "registered", "ptcp", "regist"]
    for col in d2d_veh_reset.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)
    mean_indicators.remove('multihoming')
    # mean_indicators = ['exp_inc_0', 'exp_inc_1', 'expected_income_0', 'expected_income_1', 'relev_perc_util']
    # count_indicators = ['informed', 'registered_0', 'registered_1', 'ptcp_0', 'ptcp_1','new_regist_sh_0', 'new_deregist_sh_0', 'new_regist_sh_1', 'new_deregist_sh_1', 'new_regist_mh', 'new_deregist_mh']
    grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'multihoming'])[count_indicators].sum()
    grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'multihoming'])[mean_indicators].mean()
    grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
    pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                            columns='multihoming', 
                                            values=grouped_result.columns, 
                                            fill_value=0)
    any_driver_mh = driver_attr.multihoming.any()
    any_driver_sh = not driver_attr.multihoming.all()
    d2d_veh_stats = pd.DataFrame()
    d2d_veh_stats['informed_mh'] = pivot_result['informed'][True] if any_driver_mh else np.nan
    d2d_veh_stats['informed_sh'] = pivot_result['informed'][False] if any_driver_sh else np.nan
    d2d_veh_stats['registered_mh'] = pivot_result['registered_0'][True] if any_driver_mh else np.nan
    d2d_veh_stats['registered_sh_0'] = pivot_result['registered_0'][False] if any_driver_sh else np.nan
    d2d_veh_stats['registered_sh_1'] = pivot_result['registered_1'][False] if any_driver_sh and 'registered_1' in pivot_result.columns else np.nan
    d2d_veh_stats['ptcp_mh'] = pivot_result['ptcp_0'][True] if any_driver_mh else np.nan
    d2d_veh_stats['ptcp_sh_0'] = pivot_result['ptcp_0'][False] if any_driver_sh else np.nan
    d2d_veh_stats['ptcp_sh_1'] = pivot_result['ptcp_1'][False] if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['exp_inc_mh'] = pivot_result['exp_inc_0'][True] if any_driver_mh else np.nan
    d2d_veh_stats['exp_inc_sh_0'] = pivot_result['exp_inc_0'][False] if any_driver_sh else np.nan
    d2d_veh_stats['exp_inc_sh_1'] = pivot_result['exp_inc_1'][False] if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['perc_inc_mh'] = pivot_result['expected_income_0'][True] if any_driver_mh else np.nan
    d2d_veh_stats['perc_inc_sh_0'] = pivot_result['expected_income_0'][False] if any_driver_sh else np.nan
    d2d_veh_stats['perc_inc_sh_1'] = pivot_result['expected_income_1'][False] if any_driver_sh and 'expected_income_1' in pivot_result.columns else np.nan
    d2d_veh_stats[['new_regist_sh_0', 'new_deregist_sh_0', 'new_regist_sh_1', 'new_deregist_sh_1', 'new_regist_mh', 'new_deregist_mh']] = d2d_veh_reset.groupby(['repl', 'day'])[count_indicators].sum()[['new_regist_sh_0', 'new_deregist_sh_0', 'new_regist_sh_1', 'new_deregist_sh_1', 'new_regist_mh', 'new_deregist_mh']]

    # Select winning platform
    d2d_veh_stats = d2d_veh_stats.reset_index().merge(last_day_df.reset_index(), on='repl', how='left').set_index(['repl', 'day'])      # Merge the d2d_veh_stats with last_day_df to get 'plf_0_winner' for each 'repl'
    d2d_veh_stats['ptcp_sh_largest'] = d2d_veh_stats.apply(lambda row: row['ptcp_sh_0'] if row['plf_0_winner'] else row['ptcp_sh_1'], axis=1) if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['exp_inc_sh_largest'] = d2d_veh_stats.apply(lambda row: row['exp_inc_sh_0'] if row['plf_0_winner'] else row['exp_inc_sh_1'], axis=1) if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['exp_inc_sh_smallest'] = d2d_veh_stats.apply(lambda row: row['exp_inc_sh_1'] if row['plf_0_winner'] else row['exp_inc_sh_0'], axis=1) if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['perc_inc_sh_largest'] = d2d_veh_stats.apply(lambda row: row['perc_inc_sh_0'] if row['plf_0_winner'] else row['perc_inc_sh_1'], axis=1) if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats['perc_inc_sh_smallest'] = d2d_veh_stats.apply(lambda row: row['perc_inc_sh_1'] if row['plf_0_winner'] else row['perc_inc_sh_0'], axis=1) if any_driver_sh and 'ptcp_1' in pivot_result.columns else np.nan
    d2d_veh_stats = d2d_veh_stats.drop(columns=['plf_0_winner'])

    with open(os.path.join(aggr_scn_path, 'repl_sup_select.pkl'), 'wb') as f:
        pickle.dump(d2d_veh_stats, f)

    # From hereon, only consider days in equilibrium (and reindex starting from 0)
    d2d_veh = select_eql_days_indiv(d2d_veh, 'veh')
    d2d_veh_stats = select_eql_days_aggr(d2d_veh_stats)
    d2d_veh_reset = select_eql_days_indiv(d2d_veh_reset, 'veh')

    # Create the pivot results - multihoming / registered
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "registered", "ptcp"]
    mean_string_list = ["inc", "util"]
    for col in d2d_veh_reset.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        if contains_item_from_list(col, mean_string_list):
            mean_indicators.append(col)
    count_indicators.remove('registered_0')
    # mean_indicators = ['exp_inc_0', 'exp_inc_1', 'expected_income_0', 'expected_income_1', 'relev_perc_util']
    # count_indicators = ['informed', 'registered_1', 'ptcp_0', 'ptcp_1']
    grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'registered_0', 'multihoming'])[count_indicators].sum()
    grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'registered_0', 'multihoming'])[mean_indicators].mean()
    grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
    pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                            columns=['registered_0','multihoming'], 
                                            values=grouped_result.columns, 
                                            fill_value=0)
    d2d_veh_stats['perc_inc_reg_0'] = pivot_result['expected_income_0'][True][False] if any_driver_sh else np.nan
    d2d_veh_stats['perc_inc_reg_mh'] = pivot_result['expected_income_0'][True][True] if any_driver_mh else np.nan
    d2d_veh_stats['perc_inc_notreg_mh'] = pivot_result['expected_income_0'][False][True] if any_driver_mh else np.nan

    if 'registered_1' in d2d_veh_reset.columns: # if there are two platforms
        count_indicators = ['informed', 'registered_0', 'ptcp_0', 'ptcp_1']
        grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'registered_1', 'multihoming'])[count_indicators].sum()
        grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'registered_1', 'multihoming'])[mean_indicators].mean()
        grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
        pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                                columns=['registered_1','multihoming'], 
                                                values=grouped_result.columns, 
                                                fill_value=0)
        d2d_veh_stats['perc_inc_reg_1'] = pivot_result['expected_income_1'][True][False] if any_driver_sh else np.nan

    mean_indicators = ['res_wage']
    grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'registered_0', 'multihoming'])[mean_indicators].mean()
    if 'registered_1' in d2d_veh_reset.columns: # two platforms
        count_indicators = ['registered_1', 'ptcp_1']
        grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'registered_0', 'multihoming'])[count_indicators].sum()
        grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
    else:
        grouped_result = grouped_result_mean
    pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                            columns=['registered_0','multihoming'], 
                                            values=grouped_result.columns, 
                                            fill_value=0)
    d2d_veh_stats['res_wage_reg_0'] = pivot_result['res_wage'][True][False] if any_driver_sh else np.nan
    d2d_veh_stats['res_wage_reg_mh'] = pivot_result['res_wage'][True][True] if any_driver_mh else np.nan

    mean_indicators = ['res_wage']
    count_indicators = []
    grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'ptcp_0', 'multihoming'])[count_indicators].sum()
    grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'ptcp_0', 'multihoming'])[mean_indicators].mean()
    grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
    pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                            columns=['ptcp_0','multihoming'], 
                                            values=grouped_result.columns, 
                                            fill_value=0)
    d2d_veh_stats['res_wage_ptcp_0'] = pivot_result['res_wage'][True][False] if any_driver_sh else np.nan
    d2d_veh_stats['res_wage_ptcp_mh'] = pivot_result['res_wage'][True][True] if any_driver_mh else np.nan

    count_indicators = []
    mean_indicators = ['res_wage']
    if 'registered_1' in d2d_veh_reset.columns: # more than one platform
        grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'registered_1', 'multihoming'])[count_indicators].sum()
        grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'registered_1', 'multihoming'])[mean_indicators].mean()
        grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
        pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                                columns=['registered_1','multihoming'], 
                                                values=grouped_result.columns, 
                                                fill_value=0)
        d2d_veh_stats['res_wage_reg_1'] = pivot_result['res_wage'][True][False] if any_driver_sh else np.nan

        mean_indicators = ['res_wage']
        grouped_result_count = d2d_veh_reset.groupby(['repl', 'day', 'ptcp_1', 'multihoming'])[count_indicators].sum()
        grouped_result_mean = d2d_veh_reset.groupby(['repl', 'day', 'ptcp_1', 'multihoming'])[mean_indicators].mean()
        grouped_result = pd.concat([grouped_result_count, grouped_result_mean], axis=1)
        pivot_result = grouped_result.pivot_table(index=['repl','day'], 
                                                columns=['ptcp_1','multihoming'], 
                                                values=grouped_result.columns, 
                                                fill_value=0)
        d2d_veh_stats['res_wage_ptcp_1'] = pivot_result['res_wage'][True][False] if any_driver_sh else np.nan

    # For each indicator in d2d_veh, determine whether it is a 'count' or 'mean' indicator (in the population of agents)
    mean_indicators, count_indicators = [], []
    count_string_list = ["informed", "regist", "ptcp", "dist", "occ"]
    for col in d2d_veh.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)
    aggr_orig_sum = d2d_veh[count_indicators].groupby(['repl', 'day']).sum().groupby(['day']).mean() # Aggregating by taking average across 'repl' and summing over 'veh' for each column
    aggr_orig_mean = d2d_veh[mean_indicators].groupby(['repl', 'day']).mean().groupby(['day']).mean() # Aggregating by taking average across 'repl' and mean over 'veh' for each column
    
    mean_indicators, count_indicators = [], []
    for col in d2d_veh_stats.columns:
        if contains_item_from_list(col, count_string_list): # if the indicator is a count indicator
            count_indicators.append(col) # add (possibly platform-specific) indicator to count_indicators list
        else:
            mean_indicators.append(col)
    
    aggr_new_sum = d2d_veh_stats[count_indicators].groupby(['repl', 'day']).sum().groupby(['day']).mean() # Aggregating by taking average across 'repl' and summing over 'veh' for each column
    aggr_new_mean = d2d_veh_stats[mean_indicators].groupby(['repl', 'day']).mean().groupby(['day']).mean() # Aggregating by taking average across 'repl' and mean over 'veh' for each column

    aggr_sup_df = pd.concat([aggr_orig_sum, aggr_orig_mean, aggr_new_sum, aggr_new_mean], axis=1)

    # TODO: add hetoregeneity stats, e.g. standard deviations

    # Plot evolution of different replications - main indicators: participation and perceived utility
    for col in d2d_veh.columns:
        if col.startswith('out'):
            plf_id = col.split("_")[-1]
            d2d_veh['ptcp_{}'.format(plf_id)] = ~d2d_veh[col]
            aggregated = d2d_veh.groupby(['repl', 'day']).agg({'ptcp_{}'.format(plf_id): 'sum'})
            aggregated.unstack('repl').plot(kind='line', figsize=(10, 6))
            plt.xlabel('Day')
            plt.ylabel('Participation')
            plt.title('Participating drivers with platform {}'.format(plf_id))
            plt.legend(title='repl')
            plt.savefig(os.path.join(aggr_scn_path, 'evo_ptcp_sup_{}.png'.format(plf_id)))
    for col in d2d_pax.columns:
        if col.startswith('requests'):
            plf_id = col.split("_")[-1]
            aggregated = d2d_pax.groupby(['repl', 'day']).agg({'requests_{}'.format(plf_id): 'sum'})
            aggregated.unstack('repl').plot(kind='line', figsize=(10, 6))
            plt.xlabel('Day')
            plt.ylabel('Participation')
            plt.title('Participating travellers with platform {}'.format(plf_id))
            plt.legend(title='repl')
            plt.savefig(os.path.join(aggr_scn_path, 'evo_ptcp_dem_{}.png'.format(plf_id)))

    # Now select rows corresponding to equilibrium - should be done per replication
    eql_df = conv_df.groupby(['repl']).tail(conv_steady_days + moving_average_days)

    # For each perceived platform indicator, determine how many replications are needed based on values in an initial number of replications
    current_n_repl = len(d2d_pax.index.get_level_values('repl').unique()) # current number of replications to determine degrees of freedom
    req_repl_indicator = dict()
    if 'ptcp_dem_sh_1' in eql_df.columns.values:
        indicators = ['ptcp_dem_mh','ptcp_dem_sh_0','ptcp_sup_mh','ptcp_sup_sh_0','ptcp_dem_sh_1','ptcp_sup_sh_1']
    else:
        indicators = ['ptcp_dem_mh','ptcp_dem_sh_0','ptcp_sup_mh','ptcp_sup_sh_0']
    for indic in indicators:
        eql_df[indic] = pd.to_numeric(eql_df[indic], errors='coerce')
        conv_df[indic] = pd.to_numeric(conv_df[indic], errors='coerce')
        avg_repl_perc_indicator = eql_df.groupby('repl')[indic].mean().mean()
        std_repl_perc_indicator = eql_df.groupby('repl')[indic].std().mean()
        req_repl_indicator[indic] = determine_req_repl_indicator(current_n_repl, avg_repl_perc_indicator, std_repl_perc_indicator, conv_signif, conv_max_error)
    req_n_repl = max(req_repl_indicator.values())
    # Check if insufficient replications have been run for scenario
    if current_n_repl >= req_n_repl: 
        print('Success: Sufficient replications have been run for scenario {}: {} out of {} required'.format(scn_name, current_n_repl, req_n_repl))
    else:
        print('WARNING: Insufficient replications for scenario {}: {} out of {} required'.format(scn_name, current_n_repl, req_n_repl))
        print('Required replications per indicator: {}'.format(req_repl_indicator))
    
    # Store required number of replications in dataframe
    req_repl_indicator['current_n_repl'] = current_n_repl
    req_repl = req_repl.append({**req_repl_indicator, **scn_dict}, ignore_index=True)
    req_repl = req_repl.set_index(list(keys))
    
    # Plot the convergence indicators
    sup_ptcp = ['ptcp_sup_mh','ptcp_sup_sh_0','ptcp_sup_sh_1'] if 'ptcp_sup_sh_1' in eql_df.columns.values else ['ptcp_sup_mh','ptcp_sup_sh_0']
    aggregated = conv_df[sup_ptcp].groupby(['repl']).mean() #agg({col: 'mean'})
    aggregated.unstack('repl').plot(kind='line', figsize=(10, 6))
    plt.xlabel('Day')
    plt.ylabel('Drivers')
    plt.title('Fleet size')
    plt.legend(title='repl')
    plt.savefig(os.path.join(aggr_scn_path, 'ptcp_sup.png'))

    dem_ptcp = ['ptcp_dem_mh','ptcp_dem_sh_0','ptcp_dem_sh_1'] if 'ptcp_dem_sh_1' in eql_df.columns.values else ['ptcp_dem_mh','ptcp_dem_sh_0']
    aggregated = conv_df[dem_ptcp].groupby(['repl']).mean()
    aggregated.unstack('repl').plot(kind='line', figsize=(10, 6))
    plt.xlabel('Day')
    plt.ylabel('Requests')
    plt.title('Demand')
    plt.legend(title='repl')
    plt.savefig(os.path.join(aggr_scn_path, 'ptcp_dem.png'))

    # Save the aggregated dataframe for this scenario in pickle format
    with open(os.path.join(aggr_scn_path, 'aggr_dem.pkl'), 'wb') as f:
        pickle.dump(aggr_dem_df, f)
    with open(os.path.join(aggr_scn_path, 'aggr_sup.pkl'), 'wb') as f:
        pickle.dump(aggr_sup_df, f)
    with open(os.path.join(aggr_scn_path, 'mode_no_rs.pkl'), 'wb') as f:
        pickle.dump(mode_no_rs, f)
    # with open(os.path.join(aggr_res_path, 'pax_attr.pkl'), 'wb') as f:
    #     pickle.dump(pax_attr, f)
    # with open(os.path.join(aggr_res_path, 'driver_attr.pkl'), 'wb') as f:
    #     pickle.dump(driver_attr, f)
    # with open(os.path.join(aggr_res_path, 'plf_attr.pkl'), 'wb') as f:
    #     pickle.dump(plf_attr, f)
    # with open(os.path.join(aggr_res_path, 'all_pax_attr.pkl'), 'wb') as f: # TODO: how to handle all_pax statistics?
    #     pickle.dump(all_pax_attr, f)
    # TODO:  need / how to save params
    