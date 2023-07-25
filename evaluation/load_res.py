### Script for loading single scenario (considering different replications)
# For each replication we determine the equilibrium statistics
# And we check the difference between different replication
import os
path = os.getcwd()
import pandas as pd
import numpy as np
import json
import ast
from utils import create_d2d_df, create_attr_df, determine_req_repl_indicator
import math
import itertools

### ---------------------- INPUT -----------------------###

# Which scenarios?
var_dict = {'cmpt_type': ['sp'], 'dem_mh_share': [0.6], 'sup_mh_share': [0.5]}

# Required parameter values for statistical significance of equilibria
conv_signif = 0.05
conv_max_error = 0.02
conv_days = 3 # TODO: take from params file

# Results path
res_path = os.path.join(path, 'results')

### --------------------- SCRIPT --------------------- ###

# Initialise files
d2d_pax, d2d_veh, pax_attr, driver_attr, plf_attr, all_pax_attr, req_repl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Determine scenarios
keys, values_lists = zip(*var_dict.items())
scenarios = list(itertools.product(*values_lists))
# scenarios_strings = [f"{key}-{value}" for scenario in scenarios for key, value in zip(keys, scenario)]
scenario_names = ['-'.join(f"{key}-{value}" for key, value in zip(keys, scenario)) for scenario in scenarios]
scenario_names = [s.strip('cmpt_type-') for s in scenario_names]  # drop 'cmpt_type' from scn name

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
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(res_path, item)) and item.startswith(scn_name)] # Filter only the directories

    # Open replication-independent (but scenario-specific) files - i.e. agent properties TODO: saving params to be used in analysis
    f = open(os.path.join(res_path, folder_names[0], '0_params.json'))
    params = json.load(f)

    for repl_folder in folder_names:
        repl_id = int(repl_folder.split("-")[-1])
        repl_folder_path = os.path.join(res_path, repl_folder)
        for item in os.listdir(repl_folder_path):
            if item.startswith('day'):
                day_id = int(item.split("_")[1])
                if item.endswith('travs.csv'):
                    d2d_pax = create_d2d_df(scn_dict, repl_id, day_id, repl_folder_path, item, agent_type='pax', d2d_df=d2d_pax)
                else:
                    d2d_veh = create_d2d_df(scn_dict, repl_id, day_id, repl_folder_path, item, agent_type='veh', d2d_df=d2d_veh)
            if item.endswith('1_pax-properties.csv'):
                pax_attr = create_attr_df(scn_dict, repl_id, repl_folder_path, item, index_name='pax_id', attr_df=pax_attr)
                pax_attr = pax_attr.drop(columns=['Unnamed: 0'])
            if item.endswith('2_driver-properties.csv'):
                driver_attr = create_attr_df(scn_dict, repl_id, repl_folder_path, item, index_name='veh_id', attr_df=driver_attr)
            if item.endswith('3_platform-properties.csv'):
                plf_attr = create_attr_df(scn_dict, repl_id, repl_folder_path, item, index_name='id', attr_df=plf_attr)
            if item.endswith('4_out-filter-pax.csv'):
                all_pax_attr = create_attr_df(scn_dict, repl_id, repl_folder_path, item, index_name='pax_id', attr_df=all_pax_attr)
    d2d_pax = d2d_pax.sort_index(level=['repl','day','pax']) # Sort multi-index df based on day id
    d2d_veh = d2d_veh.sort_index(level=['repl','day','veh']) # Sort multi-index df based on day id

    # Now select rows corresponding to equilibrium
    eql_pax = d2d_pax.groupby(['repl', 'pax']).tail(conv_days)
    eql_veh = d2d_veh.groupby(['repl', 'veh']).tail(conv_days)

    # For each perceived platform indicator, determine how many replications are needed based on values in an initial number of replications
    current_n_repl = len(d2d_pax.index.get_level_values('repl').unique()) # current number of replications to determine degrees of freedom
    req_repl_indicator = dict()
    for col in eql_pax.columns:
        if col.startswith('init_perc') and not col.startswith('init_perc_util'):
            avg_repl_perc_indicator = eql_pax.groupby('repl')[col].mean().mean()
            std_repl_perc_indicator = eql_pax.groupby('repl')[col].mean().std()
            # std_repl_perc_indicator = eql_pax.groupby('repl')[col].std()
            req_repl_indicator[col] = determine_req_repl_indicator(current_n_repl, avg_repl_perc_indicator, std_repl_perc_indicator, conv_signif, conv_max_error)
    for col in eql_veh.columns:
        if col.startswith('init_perc') and not col.startswith('init_perc_util'):
            avg_repl_perc_indicator = eql_veh.groupby('repl')[col].mean().mean()
            std_repl_perc_indicator = eql_veh.groupby('repl')[col].mean().std()
            # std_repl_perc_indicator = eql_veh.groupby('repl')[col].std()
            req_repl_indicator[col] = determine_req_repl_indicator(current_n_repl, avg_repl_perc_indicator, std_repl_perc_indicator, conv_signif, conv_max_error)
    req_repl_indicator['current_n_repl'] = current_n_repl
    req_repl = req_repl.append({**req_repl_indicator, **scn_dict}, ignore_index=True)
    req_repl = req_repl.set_index(list(scn_dict.keys()))

# Check for which scenarios insufficient replications have been run
req_repl['req_n_repl'] = req_repl.drop(columns=['current_n_repl']).max(axis=1)
req_repl['sufficient_req'] = req_repl.current_n_repl >= req_repl.req_n_repl
if req_repl.sufficient_req.all(): 
    print('Success: Sufficient replications have been run for each scenario')
else:
    for index, row in req_repl.loc[~req_repl.sufficient_req].iterrows():
        print('Insufficient replications for scenario {}: {} out of {} required'.format(dict(zip(keys, index)), int(row.current_n_repl), math.ceil(row.req_n_repl)))

# print('Replications needed based on individual indicators: {}'.format(req_repl_indicator))
# print('Total number of replications needed to satisfy all indicators: {}, due to indicator {}'.format(math.ceil(max(req_repl_indicator.values())), max(req_repl_indicator, key=req_repl_indicator.get)))