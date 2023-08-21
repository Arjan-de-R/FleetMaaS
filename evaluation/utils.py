import pandas as pd
import os
from scipy.stats import t

def create_d2d_df_list(repl_id, day_id, repl_folder_path, item, agent_type, d2d_df_list):
    day_df = pd.read_csv(os.path.join(repl_folder_path, item), index_col=False)
    day_df['repl'] = repl_id
    day_df['day'] = day_id
    index_list = ['repl', 'day', agent_type]
    d2d_df_list = d2d_df_list + [day_df.set_index(index_list)]
    return d2d_df_list


def create_attr_df_list(repl_id, repl_folder_path, item, index_name, attr_df_list):
    df = pd.read_csv(os.path.join(repl_folder_path, item), index_col=False)
    df['repl'] = repl_id
    index_list = ['repl']
    if index_name == 'veh_id':
        df = df.rename(columns={"Unnamed: 0": "veh_id"})
    attr_df_list = attr_df_list + [df.set_index(index_list + [index_name])]
    return attr_df_list


def determine_req_repl_indicator(current_n_repl, avg_repl_perc_indicator, std_repl_perc_indicator, conv_signif, conv_max_error):
    '''determine how many replications are needed based on mean, st. dev and current number of replications'''
    crit_t = t.ppf(conv_signif, current_n_repl-1)
    req_repl_indicator = ((std_repl_perc_indicator * crit_t) / (avg_repl_perc_indicator * conv_max_error)) ** 2
    return req_repl_indicator


def contains_item_from_list(target_string, string_list):
        for item in string_list:
            if item in target_string:
                return True
        return False