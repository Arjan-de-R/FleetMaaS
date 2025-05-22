import numpy as np
import pandas as pd
import os
import networkx as nx
import math
from functools import lru_cache
from FleetPy.src.misc.globals import *
from FleetPy.src.routing.NetworkTTMatrix import NetworkTTMatrix
from MaaSSim.src_MaaSSim.d2d_demand import mode_probs
from scipy.special import erfinv

def trip_credit_cost(inData, params):
    '''Determine credit cost for each mode for travellers' trip itineraries'''

    # Modes other than ridesourcing
    inData.requests['bike_credit'] = (params.tmc.credit_mode.bike.base + inData.requests.dist_bike / 1000 * params.tmc.credit_mode.bike.dist).round()
    inData.requests['car_credit'] = (params.tmc.credit_mode.car.base + inData.requests.dist / 1000 * (params.tmc.credit_mode.car.dist + params.tmc.credit_mode.car.get('dist_add_center', 0) * inData.requests.through_center)).round()
    inData.requests['pt_credit'] = (params.tmc.credit_mode.pt.base + inData.requests.PTdistance / 1000 * params.tmc.credit_mode.pt.dist).round()
    
    def rs_plf_credit(req_dist, service_type, through_center):
        '''determine required credits for solo and pooling trip for a given trip request'''
        if service_type == 'solo':
            trip_credit = params.tmc.credit_mode.solo.base + req_dist / 1000 * (params.tmc.credit_mode.solo.dist + params.tmc.credit_mode.solo.get('dist_add_center', 0) * through_center)
        else:
            trip_credit = params.tmc.credit_mode.pool.base + req_dist / 1000 * (params.tmc.credit_mode.pool.dist + params.tmc.credit_mode.pool.get('dist_add_center', 0) * through_center)
        
        return trip_credit

    # Ridesourcing, depending on solo or pooling
    inData.requests['rs_credit'] = inData.requests.apply(lambda row: np.array([rs_plf_credit(row.dist, params.platforms.service_types[plat_id], row.through_center) for plat_id in range(0,len(params.platforms.service_types))]).round(decimals=0), axis=1)

    return inData.requests


def deduct_credit_mode(chosen_mode, car_credit, bike_credit, pt_credit, rs_credit):
    '''Determine the number of credits that is deducted from a traveller's balance based on the chosen mode'''

    if chosen_mode == 'car':
        credit_cost = car_credit
    elif chosen_mode == 'bike':
        credit_cost = bike_credit
    elif chosen_mode == 'pt':
        credit_cost = pt_credit
    elif chosen_mode.startswith("rs"):
        plf_id = int(chosen_mode.split("_")[-1])
        credit_cost = rs_credit[plf_id]
    else: # not enough credit to travel
        credit_cost = 0

    return credit_cost


def buy_table_dimensions(params):
    '''Determine which values are included in the table with quantities depending on price and credit balance'''
    min_price_step = params.tmc.price.get('step', 0.01)
    max_buy_quant = params.tmc.get('max_buy_day', None)

    # Determine dimensions of database: balance quantities and credit price levels
    price_values = np.arange(params.tmc.price.get('min',0.01), params.tmc.price.get('max', 10) + min_price_step, min_price_step)
    if max_buy_quant is None:
        buy_values = np.arange(-max_buy_quant,max_buy_quant+1)
        value_dict = {'price': price_values, 'quantity': buy_values}
    else:
        value_dict = {'price': price_values}

    return value_dict


def market_transactions(market_orders):
    '''Determine how many credits are bought and sold by each individual (and rejected orders) based on credit price'''

    # Determine which orders are satisfied
    buy_orders = market_orders[market_orders > 0].sort_values(ascending=False)
    sell_orders = market_orders[market_orders < 0].sort_values(ascending=True)
    satisfy_net_buy_quant = market_orders.copy()
    if market_orders.sum() < 0: # supply exceeds demand for the current credit price
        satisfy_net_buy_quant[satisfy_net_buy_quant < 0] = 0
        remaining_buy_quant = buy_orders.sum()
        # Satisfy sell orders from large to small (as long as credits are available)
        for index, value in sell_orders.items():
            assert remaining_buy_quant >= 0
            if remaining_buy_quant == 0:
                break
            if abs(value) > remaining_buy_quant:
                satisfy_net_buy_quant[index] = -remaining_buy_quant
                remaining_buy_quant = 0
            else:
                satisfy_net_buy_quant[index] = value
                remaining_buy_quant += value
    elif market_orders.sum() > 0: # demand exceeds supply
        satisfy_net_buy_quant[satisfy_net_buy_quant > 0] = 0
        remaining_sell_quant = abs(sell_orders).sum()
        # Satisfy buy orders from large to small (as long as credits are available)
        for index, value in buy_orders.items():
            assert remaining_sell_quant >= 0
            if remaining_sell_quant == 0:
                break
            if value > remaining_sell_quant:
                satisfy_net_buy_quant[index] = remaining_sell_quant
                remaining_sell_quant = 0
            else:
                satisfy_net_buy_quant[index] = value
                remaining_sell_quant -= value
    denied_orders = market_orders - satisfy_net_buy_quant

    return satisfy_net_buy_quant, denied_orders


def update_balances(inData, satisfied_orders, denied_orders):
    '''Update credit and monetary balance based on satisfied trading orders (buy and sell) and price'''
    inData.passengers['net_purchase'] = satisfied_orders.copy()
    inData.passengers['denied_order'] = denied_orders.copy()

    return inData.passengers


def save_tmc_market_indicators(inData, result_path, mode_choice_iter, transp_model_iter, credit_price, satisfied_orders, denied_orders):
    '''Append day's trading market indicators to csv'''
    market_indic = pd.DataFrame([{'i_mode_choice': mode_choice_iter, 'price': credit_price, 'transaction_volume': satisfied_orders[satisfied_orders > 0].sum(), 'oversupply': denied_orders.sum()}])
    if mode_choice_iter == 0 and transp_model_iter == 0: # include the headers on the first day
        if os.path.exists(os.path.join(result_path,'6_tmc-indicators.csv')):
            os.remove(os.path.join(result_path,'6_tmc-indicators.csv'))
        market_indic.to_csv(os.path.join(result_path,'6_tmc-indicators.csv'), mode='a', index=False, header=True)
    else:
        market_indic.to_csv(os.path.join(result_path,'6_tmc-indicators.csv'), mode='a', index=False, header=False)

    return 0


def order_per_price(pax, params, value_dict, rem_days, expected_price):
    # Load regression-based order function if specifically specified, otherwise utility-based order function
    if params.tmc.pref_trading.get('method', False) == "regression":
        order_func = order_per_price_regression
        if expected_price == None:
            expected_price = value_dict['price']  # on the first day, if there is no price expectation, the expected price is the current price
        quantity = order_func(pax, params, value_dict, rem_days, expected_price)
    else:
        order_func = order_per_price_util
        quantity = order_func(pax, params, value_dict, rem_days)

    # Ensure that no credits are sold when price equals 0
    zero_price_index = np.where(value_dict['price'] == 0)[0]
    if len(zero_price_index) > 0:
        zero_price_index = zero_price_index[0]
        if quantity[zero_price_index] < 0:
            quantity[zero_price_index] = 0

    return quantity


def order_per_price_util(pax, params, value_dict, rem_days):
    '''Determine a traveller's buy/sell order for each possible credit price depending on their balance and time left to spend credits, using utility of money and balance'''

    def util_buy(params, balance, price, buy_quant, rem_days, max_balance=np.inf, probabilistic=True):
        '''Determine utility associated with buying and selling, trading off financial gains/costs and utility of having credits'''
        value_credit_in_balance = params.tmc.pref_trading.get('balance_util_percept', 1) # util/credit
        excess_credit_scaling_param = params.tmc.pref_trading.get('excess_credit_param', 1)

        # Determine utility of spent/earned money (opportunity cost) when buying/selling
        util_cost = np.array([price]).T * buy_quant * params.tmc.pref_trading.get('beta_monetary', -1)
        # Utility of having additional / fewer credits in balance than before
        util_orig_balance = value_credit_in_balance * np.log(excess_credit_scaling_param * (balance / rem_days) + 1)
        util_new_balance = value_credit_in_balance * np.log(excess_credit_scaling_param * ((balance + buy_quant) / rem_days) + 1)
        util_balance_purchase = util_new_balance - util_orig_balance
        util_balance_purchase[(-buy_quant > balance) | ((balance + buy_quant) > max_balance)] = np.nan # np.nan for infeasbile quantities (selling more than balance or buying over allowed balance)
        net_util_buy = util_balance_purchase + util_cost
        if probabilistic:
            std_dev_error_term = np.sqrt((np.pi**2) / 6)
            error_terms = np.random.normal(loc=0, scale=std_dev_error_term, size=net_util_buy.shape[1])
            net_util_buy = net_util_buy + error_terms

        return net_util_buy

    credit_balance = pax.tmc_balance

    if rem_days > 0:
        max_balance = np.max(value_dict['balance'])
        util_buy_price_quant = util_buy(params, credit_balance, value_dict['price'], value_dict['quantity'], rem_days, max_balance=max_balance) # observed utility
        max_indices = np.nanargmax(util_buy_price_quant, axis=1)
        quantity = value_dict['quantity'][max_indices]
    else:
        quantity = np.zeros(len(value_dict['price']))

    return quantity

  
def order_per_price_regression(pax, params, value_dict, rem_days, expected_price, ever_bought=False, ever_sold=False):
    """
    Determine a traveller's buy/sell order for each possible credit price depending on their balance and time left to spend credits, using regression
    
    Parameters:
    - params: parameters of the simulation, including regression function type and coefficients
    - value_dict: dictionary with balance, price and quantity values    
    - rem_days: remaining days of the TMC period
    - credit_balance: current credit balance of the traveller
    - expected_price: expected price of the credit (from past prices)
    - ever_bought: boolean indicating whether the traveller has ever bought credits before
    - ever_sold: boolean indicating whether the traveller has ever sold credits before

    Returns:
    - quantity: buy/sell order for each possible credit price
    """

    def linear_regression():
        quantity = beta_constant + beta_balance * (credit_balance - reference_balance) + beta_price * (value_dict['price'] - expected_price) + beta_days * rem_days + beta_hist_buy * +(ever_bought) + beta_hist_sell * +(ever_sold) + error_term

        return quantity
    
    def linear_regression_expected_usage():
        '''linear regression but based on expected credit usage'''
        credit_balance_per_day = credit_balance / rem_days
        quantity = beta_constant + beta_balance * (credit_balance_per_day - pax.expected_credit_usage_per_price) + beta_price * (value_dict['price'] - expected_price) + beta_days * rem_days + beta_hist_buy * +(ever_bought) + beta_hist_sell * +(ever_sold) + error_term

        return quantity
    
    def logarithmic_regression():
        transformed_quant = beta_constant + beta_balance * (np.log(credit_balance) - np.log(reference_balance)) + beta_price * (value_dict['price'] - expected_price) + beta_days * rem_days + beta_hist_buy * +(ever_bought) + beta_hist_sell * +(ever_sold) + error_term
        quantity = np.exp(transformed_quant) - 1 if transformed_quant > 0 else -np.exp(-transformed_quant) - 1

        return quantity
    
    # def linear_regression_price_factor():
    #     '''linear regression but based on price factor relative to expectation, instead of price difference with expectation'''
    #     price_factor = value_dict['price'] / expected_price
    #     price_factor = 1 / price_factor if price_factor < 1 else price_factor  # 5 means, 5 times more expensive than expectation, -5 means 5 times cheaper
    #     quantity = beta_constant + beta_balance * (credit_balance - reference_balance) + beta_price * price_factor + beta_days * rem_days + beta_hist_buy * +(ever_bought) + beta_hist_sell * +(ever_sold) + error_term

    #     # what if perc price is zero
    #     quantity = beta_constant + beta_balance * (credit_balance - reference_balance) + beta_price * (value_dict['price'] / expected_price) + beta_days * rem_days + beta_hist_buy * +(ever_bought) + beta_hist_sell * +(ever_sold) + error_term
    #     return 0

    if rem_days > 0:
        credit_balance = pax.tmc_balance
        reference_balance = rem_days * params.tmc.allocated_credits_per_day
        regression_type = params.tmc.pref_trading.get('regression_type', 'linear')
        error_term = np.random.normal(0, params.tmc.pref_trading.get('sd_error_term', 0))

        beta_constant = pax.get('beta_constant', params.tmc.pref_trading.get('beta_constant', 0))
        beta_balance = pax.get('beta_balance', params.tmc.pref_trading.get('beta_balance', 0))
        beta_price = pax.get('beta_price', params.tmc.pref_trading.get('beta_price', 0))
        beta_days = pax.get('beta_days', params.tmc.pref_trading.get('beta_days', 0))
        beta_hist_buy = pax.get('beta_hist_buy', params.tmc.pref_trading.get('beta_hist_buy', 0))
        beta_hist_sell = pax.get('beta_hist_sell', params.tmc.pref_trading.get('beta_hist_sell', 0))

        if regression_type == 'linear':
            if pax.get('expected_credit_usage', None) is None and pax.get('expected_credit_usage_per_price', None) is None:
                quantity = linear_regression()
            else:
                quantity = linear_regression_expected_usage()
        elif regression_type == 'logarithmic':
            quantity = logarithmic_regression()
        else:   
            raise ValueError("Regression type not supported")
        
        max_balance = np.max(value_dict['balance'])
        quantity[(credit_balance + quantity > max_balance)] = max_balance - credit_balance # buy as much as possible without exceeding maximum balance if you would like to buy more
        quantity[(-quantity > credit_balance)] = -credit_balance # sell all available credits if you would like to sell more than your balance
        quantity = np.round(quantity, decimals=0) # ensure integer order quantities

    else:
        quantity = np.zeros(len(value_dict['price']))

    return quantity


def shortest_path_through_area(nw, osm_to_fp_id, nodes_in_area, o_id, d_id):
    "determine if shortest path (in FleetPy) between two nodes (o_id, d_id) passes through pre-defined area (list of nodes), returns Boolean"
    fp_to_osm_id = {v: int(k) for k, v in osm_to_fp_id.items()}
    sp_passes_through_area = False
    o_pos = (osm_to_fp_id[str(o_id)], None, None)
    d_pos = (osm_to_fp_id[str(d_id)], None, None)
    od_node_list = nw.return_best_route_1to1(o_pos, d_pos)
    for node_id in od_node_list:
        osm_id = fp_to_osm_id[node_id]
        if osm_id in nodes_in_area:
            sp_passes_through_area = True
            break

    return sp_passes_through_area

def prep_fp_shortest_paths(params):
    '''prepare shortest path generator in FleetPy for determining whether requests traverse certain area'''
    osm_to_fp_id = convert_fpid_to_osmid(params)
    nw = init_networktt_fp_class(params)

    return nw, osm_to_fp_id


def convert_fpid_to_osmid(params):
    '''convert fleetpy node id's to osm id's'''
    graphml_file = params.paths.G
    graph = nx.read_graphml(graphml_file)
    osm_to_fp_id = {}
    c_id = 0
    for node in graph.nodes:
        osm_to_fp_id[graph.nodes[node]["osmid"]] = c_id
        c_id += 1
    
    return osm_to_fp_id


def init_networktt_fp_class(params):
    network_name = params.city.split(",")[0]
    network_dir = os.path.join("FleetPy", "data", "networks", network_name)
    nw = NetworkTTMatrix(network_dir)

    return nw


def ASCs_and_economic_attributes(params):
    "determine properties of alternative modes for group of travellers"
    prefs = params.evol.travellers.mode_pref

    def vot_from_income():
        "determine travellers' income and convert to Value of Time"
        mean_income = params.evol.travellers.get('mean_income', 30000)  # mean annual income in euro
        gini_income = prefs.get('gini', 0.3)  # Gini coefficient of the income distribution

        # Draw income of travellers based on lognormal distribution
        lognorm_std = 2 * erfinv(gini_income)
        lognorm_mean = np.log(mean_income) - (lognorm_std ** 2) / 2
        income = np.random.lognormal(lognorm_mean, lognorm_std, params.nP)  # euro/h

        # Parameters for converting income to VoT
        beta = params.evol.travellers.get('baseline_log_VoT', 0)      # Baseline log VoT
        gamma = params.evol.travellers.get('income_elasticity', 0.25)  # Income elasticity of VoT
        sigma = params.evol.travellers.get('random_var_income', 1)     # Random variation

        # Generate normal-distributed random variation
        z = np.random.normal(0, 1, params.nP)

        # Compute VoT using the log-log model
        log_VoT = beta + gamma * np.log(income) + sigma * z
        VoT = np.exp(log_VoT)  # Convert back from log scale

        return income, VoT

    vot_determination = params.evol.travellers.get('vot_determination', 'direct')  # direct vot determination based on distribution or from income
    if vot_determination == 'distribution':
        # Draw Value of Time and corresponding beta's for travellers
        vot = np.random.lognormal(mean=prefs.ivt_mean_lognorm, sigma=prefs.ivt_sigma_lognorm, size=params.nP) * (-1) / prefs.beta_cost * 60  # VoT in euro/h
        income = None
    elif vot_determination == 'from_income':
        income, vot = vot_from_income()

    # Draw mode preferences (ASCs) for travellers
    ASC_car = np.random.normal(prefs.ASC_car, prefs.ASC_car_sd, params.nP)
    ASC_bike = np.random.normal(0, prefs.ASC_bike_sd, params.nP)

    # PT alternative - if included in simulation
    if params.paths.get('PT_trips',False):
        ASC_pt = np.random.normal(prefs.ASC_pt, prefs.ASC_pt_sd, params.nP)
    else:
        ASC_pt = -np.inf

    ASCs = pd.DataFrame({'bike': ASC_bike, 'car': ASC_car, 'pt': ASC_pt})
    
    return income, ASCs, vot


def prefs_travs_tmc(inData, params):
    "draw mode preferences for the group of travellers"
    prefs = params.evol.travellers.mode_pref
    passengers = inData.passengers

    if 'VoT' in passengers.columns:
        vot = passengers['VoT']
        _, ASCs, _ = ASCs_and_economic_attributes(params)
    else:
        income, ASCs, vot = ASCs_and_economic_attributes(params)
        passengers['VoT'] = vot
        passengers['income'] = income
    passengers['ASC_bike'] = ASCs.bike
    passengers['ASC_car'] = ASCs.car
    passengers['ASC_pt'] = ASCs.pt
    passengers['ASC_rs'] = np.random.normal(prefs.ASC_rs, prefs.ASC_rs_sd,len(inData.passengers))
    passengers['ASC_pool'] = passengers.ASC_rs + np.random.uniform(prefs.min_wts_constant, 0, len(inData.passengers))

    return passengers


def mode_preday_plf_choice_tmc(inData, params, **kwargs):
    "determine the mode at the start of a day for a pool of travellers (if they are single-homing yet possibly registered with more than 1 platform and still have to choose)"
    requests = inData.requests
    passengers = inData.passengers
    df = passengers.copy()
    df['tmc_balance'] = df.tmc_balance if params.dem_mgmt == 'tmc' else 0
    credit_price = kwargs.get('credit_price')
    perc_congest_factor = kwargs.get('perc_congest_factor', 1)
    mode_attr = {}
    utils = {}

    ## Establish attributes in mode choice for each mode
    mode_attr = mode_attributes(params, requests, passengers, inData, perc_congest_factor, mode_attr)

    # Determine utility of each mode
    for mode in ['bike', 'car', 'pt', 'rs']:
        utils[mode] = util_mode(params, passengers, mode_attr[mode], df, credit_price) # utils if credit balance was not a constraint
        utils[mode] = apply_insufficient_balance(utils[mode], mode_attr[mode]['credits'], df.tmc_balance, mode) # utils considering one's credit balance
        if mode == 'rs':
            utils[mode], chosen_plf = choose_ridehailing_platform(inData, df, utils[mode])
        if mode == 'car' and params.dem_mgmt == 'lpr': # license plate rationing - cars allowed to drive on odd / even days
            utils[mode] = apply_license_plate_rationing(inData, utils[mode], kwargs)
    utils_df = pd.DataFrame.from_dict(utils)

    ## MODE CHOICE
    probabilities = mode_probs(utils_df)
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities['pref_rs_plf'] = chosen_plf

    df['U_bike'] = utils['bike']
    df['U_car'] = utils['car']
    df['U_pt'] = utils['pt']
    if params.dem_mgmt == 'tmc':
        # opt out if not enough credit to travel (for any mode)
        probabilities['insuff_credit'] = df.apply(lambda row: (row.U_bike == -math.inf) and (row.U_car == -math.inf) and (row.U_pt == -math.inf) and (row.U_rs == -math.inf), axis=1)
        probabilities['decis'] = probabilities.apply(lambda row: "no_mode_available" if row.insuff_credit else row.decis, axis=1)

    probabilities['decis'] = probabilities.apply(lambda row: row.decis + '_' + str(row.pref_rs_plf) if row.decis == 'rs' else row.decis, axis=1)
    passengers['mode_day'] = probabilities.decis

    passengers['U_bike'] = utils['bike']
    passengers['U_car'] = utils['car']
    passengers['U_pt'] = utils['pt']
    
    requests['chosen_mode_perc_gtt'] = return_gtt_chosen_mode(passengers, mode_attr)

    return passengers, requests


def expected_credit_usage_and_actual_mode_choice(inData, params, **kwargs):
    """determine:
    - the expected credit usage based on the probability of choosing each mode for the expected credit price
    - the actual mode choice based on the utility of each mode and the probability of choosing each mode
    """
    requests = inData.requests
    passengers = inData.passengers
    df = passengers.copy()
    df['tmc_balance'] = df.tmc_balance if params.dem_mgmt == 'tmc' else 0
    credit_price = kwargs.get('credit_price', 0)
    credit_price = 0 if credit_price is None else credit_price
    # credit_price = perc_credit_price
    mode_attr = {}
    utils = {}

    ## Establish attributes in mode choice for each mode
    mode_attr = mode_attributes(params, requests, passengers, inData, mode_attr)

    # Determine utility of each mode
    for mode in ['bike', 'car', 'pt', 'rs']:
        utils[mode] = util_mode(params, passengers, mode_attr[mode], df, credit_price) # utils if credit balance was not a constraint
        # utils[mode] = apply_insufficient_balance(unconstrained_utils[mode], mode_attr[mode]['credits'], df.tmc_balance, mode) # utils considering one's credit balance
        if mode == 'rs':
            utils[mode], rs_plf_probs = ridehailing_platform_util_and_plf_prob(inData, df, utils[mode])
    utils_df = pd.DataFrame.from_dict(utils)

    ## Expected credit usage
    probabilities = mode_probs(utils_df)
    mode_credit_costs_df = pd.DataFrame({
        'bike': mode_attr['bike']['credits'],
        'car': mode_attr['car']['credits'],
        'pt': mode_attr['pt']['credits'],
        'rs': mode_attr['rs']['credits']
    }, index=passengers.index)
    mode_credit_costs_df['rs_plf_probs'] = rs_plf_probs
    mode_credit_costs_df['rs'] = mode_credit_costs_df.apply(lambda row: np.array([row.rs_plf_probs[plf] * row.rs[plf] for plf in range(len(row.rs))]).sum(), axis=1)
    mode_credit_costs_df = mode_credit_costs_df.drop(columns=['rs_plf_probs']) #.rename(columns={'rs_credit_cost': 'rs'})
    expected_credit_usage = (probabilities * mode_credit_costs_df).sum(axis=1)

    ## Actual mode choice
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities['chosen_rs_plf'] = rs_plf_probs.apply(lambda row: np.random.choice(len(row), p=np.nan_to_num(row)))
    probabilities['decis'] = probabilities.apply(lambda row: row.decis + '_' + str(row.chosen_rs_plf) if row.decis == 'rs' else row.decis, axis=1)
    probs = np.vstack(rs_plf_probs.values)
    probabilities['rs_0'] = probabilities['rs'].values * probs[:, 0]
    probabilities['rs_1'] = probabilities['rs'].values * probs[:, 1]
    probabilities.drop(columns=['rs'], inplace=True)

    chosen_mode_perc_gtt = return_gtt_chosen_mode(probabilities['decis'], mode_attr)
    prob_mode_attr = dict()
    prob_mode_attr['gtt'] = return_expected_attr(probabilities, mode_attr, attr_label='gtt')
    prob_mode_attr['cost'] = return_expected_attr(probabilities, mode_attr, attr_label='cost')
    prob_mode_attr['credits'] = return_expected_attr(probabilities, mode_attr, attr_label='credits')
    prob_mode_attr['constant'] = return_expected_attr(probabilities, mode_attr, attr_label='constant')

    return expected_credit_usage, probabilities, chosen_mode_perc_gtt, prob_mode_attr


def expected_credit_usage_and_mode_choice_per_price(inData, params, value_dict, perc_congest_factor):
    "determine the expected credit usage based on the probability of choosing each mode for all possible credit prices"
    expected_usage_dict, probabilities, gtt_chosen_mode = np.array([expected_credit_usage_and_actual_mode_choice(inData, params, credit_price=price, perc_congest_factor=perc_congest_factor) for price in value_dict['price']])

    return expected_usage_dict.T, probabilities, gtt_chosen_mode


def mode_attributes(params, requests, passengers, inData, mode_attr):
    ''' Determine the attributes (time, cost, etc.) of all modes'''
    prefs = params.evol.travellers.mode_pref
    props = params.alt_modes
    df = passengers.copy()
    
    # Bike
    mode_attr['bike'] = {}
    mode_attr['bike']['gtt'] = requests.ttrav_bike * prefs.bike_multip
    mode_attr['bike']['cost'] = 0
    mode_attr['bike']['credits'] = requests.bike_credit if params.dem_mgmt == 'tmc' else 0
    mode_attr['bike']['constant'] = passengers.ASC_bike
    mode_attr['bike']['option'] = passengers.bike_option if 'bike_option' in passengers.columns else pd.Series(True, index=mode_attr['bike']['constant'].index)
    # Private car
    mode_attr['car'] = {}
    car_ivt = requests.ttrav * passengers.expected_ttf  # assumed same as RS (solo)
    requests['car_park_cost'] = props.car.park_cost
    if props.car.diff_parking:
        requests['dest_center'] = requests.apply(lambda x: inData.nodes.center.loc[x.destination], axis=1)
        requests.loc[requests.dest_center, 'car_park_cost'] = props.car.park_cost_center
    mode_attr['car']['cost'] = props.car.km_cost * (requests.dist / 1000) + requests.car_park_cost
    if params.dem_mgmt == 'cgp':
        if not params.get('city_charge', False):
            mode_attr['car']['cost'] = mode_attr['car']['cost'] + requests.through_center * params.zone_charge.get('car', 5)
        else:
            mode_attr['car']['cost'] = mode_attr['car']['cost'] + params.city_charge.get('car', 5)
    mode_attr['car']['gtt'] = prefs.access_multip * props.car.access_time + car_ivt # generalised travel time
    mode_attr['car']['constant'] = passengers.ASC_car
    mode_attr['car']['credits'] = requests.car_credit if params.dem_mgmt == 'tmc' else 0
    mode_attr['car']['option'] = passengers.car_option if 'car_option' in passengers.columns else pd.Series(True, index=mode_attr['car']['constant'].index)
    # Public transport (if included)
    if params.paths.get('PT_trips',False):
        mode_attr['pt'] = {}
        pt_trans_pen = requests.transfers * prefs.transfer_pen
        pt_ivt = requests.transitTime + pt_trans_pen
        pt_wait = requests.waitingTime
        pt_access = requests.walkDistance / params.speeds.walk
        mode_attr['pt']['gtt'] = prefs.access_multip * pt_access + prefs.wait_multip * pt_wait + pt_ivt
        mode_attr['pt']['cost'] = requests.PTfare
        mode_attr['pt']['constant'] = passengers.ASC_pt
        mode_attr['pt']['credits'] = requests.pt_credit if params.dem_mgmt == 'tmc' else 0
        mode_attr['pt']['option'] = passengers.pt_option if 'pt_option' in passengers.columns else pd.Series(True, index=mode_attr['pt']['constant'].index)
    # Ride-hailing
    mode_attr['rs'] = {}
    if params.dem_mgmt == 'cgp':
        congestion_charge = []
        for plf in range(len(params.platforms.service_types)):
            if not params.get('city_charge', False):
                plf_charge = params.zone_charge.get('solo', 5) if params.platforms.service_types[plf] == 'solo' else params.zone_charge.get('pool', 0)
            else:
                plf_charge = params.city_charge.get('solo', 5) if params.platforms.service_types[plf] == 'solo' else params.city_charge.get('pool', 0)
            congestion_charge.append(plf_charge)
        mode_attr['rs'] = rs_attr_tmc(inData, params, df.expected_wait, df.expected_ivt, df.expected_km_fare, inData.requests.dist, congestion_charge=congestion_charge) # congestion delay already included in expectation
    else:
        mode_attr['rs'] = rs_attr_tmc(inData, params, df.expected_wait, df.expected_ivt, df.expected_km_fare, inData.requests.dist) # congestion delay already included in expectation
    mode_attr['rs']['credits'] = requests.rs_credit.copy() if params.dem_mgmt == 'tmc' else 0
    mode_attr['rs']['option'] = passengers.rs_option if 'rs_option' in passengers.columns else pd.Series(True, index=mode_attr['rs']['constant'].index)

    return mode_attr


def util_mode(params, passengers, mode_attr, tmc_balance, credit_price):
    '''Determine the utility of an invidual mode'''
    if params.evol.travellers.mode_pref.get('credit_percept', "monetary") == "monetary":    # convert credit charge to monetary costs
        mode_util = util_credit_to_cost(params, mode_attr, credit_price, passengers.VoT)
    else: # credit costs perceived separately in utility
        mode_util = util_credit_time(params, mode_attr, credit_price, tmc_balance, passengers.VoT)
    # Remove option from choice set if not available
    mode_util = mode_util.where(mode_attr['option'], -math.inf)

    return mode_util


def rs_attr_tmc(inData, params, rs_wait, rs_ivt, rs_km_fare, rs_dist, trav_vot=False, trav_ASC=False, congestion_charge=None, through_center=False):
    '''determine main ridesourcing attributes (time, costs, credits), either aggregated (if no trav_vot is provided) or for an individual traveller'''
    passengers = inData.passengers
    prefs = params.evol.travellers.mode_pref
    
    if not trav_vot: # determine utility for all passengers
        rs_fare = np.ones(len(inData.passengers)) * params.platforms.base_fare + rs_km_fare * rs_dist / 1000
        # TODO: different minimum fare for pooling provider (= (1-discount) * min_solo_fare)
        # rs_fare[rs_fare < params.platforms.min_fare] += params.platforms.min_fare # min fare for solo ride
        rs_fare =  rs_fare.apply(lambda arr: np.maximum(params.platforms.min_fare, arr))
        if congestion_charge is not None:
            df_cost = pd.DataFrame()
            df_cost['rs_fare'] = rs_fare
            if not params.get('city_charge', False): # zone charge
                df_cost['congest_charge'] = inData.requests.apply(lambda row: np.array(congestion_charge) * row.through_center, axis=1)
            else: # city-wide congestion charge
                df_cost['congest_charge'] = inData.requests.apply(lambda row: np.array(congestion_charge), axis=1) 
            rs_fare = df_cost.apply(lambda row: row.rs_fare + row.congest_charge, axis=1)
        ASC_rs = passengers.ASC_rs
    else:  # only for an individual traveller
        rs_fare = params.platforms.base_fare + rs_km_fare * rs_dist / 1000
        rs_fare = max(rs_fare, params.platforms.min_fare)
        if congestion_charge is not None:
            congestion_charge = np.array(congestion_charge) * through_center
            rs_fare = rs_fare + congestion_charge
        ASC_rs = trav_ASC

    gtt = prefs.wait_multip * rs_wait + rs_ivt
    attributes = {'gtt': gtt, 'cost': rs_fare, 'constant': ASC_rs}
    
    return attributes


def util_credit_time(params, attr, credit_price, balance, VoT):
    """determine mode utility depending on generalised travel time, normal costs, credit costs and balance, when credit costs are perceived separately from cost"""
    prefs = params.evol.travellers.mode_pref
    mean_beta_time = math.exp(prefs.ivt_mean_lognorm + (prefs.ivt_sigma_lognorm**2)/2) # util/min
    mean_beta_time_credit = math.exp(prefs.ivt_credit_mean_lognorm + (prefs.ivt_credit_sigma_lognorm**2)/2) # credit/min
    beta_time = VoT * prefs.beta_cost / 3600  # util/s

    # determine util/credit depending on credit price and balance
    beta_credit = -mean_beta_time / (mean_beta_time_credit + prefs.beta_credit_price * credit_price + prefs.beta_balance * balance)
    # determine utility
    mode_util = attr['constant'] + beta_credit * attr['credits'] + prefs.beta_cost * attr['cost'] + beta_time * attr['gtt']

    return mode_util


def util_credit_to_cost(params, attr, credit_price, VoT):
    """determine mode utility depending on generalised travel time, normal costs and historical credit costs, when credit charge is perceived as monetary cost"""
    prefs = params.evol.travellers.mode_pref
    beta_time = VoT * prefs.beta_cost / 3600  # util/s

    # Convert credit price to cost
    total_cost = attr['cost'] + credit_price * attr['credits']

    # Determine utility
    mode_util = attr['constant'] + prefs.beta_cost * total_cost + beta_time * attr['gtt']

    return mode_util


def apply_insufficient_balance(utils, credit_costs, balance, mode):
    '''exclude mode (other than ridesourcing) from choice set if one has insufficient credits for this mode by setting utility to -inf'''
    df = pd.DataFrame()
    df['util'] = utils
    df['credit_cost'] = credit_costs
    df['balance'] = balance
    df['suff_balance'] = df.apply(lambda row: row.credit_cost <= row.balance, axis=1)
    if mode == 'rs':
        df['util'] = df.apply(lambda row: util_suff_balance_rs(row), axis=1)
    else:
        df['util'] = df.apply(lambda row: row.util if row.suff_balance else -math.inf, axis=1)
    return df['util']

def util_suff_balance_rs(row):
    '''return -inf for ridesourcing platforms for which user has insufficient credit'''
    row.util[row.suff_balance == False] = -math.inf
    return row.util

def unregist_to_nan(arr):
    arr[~arr] = np.nan
    return arr

def util_rs_plf(row):
    '''determine utility of ridesourcing option based on specific platform'''
    if not np.all(np.isneginf(row.U_rs_plf)):  # enough credit to use at least one of the platforms
        chosen_plf_index = np.random.choice(len(row.prob_plf), p=np.nan_to_num(row.prob_plf))
        U_rs = row.U_rs_plf[chosen_plf_index]
    else: # not enough credit for ridesourcing
        chosen_plf_index = 0
        U_rs = -math.inf

    return U_rs, int(chosen_plf_index)


def return_gtt_chosen_mode(chosen_mode, mode_attr):
    '''returns generalised travel time of chosen mode for all travellers'''

    df = pd.DataFrame()
    df['mode_day'] = chosen_mode
    df['gtt_bike'] = mode_attr['bike']['gtt']
    df['gtt_car'] = mode_attr['car']['gtt']
    df['gtt_pt'] = mode_attr['pt']['gtt']
    df['gtt_rs'] = mode_attr['rs']['gtt']

    gtt_chosen_mode = df.apply(lambda row: seek_indiv_mode_gtt(row), axis=1)

    return gtt_chosen_mode


def return_expected_attr(probabilities, mode_attr, attr_label=None):
    '''returns expected attribute based on probability of choosing each mode for all travellers'''

    # Remove non-mode columns if present
    probabilities = probabilities.copy()
    for col in ['decis', 'chosen_rs_plf']:
        if col in probabilities.columns:
            probabilities = probabilities.drop(columns=col)

    # Build GTT DataFrame for each mode
    df = pd.DataFrame({
        'bike': mode_attr['bike'][attr_label],
        'car': mode_attr['car'][attr_label],
        'pt': mode_attr['pt'][attr_label],
        'rs': mode_attr['rs'][attr_label]
    })

    # Handle ride-hailing platforms (rs_0, rs_1, ...)
    if any(col.startswith('rs_') for col in probabilities.columns):
        rs_cols = [col for col in probabilities.columns if col.startswith('rs_')]
        for col in rs_cols:
            idx = int(col.split('_')[1])
            df[col] = mode_attr['rs'][attr_label].apply(lambda arr: arr[idx] if isinstance(arr, (list, np.ndarray)) and len(arr) > idx else np.nan)
    elif 'rs' in probabilities.columns:
        df['rs'] = mode_attr['rs'][attr_label]

    # Multiply probabilities by GTT for each mode, then sum across modes
    expected_attr = (probabilities * df).sum(axis=1)

    return expected_attr


def seek_indiv_mode_gtt(row):
    '''for individual traveller, return gtt corresponding to the chosen mode'''

    if row.mode_day == 'bike':
        gtt = row.gtt_bike
    elif row.mode_day == 'car':
        gtt = row.gtt_car
    elif row.mode_day == 'pt':
        gtt = row.gtt_pt
    elif row.mode_day == 'rs_0':
        gtt = row.gtt_rs[0]
    elif row.mode_day == 'rs_1':
        gtt = row.gtt_rs[1]
    else:
        gtt = None

    return gtt


def determine_congestion(params, inData, network_name, fp_run_id, fleetpy_dir, fleetpy_study_name):
    '''determine congestion factor which may vary during the day based on the vehicle kilometres of car and ridesourcing rides in each period, which are also returned'''

    def driving_time_per_time_period(ttf_df, fleet_movements, mode='rs'):
        '''determine the driving time in each time period based on the start time and end time of the driving activity'''

        ttf_df['travel_time'] = 0.0

        # Iterate over each row in fleet_movements
        for _, row in fleet_movements.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']

            # Iterate over each time period in ttf_df
            for i, period in ttf_df.iterrows():
                period_start = np.float64(i)
                period_end = ttf_df.index[ttf_df.index.get_loc(i) + 1] if ttf_df.index.get_loc(i) + 1 < len(ttf_df) else np.inf

                # Calculate the overlap between the driving activity and the time period
                overlap_start = max(start_time, period_start)
                overlap_end = min(end_time, period_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # For private car, time needs to be multiplied by the congestion factor still (for rs this is already done in sim)
                if mode == 'car':
                    # Calculate the congestion factor based on the time period
                    tt_factor = ttf_df.at[i, 'travel_time_factor']
                    overlap_duration *= tt_factor

                # Add the overlap duration to the total duration for this time period
                ttf_df.at[i, 'travel_time'] += overlap_duration

        return ttf_df['travel_time'].copy()

    # First, we need to take the travel time factors dataframe to determine the congestion periods
    ttf_df = inData.tt_factors.copy()

    # We determine total ride-hailing vehicle driving time per time period)
    result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored
    fleet_movements = pd.read_csv(os.path.join(result_dir, '2-2_op-stats.csv'))
    fleet_movements = fleet_movements.loc[fleet_movements['status'] != 'boarding'].copy()  # remove boarding times as it does not count as driving time
    # fleet_movements['duration'] = fleet_movements['end_time'] - fleet_movements['start_time']
    ttf_df['rs_travel_time'] = driving_time_per_time_period(ttf_df, fleet_movements, mode='rs')

    # We do the same for car travel time (but we need to consider the previous travel time factors)
    car_trips = inData.requests.loc[inData.passengers.mode_day == 'car'].copy()
    car_trips.rename(columns={'treq': 'start_time'}, inplace=True)
    car_trips['end_time'] = car_trips['start_time'] + car_trips['ttrav']  # end time of the trip
    ttf_df['car_travel_time'] = driving_time_per_time_period(ttf_df, car_trips, mode='car')

    # We need to determine the total travel time for each time period (for both modes), considering background traffic
    ttf_df['background_travel_time'] = ttf_df['background_traffic_share'] * params.congestion.get('uncongested_background_travel_time', 0) * ttf_df['travel_time_factor']  # background travel time (in seconds)
    ttf_df['total_travel_time'] = ttf_df['rs_travel_time'] + ttf_df['car_travel_time'] + ttf_df['background_travel_time'] 

    # Now we convert the total vehicle hours to vehicle density (veh/km per time bin)
    ttf_df['duration'] = -ttf_df.index.diff(periods=-1).fillna(0)  # duration of each time period in seconds
    # The last value is computed as the difference between the last time period and the first time period (to get the last time period)
    ttf_df.iloc[-1, ttf_df.columns.get_loc('duration')] = (params.t0 + params.simTime * 3600 - ttf_df.index[-1])
    total_road_dist = params.get('total_road_dist', pd.read_csv(os.path.join(fleetpy_dir, "data", "networks", network_name, "base", "edges.csv")).distance.sum() / 1000)  # km
    ttf_df['veh_density'] = ttf_df['total_travel_time'] / (ttf_df['duration'] * total_road_dist)  # veh/km
    # Then we can determine the relative travel speed (or travel time factor) using the formula of Sloot
    ttf_df['new_travel_time_factor'] = ttf_df.apply(lambda row: ttf_from_density(params, row['veh_density']), axis=1)  # ttf from density (in km/h)
    ttf_df = ttf_df[['travel_time_factor', 'new_travel_time_factor', 'background_traffic_share']].copy()

    return ttf_df


def ttf_from_density(params, density):
    '''determine travel time factor (delay) from density based on the fundamental diagram of traffic based on Sloot (2019)'''
    cong_params = params.congestion
    # Parameters
    v_free = cong_params.get('free_flow', 10)       # Free-flow speed (km/h)
    P_max = cong_params.get('max_flow', 1000)       # Maximum flow (veh/h)
    P_min = cong_params.get('min_flow', 100)        # Minimum flow (veh/h)

    k1 = cong_params.get('dens_free_flow_end', 10)              # End of free-flow regime
    k2 = cong_params.get('dens_p_max', 20)                      # Where P_max is reached
    k3 = cong_params.get('dens_flow_towards_p_min', 30)         # Where flow starts dropping to P_min
    k_j = cong_params.get('dens_jam', 40)                       # Jam density

    # Define flow function
    def flow(k):
        if k < k1:
            return v_free * k
        elif k1 <= k < k2:
            a = -2 * (P_max - v_free * k1) / ((k2 - k1) ** 2)
            return a * (k - k1) ** 2 + (P_max + v_free * k1) - a * (k2 - k1) ** 2
        elif k2 <= k < k3:
            return P_max
        elif k3 <= k < k_j:
            # Linear drop from P_max to P_min
            return P_max - (P_max - P_min) * (k - k3) / (k_j - k3)
        else:
            return P_min

    # Use flow function to determine speed
    P = flow(density)
    v = P / density
    ttf = v_free / v  # relative speed (compared to free flow speed)

    return ttf


def ttf_convergence_check(params, ttfs):
    '''check for convergence of travel time factors'''
    # Then we need to check for convergence of the travel time factors (for each time period)
    ttfs['rel_change_in_ttf'] = (ttfs['new_travel_time_factor'] - ttfs['travel_time_factor']) / ttfs['travel_time_factor']  # change in ttf (in %)
    ttfs['ttf_converged'] = ttfs['rel_change_in_ttf'].abs() < params.convergence.get('ttf', 0.01)  # check for convergence (in %)
    # If all travel time factors have converged, we can stop the iteration
    if ttfs['ttf_converged'].all():
        return True
    else:
        return False


def learn_credit_price(credit_price, perc_credit_price, rem_days, params):
    '''learn credit price based on previous expected credit price and latest price'''
    learning_weight = params.evol.travellers.get('kappa_credit_price', 0.2)

    if rem_days == params.tmc.duration-1:
        perc_credit_price = credit_price
    else:
        perc_credit_price = learning_weight * credit_price + (1 - learning_weight) * perc_credit_price

    return perc_credit_price


def charge_based_on_mode_and_location(params, row):
    '''find mode and check whether shortest path of trips traverses congestion zone'''
    if params.get('city_charge', False): # city-wide charge
        if row.mode_day == 'car':
            return params.city_charge.get('car', 5)
        elif row.mode_day.startswith('rs'):
            plf_id = int(row.mode_day.split("_")[-1])
            return params.city_charge.get('solo', 5) if params.platforms.service_types[plf_id] == 'solo' else params.city_charge.get('pool', 0)
        else:
            return 0
    else: # zone-specific charge (city centre)
        if not row.through_center:
            return 0
        elif row.mode_day == 'car':
            return params.zone_charge.get('car', 5)
        elif row.mode_day.startswith('rs'):
            plf_id = int(row.mode_day.split("_")[-1])
            return params.zone_charge.get('solo', 5) if params.platforms.service_types[plf_id] == 'solo' else params.zone_charge.get('pool', 0)
        else:
            return 0
    

def determine_congestion_charge(inData, params):
    '''determine travellers paid congestion charge'''

    df = pd.concat([inData.passengers.mode_day, inData.requests.through_center], axis=1)
    df['paid_cgp'] = df.apply(lambda row: charge_based_on_mode_and_location(params, row), axis=1)

    return df['paid_cgp']


def determine_starting_balance(inData, params, credits_per_day):
    '''set travellers' starting balance (possibly depending on their VoT)'''
    if isinstance(credits_per_day, (int, float, None)):
        tmc_balance = credits_per_day * params.tmc.get('duration', 1)
    else:
        vot_class = pd.qcut(inData.passengers['VoT'], q=len(credits_per_day), labels=False)
        tmc_balance = vot_class.map(lambda x: credits_per_day[x] * params.tmc.get('duration', 1))

    return tmc_balance


def set_trading_prefs(passengers, params):
    '''draws individual trading preferences for travellers from distribution, if specified'''
    if 'sd_beta_constant' in params.tmc.pref_trading:
        betas_constant = np.random.normal(params.tmc.pref_trading.get('beta_constant', 0), params.tmc.pref_trading.sd_beta_constant, len(passengers))
        passengers['beta_constant'] = betas_constant
    if 'sd_beta_balance' in params.tmc.pref_trading:
        if params.tmc.pref_trading.get('beta_balance', 0) > 0:
            betas_balance = np.random.lognormal(np.log(params.tmc.pref_trading.get('beta_balance', 0)) - params.tmc.pref_trading.sd_beta_balance ** 2 / 2, params.tmc.pref_trading.sd_beta_balance, len(passengers))
        elif params.tmc.pref_trading.get('beta_balance', 0) < 0:
            betas_balance = -np.random.lognormal(np.log(-params.tmc.pref_trading.get('beta_balance', 0)) - params.tmc.pref_trading.sd_beta_balance ** 2 / 2, params.tmc.pref_trading.sd_beta_balance, len(passengers))
        else:
            betas_balance = np.random.lognormal(np.log(1e-10) - params.tmc.pref_trading.sd_beta_balance ** 2 / 2, params.tmc.pref_trading.sd_beta_balance, len(passengers))
        passengers['beta_balance'] = betas_balance
    if 'sd_beta_price' in params.tmc.pref_trading:
        if params.tmc.pref_trading.get('beta_price', 0) > 0:
            betas_price = np.random.lognormal(np.log(params.tmc.pref_trading.get('beta_price', 0)) - params.tmc.pref_trading.sd_beta_price ** 2 / 2, params.tmc.pref_trading.sd_beta_price, len(passengers))
        elif params.tmc.pref_trading.get('beta_price', 0) < 0:
            betas_price = -np.random.lognormal(np.log(-params.tmc.pref_trading.get('beta_price', 0)) - params.tmc.pref_trading.sd_beta_price ** 2 / 2, params.tmc.pref_trading.sd_beta_price, len(passengers))
        else:
            betas_price = np.random.lognormal(np.log(1e-10) - params.tmc.pref_trading.sd_beta_price ** 2 / 2, params.tmc.pref_trading.sd_beta_price, len(passengers))
        passengers['beta_price'] = betas_price
    if 'sd_beta_days' in params.tmc.pref_trading:
        betas_time = np.random.normal(params.tmc.pref_trading.get('beta_time', 0), params.tmc.pref_trading.sd_beta_days, len(passengers))
        passengers['beta_time'] = betas_time
    if 'sd_beta_hist_buy' in params.tmc.pref_trading:
        betas_hist_buy = np.random.normal(params.tmc.pref_trading.get('beta_hist_buy', 0), params.tmc.pref_trading.sd_beta_hist_buy, len(passengers))
        passengers['beta_hist_buy'] = betas_hist_buy
    if 'sd_beta_hist_sell' in params.tmc.pref_trading:
        betas_hist_sell = np.random.normal(params.tmc.pref_trading.get('beta_hist_sell', 0), params.tmc.pref_trading.sd_beta_hist_sell, len(passengers))
        passengers['beta_hist_sell'] = betas_hist_sell
    
    return passengers


def choose_ridehailing_platform(inData, df, rs_util):
    '''decide which ridehailing platform to choose, and determine overall ridehailing utility following the choice for that platform'''
    
    df['U_rs_plf'] = rs_util
    df['U_rs_plf'] = df.apply(lambda row: row.U_rs_plf * unregist_to_nan(row.registered), axis=1) # only keep utility of platforms one is registered with
    df['prob_plf'] = df.apply(lambda row: np.array([(np.exp(row.U_rs_plf[plf]) / np.exp(row.U_rs_plf).sum()) if np.exp(row.U_rs_plf).sum() != 0 else 0 for plf in inData.platforms.index]), axis=1)
    df[['U_rs','chosen_plf_index']] = df.apply(lambda row: util_rs_plf(row), axis=1, result_type='expand')
    chosen_plf = df['chosen_plf_index'].astype(int).copy()
    rs_util = df['U_rs'].copy()

    return rs_util, chosen_plf


def ridehailing_platform_util_and_plf_prob(inData, df, rs_util):
    '''decide utility of ride-hailing alternative in overall mode choice considering utility of individual platforms, also return probabilities of choosing each platform'''
    
    df['U_rs_plf'] = rs_util
    df['U_rs_plf'] = df.apply(lambda row: row.U_rs_plf * unregist_to_nan(row.registered), axis=1) # only keep utility of platforms one is registered with
    df['prob_plf'] = df.apply(lambda row: np.array([(np.exp(row.U_rs_plf[plf]) / np.exp(row.U_rs_plf).sum()) if np.exp(row.U_rs_plf).sum() != 0 else 0 for plf in inData.platforms.index]), axis=1)
    df['U_rs'] = df.apply(lambda row: sum(row.U_rs_plf * row.prob_plf), axis=1) # overall utility of ridesourcing (considering probabilities and utils of each platform)
    prob_plf = df['prob_plf'].copy()
    rs_util = df['U_rs'].copy()

    return rs_util, prob_plf


def apply_license_plate_rationing(inData, car_util, kwargs):
    '''exclude cars from choice set if they are not allowed to drive on this day'''        
    day = kwargs.get('day', None)
    odd_day = ((day % 2) != 0)
    allowed_to_drive = inData.passengers.odd_license if odd_day else ~inData.passengers.odd_license
    car_util[~allowed_to_drive] = -math.inf

    return car_util


def order_based_on_need(credits_allocated_per_day, expected_credit_usage_per_day):
    '''order based solely on difference between expected credit usage and credit balance, neglecting any trading dynamics in the market
    Positive order quantity means that the traveller wants to buy credits, negative order quantity means that the traveller wants to sell credits'''
    order_quantity = expected_credit_usage_per_day - credits_allocated_per_day

    return order_quantity


def supply_demand_gap(inData, params, credit_price):
    '''Returns excess demand for credits (positive) and market order (and the mode that is actually chosen for this price).'''
    expected_credit_usage, probabilities, gtt_chosen_mode, attr_prob_mode = expected_credit_usage_and_actual_mode_choice(inData, params, credit_price=credit_price)
    market_order = order_based_on_need(inData.passengers.tmc_balance, expected_credit_usage)
    excess_demand = market_order.sum()
    return excess_demand, market_order, probabilities, gtt_chosen_mode, attr_prob_mode


def price_and_orders(inData, params, prev_price=None, possible_prices=None, step=0.001):
    """
    Finds the (euro) price (float) that minimizes |supply - demand| as well as the orders for that price.

    Parameters:
    - supply_demand_gap: callable(price_euro) -> difference between supply and demand
    - prev_price: float or None, warm-start center in euros
    - bounds: tuple (min_price_euro, max_price_euro)
    - step: float, price resolution (default 0.001 euros)

    Returns:
    - best_price: float, equilibrium price in euros
    - market_order: array, the market order for the best price
    """

    @lru_cache(maxsize=None)
    def compute_gap_and_order(price):
        return supply_demand_gap(inData, params, price)

    # Determine bounds and create list of candidate prices based on step size
    bounds = (possible_prices[0], possible_prices[-1]) if possible_prices is not None else (0, 5)  # default bounds
    if prev_price is not None:
        window = 0.50  # search within ±€0.50
        p_min = max(bounds[0], prev_price - window)
        p_max = min(bounds[1], prev_price + window)
    else:
        p_min, p_max = bounds

    prices = np.round(np.arange(p_min, p_max + step, step), 3)

    # Perform ternary search over discrete prices
    left, right = 0, len(prices) - 1
    best_price = None
    best_market_order = None
    best_chosen_gtt = None
    min_error = float('inf')

    while right - left > 3:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3

        gap_m1, _, _, _, _ = compute_gap_and_order(prices[m1])
        gap_m2, _, _, _, _ = compute_gap_and_order(prices[m2])

        if abs(gap_m1) < abs(gap_m2):
            right = m2
        else:
            left = m1

    # Find the best price and corresponding market order
    for i in range(left, right + 1):
        gap, market_order, probabilities, gtt, probabilistic_attr = compute_gap_and_order(prices[i])
        if abs(gap) < min_error:
            min_error = abs(gap)
            best_price = prices[i]
            best_market_order = market_order
            best_probabilities = probabilities
            best_chosen_gtt = gtt
            best_probabilistic_attr = probabilistic_attr

    return round(best_price, 3), best_market_order, best_probabilities, best_chosen_gtt, best_probabilistic_attr


def expected_modal_split_per_time_period(inData, params, probabilities):
    '''determine expected modal split per time period based on the probabilities of choosing each mode per traveller'''
    req_df = inData.requests.copy()
    req_df['rh_zone'] = inData.passengers['rh_zone'].copy()
    ttf_df = inData.tt_factors.copy()

    warm_time = params.t0 + params.get('warmup', 0)
    cooldown_time = params.t0 + params.simTime * 3600 - params.get('cooldown', 0)

    ttf_df = ttf_df.sort_index().copy()
    ttf_df['start_time'] = ttf_df.index
    ttf_df['end_time'] = ttf_df['start_time'].shift(-1)
    # Fill end time for the last row, e.g. using total simulation time
    ttf_df['end_time'] = ttf_df['end_time'].fillna(params.t0 + params.simTime * 3600)

    # --- Filter periods fully within (t_min, t_max) ---
    if warm_time is not None and cooldown_time is not None:
        valid_periods = ttf_df[(ttf_df['start_time'] >= warm_time) & (ttf_df['end_time'] <= cooldown_time)].copy()
    else:
        valid_periods = ttf_df

    # Create interval index
    periods = pd.IntervalIndex.from_arrays(valid_periods['start_time'], valid_periods['end_time'], closed='left')

    # Assign time period bin to each passenger
    req_df['time_period'] = pd.cut(req_df['treq'], bins=periods)

    # # Optional: map time_period back to period start time
    interval_to_start = dict(zip(periods, valid_periods['start_time']))
    req_df['period_start'] = req_df['time_period'].map(interval_to_start)

    modes = probabilities.columns.tolist()
    modes = [m for m in modes if m not in ['decis', 'chosen_rs_plf']]

    req_df[modes] = probabilities[modes].copy()
    df_mode_split = req_df.groupby(['period_start', 'rh_zone'], observed=False)[modes].sum()
    df_mode_split = df_mode_split / df_mode_split.sum(axis=1).values[:, None]  # normalize to 1

    return df_mode_split


def store_expected_modal_split(expected_modal_splits, mode_choice_iter, modal_split_conv_dict):
    '''store expected modal split in the convergence dictionary'''
    expected_modal_splits['mc_iter'] = mode_choice_iter  # Add iteration column  
    expected_modal_splits = expected_modal_splits.reset_index().set_index(['mc_iter', 'period_start', 'rh_zone']) 
    if modal_split_conv_dict.empty:
        modal_split_conv_dict = expected_modal_splits
    else:           
        modal_split_conv_dict = pd.concat([modal_split_conv_dict, expected_modal_splits])

    return modal_split_conv_dict


def determine_mode_choice_convergence(modal_split_conv_df, params):
    '''determine whether expected modal split has converged'''
    window = params.convergence.get('moving_average_window', 3)
    stable_iters = params.convergence.get('stable_iters', 3)
    epsilon = params.convergence.get('error_modal_split', 0.005)

    min_required = window + stable_iters - 1
    last_iter = modal_split_conv_df.index.get_level_values("mc_iter").max()

    if last_iter < min_required:
        print("Not enough iterations to evaluate convergence yet.")
        convergence_status = None  
    else:
        # Reshape: move mc_iter into columns to allow groupby over periods
        df_sorted = modal_split_conv_df.sort_index()
        df_mavg = (
            df_sorted
            .groupby(['period_start', 'rh_zone'], observed=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .droplevel([0,1])  # drop period_start and rh_zone from index to keep ('mc_iter', 'period_start', 'rh_zone')
        )

        # Compute differences in moving averages between consecutive iterations
        df_diff = df_mavg.groupby('period_start', observed=False).diff()

       # Only consider (period, rh_zone) that ever have data
        valid_combinations = (
            modal_split_conv_df
            .groupby(['period_start', 'rh_zone'], observed=False)
            .apply(lambda g: not g.isna().all().all())
        )
        valid_combinations = valid_combinations[valid_combinations].index

        # For each period, get the last `stable_iters` diffs and check max(abs) < epsilon
        converged = {}

        for (period, rh_zone), group in df_diff.groupby(['period_start','rh_zone'], observed=False):
            if (period, rh_zone) not in valid_combinations:
                continue  # Ignore this combination entirely
            group = group.dropna()
            if len(group) < stable_iters:
                converged[(period, rh_zone)] = False
                continue
            last_diffs = group.tail(stable_iters)
            max_change = last_diffs.abs().max().max()
            converged[(period, rh_zone)] = max_change < epsilon

        # Convert to DataFrame or Series
        convergence_status = pd.Series(converged, name='has_converged').all()

    return convergence_status


def determine_vkt(inData, params, result_dir):
    '''determine vehicle kilometres travelled (vkt) for each mode'''
    warm_time = params.t0 + params.get('warmup', 0)
    cooldown_time = params.t0 + params.simTime * 3600 - params.get('cooldown', 0)
    op_stats = pd.read_csv(os.path.join(result_dir,'2-2_op-stats.csv'))

    # Filter for time window
    op_stats = op_stats[(op_stats['start_time'] >= warm_time) & (op_stats['start_time'] <= cooldown_time)]

    # Sum driven_distance per operator_id
    distance_per_operator = op_stats.groupby('operator_id')['driven_distance'].sum()

    # Assign to platforms
    plf_0_dist = distance_per_operator.get(0, 0.0) / 1000
    plf_1_dist = distance_per_operator.get(1, 0.0) / 1000
    repos_dist = distance_per_operator.get(2, 0.0) / 1000

    # Car distance
    car_dist = ((inData.passengers.mode_day == 'car') * inData.requests.dist).sum() / 1000

    vkt_dict = {
        'vkt_rs_0': plf_0_dist,
        'vkt_rs_1': plf_1_dist,
        'vkt_repos': repos_dist,
        'vkt_car': car_dist
    }

    return vkt_dict