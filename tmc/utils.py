import numpy as np
import pandas as pd
import os
import networkx as nx
import math
from FleetPy.src.misc.globals import *
from FleetPy.src.routing.NetworkTTMatrix import NetworkTTMatrix
from MaaSSim.src_MaaSSim.d2d_demand import mode_probs

def trip_credit_cost(inData, params):
    '''Determine credit cost for each mode for travellers' trip itineraries'''

    # Modes other than ridesourcing
    inData.requests['bike_credit'] = params.tmc.credit_mode.bike.base + inData.requests.dist_bike / 1000 * params.tmc.credit_mode.bike.dist
    inData.requests['car_credit'] = params.tmc.credit_mode.car.base + inData.requests.dist / 1000 * (params.tmc.credit_mode.car.dist + params.tmc.credit_mode.car.get('dist_add_center', 0) * inData.requests.through_center)
    inData.requests['pt_credit'] = params.tmc.credit_mode.pt.base + inData.requests.PTdistance / 1000 * params.tmc.credit_mode.pt.dist
    
    def rs_plf_credit(req_dist, service_type, through_center):
        '''determine required credits for solo and pooling trip for a given trip request'''
        if service_type == 'solo':
            trip_credit = params.tmc.credit_mode.solo.base + req_dist / 1000 * (params.tmc.credit_mode.solo.dist + params.tmc.credit_mode.solo.get('dist_add_center', 0) * through_center)
        else:
            trip_credit = params.tmc.credit_mode.pool.base + req_dist / 1000 * (params.tmc.credit_mode.pool.dist + params.tmc.credit_mode.pool.get('dist_add_center', 0) * through_center)
        
        return trip_credit

    # Ridesourcing, depending on solo or pooling
    inData.requests['rs_credit'] = inData.requests.apply(lambda row: np.array([rs_plf_credit(row.dist, params.platforms.service_types[plat_id], row.through_center) for plat_id in range(0,len(params.platforms.service_types))]), axis=1)

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


def establish_buy_quantities(params, value_dict):
    '''Create database with buy/sell quantities depending on remaining number of days, credit balance and price'''

    # Fill the database with buy/sell values
    buy_quant_dict = {}
    for rem_days in value_dict['days']:
        balance_dict = {}
        for balance in value_dict['balance']:
            # Create numpy 2d-array with prices as rows and buy_values as columns
            util_buy_price_quant = util_buy(params, balance, value_dict['price'], value_dict['quantity'], rem_days)
            max_indices = np.nanargmax(util_buy_price_quant, axis=1)
            balance_dict[balance] = value_dict['quantity'][max_indices]
        buy_quant_dict[rem_days] = balance_dict.copy()

    return buy_quant_dict


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


def buy_table_dimensions(params):
    '''Determine which values are included in the table with quantities depending on price and credit balance'''
    min_price_step = params.tmc.price.get('step', 0.01)
    min_balance_step = params.tmc.balance.get('step', 0.1)
    allocated_credits_per_day = params.tmc.get('allocated_credits_per_day', 10)
    avg_allocated_credits_per_day = allocated_credits_per_day if isinstance(allocated_credits_per_day, (int, float)) else np.array([allocated_credits_per_day]).mean()

    max_balance = params.tmc.get('max_balance', params.tmc.max_balance_rel_to_init_allocation * avg_allocated_credits_per_day * params.tmc.get('duration', 25))
    max_buy_quant = params.tmc.get('max_buy_day', max_balance)

    # Determine dimensions of database: balance quantities and credit price levels
    balance_values = np.arange(0, max_balance + min_balance_step, min_balance_step)
    price_values = np.arange(params.tmc.price.get('min',0), params.tmc.price.get('max', 10) + min_price_step, min_price_step)
    buy_values = np.arange(-max_buy_quant,max_buy_quant+1)
    remaining_day_values = np.arange(1,params.tmc.duration+1)
    value_dict = {'balance': balance_values, 'price': price_values, 'quantity': buy_values, 'days': remaining_day_values}

    return value_dict


def trading(inData, value_dict):
    '''Determine market price and how many credits are bought and sold by each individual (and rejected orders)'''

    today_net_buy_quant_dict = {}
    for pax in inData.passengers.index:
        today_net_buy_quant_dict[pax] = inData.passengers.order_per_price.loc[pax].copy() # fill dict

    # Aggregate individual buy/sell quantities to aggregated
    agg_net_buy_quant = np.sum([buy_array for buy_array in today_net_buy_quant_dict.values()], axis=0)
    # Determine credit price by finding minimum net (absolute) buy/sell offer
    credit_price_index = np.argmin(abs(agg_net_buy_quant))
    credit_price = value_dict['price'][credit_price_index]
    net_supply = agg_net_buy_quant[credit_price_index]

    # Create pandas series with desired buy quantity per pax
    order_list = []
    for pax_id, order in today_net_buy_quant_dict.items():
        # Extract the buy/sell order corresponding to the credit price
        value = order[credit_price_index]
        # Append the value along with the pax_id to the list
        order_list.append((pax_id, value))
        order_series = pd.Series(dict(order_list))

    # Now determine which orders are satisfied
    buy_orders = order_series[order_series > 0].sort_values(ascending=False)
    sell_orders = order_series[order_series < 0].sort_values(ascending=True)
    satisfy_net_buy_quant = order_series.copy()
    if net_supply < 0: # supply exceeds demand for the current credit price
        satisfy_net_buy_quant[satisfy_net_buy_quant < 0] = 0
        remaining_buy_quant = buy_orders.sum()
        # Satisfy sell orders from large to small (as long as credits are available)
        for index, value in sell_orders.iteritems():
            assert remaining_buy_quant >= 0
            if remaining_buy_quant == 0:
                break
            if abs(value) > remaining_buy_quant:
                satisfy_net_buy_quant[index] = -remaining_buy_quant
                remaining_buy_quant = 0
            else:
                satisfy_net_buy_quant[index] = value
                remaining_buy_quant += value
    elif net_supply > 0: # demand exceeds supply
        satisfy_net_buy_quant[satisfy_net_buy_quant > 0] = 0
        remaining_sell_quant = abs(sell_orders).sum()
        # Satisfy buy orders from large to small (as long as credits are available)
        for index, value in buy_orders.iteritems():
            assert remaining_sell_quant >= 0
            if remaining_sell_quant == 0:
                break
            if value > remaining_sell_quant:
                satisfy_net_buy_quant[index] = remaining_sell_quant
                remaining_sell_quant = 0
            else:
                satisfy_net_buy_quant[index] = value
                remaining_sell_quant -= value
    denied_orders = order_series - satisfy_net_buy_quant

    return credit_price, satisfy_net_buy_quant, denied_orders


def update_balances(inData, satisfied_orders, denied_orders, credit_price):
    '''Update credit and monetary balance based on satisfied trading orders (buy and sell) and price'''
    inData.passengers['net_purchase'] = satisfied_orders.copy()
    inData.passengers['denied_order'] = denied_orders.copy()
    inData.passengers.tmc_balance = inData.passengers.tmc_balance + inData.passengers.net_purchase
    inData.passengers.tot_credit_bought = inData.passengers.apply(lambda row: row.tot_credit_bought + max(0, row.net_purchase), axis=1) 
    inData.passengers.tot_credit_sold = inData.passengers.apply(lambda row: row.tot_credit_sold + max(0, -row.net_purchase), axis=1) 
    inData.passengers.money_balance = inData.passengers.money_balance - inData.passengers.net_purchase * credit_price

    return inData.passengers


def save_tmc_market_indicators(inData, result_path, day, credit_price, satisfied_orders, denied_orders):
    '''Append day's trading market indicators to csv'''
    market_indic = pd.DataFrame([{'price': credit_price, 'transaction_volume': satisfied_orders[satisfied_orders > 0].sum(), 'oversupply': denied_orders.sum(), 'mean_balance': inData.passengers.tmc_balance.mean()}])
    if day == 0: # include the headers on the first day
        if os.path.exists(os.path.join(result_path,'6_tmc-indicators.csv')):
            os.remove(os.path.join(result_path,'6_tmc-indicators.csv'))
        market_indic.to_csv(os.path.join(result_path,'6_tmc-indicators.csv'), mode='a', index=False, header=True)
    else:
        market_indic.to_csv(os.path.join(result_path,'6_tmc-indicators.csv'), mode='a', index=False, header=False)

    return 0


def order_per_price(params, value_dict, rem_days, credit_balance):
    '''Determine a traveller's buy/sell order for each possible credit price depending on their balance and time left to spend credits'''

    if rem_days > 0:
        max_balance = np.max(value_dict['balance'])
        util_buy_price_quant = util_buy(params, credit_balance, value_dict['price'], value_dict['quantity'], rem_days, max_balance=max_balance) # observed utility
        max_indices = np.nanargmax(util_buy_price_quant, axis=1)
        quantity = value_dict['quantity'][max_indices]
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


def util_alt_modes_tmc(params):
    "determine utility of alternative modes for group of travellers"
    prefs = params.evol.travellers.mode_pref

    # Draw Value of Time and corresponding beta's for travellers
    vot = np.random.lognormal(mean=prefs.ivt_mean_lognorm, sigma=prefs.ivt_sigma_lognorm, size=params.nP) * (-1) / prefs.beta_cost * 60  # VoT in euro/h

    # Draw mode preferences (ASCs) for travellers
    ASC_car = np.random.normal(prefs.ASC_car, prefs.ASC_car_sd, params.nP)
    ASC_bike = np.random.normal(0, prefs.ASC_bike_sd, params.nP)

    # PT alternative - if included in simulation
    if params.paths.get('PT_trips',False):
        ASC_pt = np.random.normal(prefs.ASC_pt, prefs.ASC_pt_sd, params.nP)
    else:
        ASC_pt = -np.inf

    ASCs = pd.DataFrame({'bike': ASC_bike, 'car': ASC_car, 'pt': ASC_pt})
    
    return ASCs, vot


def prefs_travs_tmc(inData, params):
    "draw mode preferences for the group of travellers"
    prefs = params.evol.travellers.mode_pref
    passengers = inData.passengers

    ASCs, vot = util_alt_modes_tmc(params)
    passengers['ASC_bike'] = ASCs.bike
    passengers['ASC_car'] = ASCs.car
    passengers['ASC_pt'] = ASCs.pt

    passengers['VoT'] = vot
    
    passengers['ASC_rs'] = np.random.normal(prefs.ASC_rs, prefs.ASC_rs_sd,len(inData.passengers))
    passengers['ASC_pool'] = passengers.ASC_rs + np.random.uniform(prefs.min_wts_constant, 0, len(inData.passengers))

    return passengers


def mode_preday_plf_choice_tmc(inData, params, **kwargs):
    "determine the mode at the start of a day for a pool of travellers (if they are single-homing yet possibly registered with more than 1 platform and still have to choose)"
    requests = inData.requests
    passengers = inData.passengers
    prefs = params.evol.travellers.mode_pref
    props = params.alt_modes
    credit_price = kwargs.get('credit_price')
    perc_credit_price = kwargs.get('perc_credit_price', None)
    perc_congest_factor = kwargs.get('perc_congest_factor', 1)
    mode_attr = {}
    utils = {}

    ## Establish attributes in mode choice for each mode
    # Bike
    mode_attr['bike'] = {}
    mode_attr['bike']['gtt'] = requests.ttrav_bike.dt.total_seconds() * prefs.bike_multip
    mode_attr['bike']['cost'] = 0
    mode_attr['bike']['credits'] = requests.bike_credit if params.dem_mgmt == 'tmc' else 0
    mode_attr['bike']['constant'] = passengers.ASC_bike
    # Private car
    mode_attr['car'] = {}
    car_ivt = requests.ttrav.dt.total_seconds() * perc_congest_factor  # assumed same as RS (solo)
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
    # Ride-hailing
    df = passengers.copy()
    mode_attr['rs'] = {}
    if params.dem_mgmt == 'cgp':
        congestion_charge = []
        for plf in range(len(params.platforms.service_types)):
            if not params.get('city_charge', False):
                plf_charge = params.zone_charge.get('solo', 5) if params.platforms.service_types[plf] == 'solo' else params.zone_charge.get('pool', 0)
            else:
                plf_charge = params.city_charge.get('solo', 5) if params.platforms.service_types[plf] == 'solo' else params.city_charge.get('pool', 0)
            congestion_charge.append(plf_charge)
        mode_attr['rs'] = rs_attr_tmc(inData, params, df.expected_wait * perc_congest_factor, df.expected_ivt * perc_congest_factor, df.expected_km_fare, inData.requests.dist, congestion_charge=congestion_charge)
    else:
        mode_attr['rs'] = rs_attr_tmc(inData, params, df.expected_wait * perc_congest_factor, df.expected_ivt * perc_congest_factor, df.expected_km_fare, inData.requests.dist)
    mode_attr['rs']['credits'] = requests.rs_credit.copy() if params.dem_mgmt == 'tmc' else 0

    # Determine utility of each mode
    for mode in ['bike', 'car', 'pt', 'rs']:
        df['tmc_balance'] = df.tmc_balance if params.dem_mgmt == 'tmc' else 0
        if params.evol.travellers.mode_pref.get('credit_percept', "monetary") == "monetary":    # convert credit charge to monetary costs
            if params.dem_mgmt != "tmc":
                perc_credit_price = 0
            utils[mode] = util_credit_to_cost(params, mode_attr[mode], perc_credit_price, passengers.VoT)
        else: # credit costs perceived separately in utility
            utils[mode] = util_credit_time(params, mode_attr[mode], credit_price, df.tmc_balance, passengers.VoT)
        utils[mode] = apply_insufficient_balance(utils[mode], mode_attr[mode]['credits'], df.tmc_balance, mode)
        if mode == 'rs':
            df['U_rs_plf'] = utils[mode]
            df['U_rs_plf'] = df.apply(lambda row: row.U_rs_plf * unregist_to_nan(row.registered), axis=1) # only keep utility of platforms one is registered with
            df['prob_plf'] = df.apply(lambda row: np.array([(np.exp(row.U_rs_plf[plf]) / np.exp(row.U_rs_plf).sum()) for plf in inData.platforms.index]), axis=1)
            df[['U_rs','chosen_plf_index']] = df.apply(lambda row: util_rs_plf(row), axis=1, result_type='expand')
            df['chosen_plf_index'] = df['chosen_plf_index'].astype(int)
            utils[mode] = df['U_rs'].copy()
        if mode == 'car' and params.dem_mgmt == 'lpr': # license plate rationing - cars allowed to drive on odd / even days
            day = kwargs.get('day', None)
            odd_day = ((day % 2) != 0)
            allowed_to_drive = inData.passengers.odd_license if odd_day else ~inData.passengers.odd_license
            utils[mode][~allowed_to_drive] = -math.inf
    utils_df = pd.DataFrame.from_dict(utils)

    ## ACTUAL MODE CHOICE
    probabilities = mode_probs(utils_df)
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities['pref_rs_plf'] = df.chosen_plf_index.copy()

    df['U_bike'] = utils['bike']
    df['U_car'] = utils['car']
    df['U_pt'] = utils['pt']
    if params.dem_mgmt == 'tmc':
        # opt out if not enough credit to travel (for any mode)
        probabilities['insuff_credit'] = df.apply(lambda row: (row.U_bike == -math.inf) and (row.U_car == -math.inf) and (row.U_pt == -math.inf) and (row.U_rs == -math.inf), axis=1)
        probabilities['decis'] = probabilities.apply(lambda row: "not_enough_credit" if row.insuff_credit else row.decis, axis=1)

    probabilities['decis'] = probabilities.apply(lambda row: row.decis + '_' + str(row.pref_rs_plf) if row.decis == 'rs' else row.decis, axis=1)
    passengers['mode_day'] = probabilities.decis

    passengers['U_bike'] = utils['bike']
    passengers['U_car'] = utils['car']
    passengers['U_pt'] = utils['pt']
    
    requests['chosen_mode_perc_gtt'] = return_gtt_chosen_mode(passengers, mode_attr)

    return passengers, requests


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


def util_credit_to_cost(params, attr, perc_credit_price, VoT):
    """determine mode utility depending on generalised travel time, normal costs and historical credit costs, when credit charge is perceived as monetary cost"""
    prefs = params.evol.travellers.mode_pref
    beta_time = VoT * prefs.beta_cost / 3600  # util/s

    # Convert learned credit price to cost
    total_cost = attr['cost'] + perc_credit_price * attr['credits']

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


def return_gtt_chosen_mode(passengers, mode_attr):
    '''returns generalised travel time of chosen mode for all travellers'''

    df = passengers.copy()
    df['gtt_bike'] = mode_attr['bike']['gtt']
    df['gtt_car'] = mode_attr['car']['gtt']
    df['gtt_pt'] = mode_attr['pt']['gtt']
    df['gtt_rs'] = mode_attr['rs']['gtt']

    gtt_chosen_mode = df.apply(lambda row: seek_indiv_mode_gtt(row), axis=1)

    return gtt_chosen_mode


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
    '''determine congestion factor for the day based on the vehicle kilometres of car and ridesourcing rides, which are also returned'''
    result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored
    wd_eval = pd.read_csv(os.path.join(result_dir,'standard_eval.csv'))
    plf_0_dist = wd_eval[wd_eval['Unnamed: 0'] == 'total vkm']['MoD_0'].values[0] * 1000
    plf_0_speed = wd_eval[wd_eval['Unnamed: 0'] == 'avg driving velocity [km/h]']['MoD_0'].values[0]
    plf_0_tt = (plf_0_dist / 1000) / plf_0_speed
    plf_0_tt = 0 if np.isnan(plf_0_tt) else plf_0_tt
    plf_1_dist = wd_eval[wd_eval['Unnamed: 0'] == 'total vkm']['MoD_1'].values[0] * 1000
    plf_1_speed = wd_eval[wd_eval['Unnamed: 0'] == 'avg driving velocity [km/h]']['MoD_1'].values[0]
    plf_1_tt = (plf_1_dist / 1000) / plf_1_speed
    plf_1_tt = 0 if np.isnan(plf_1_tt) else plf_1_tt
    car_dist = ((inData.passengers.mode_day == 'car') * inData.requests.dist).sum()
    car_tt = ((inData.passengers.mode_day == 'car') * inData.requests.ttrav).sum().seconds / 3600
    total_vkt = (plf_0_dist + plf_1_dist + car_dist) / 1000
    total_tt = plf_0_tt + plf_1_tt + car_tt # hour
    avg_number_of_cars_on_road_inhabitant = total_tt / params.simTime
    avg_number_of_cars_on_road_inhabitant = params.get('total_trips', 100000) / params.nP * avg_number_of_cars_on_road_inhabitant  # how many travellers does each trav agent represent
    avg_number_of_cars_on_road_other = params.congestion.get('avg_other_cars_on_road', 10000)
    tot_avg_number_of_cars_on_road = avg_number_of_cars_on_road_inhabitant + avg_number_of_cars_on_road_other
    total_road_dist = params.get('total_road_dist', pd.read_csv(os.path.join(fleetpy_dir, "data", "networks", network_name, "base", "edges.csv")).distance.sum() / 1000)  # km
    avg_road_density = tot_avg_number_of_cars_on_road / total_road_dist  # veh/km
    ## Determine delay factor
    density_start_congest = params.congestion.get('start_congestion_density', 0)
    density_zero_speed = params.congestion.get('density_zero_speed', 100)
    if avg_road_density < density_start_congest:
        day_congest_factor = params.congestion.get('min_delay_factor', 1)
    else:
        speed_rel_to_max = max(1 - (avg_road_density - density_start_congest) / (density_zero_speed - density_start_congest), 0.0001)
        day_congest_factor = params.congestion.get('min_delay_factor', 1) / speed_rel_to_max

    return day_congest_factor, car_dist, plf_0_dist, plf_1_dist


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
        tmc_balance = credits_per_day * params.tmc.get('duration', 25)
    else:
        vot_class = pd.qcut(inData.passengers['VoT'], q=len(credits_per_day), labels=False)
        tmc_balance = vot_class.map(lambda x: credits_per_day[x] * params.tmc.get('duration', 25))

    return tmc_balance