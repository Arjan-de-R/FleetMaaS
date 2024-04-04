import numpy as np
import pandas as pd
import os

def trip_credit_cost(inData, params):
    '''Determine credit cost for each mode for travellers' trip itineraries'''

    # Modes other than ridesourcing
    inData.requests['bike_credit'] = params.tmc.credit_mode.bike.base + inData.requests.dist_bike / 1000 * params.tmc.credit_mode.bike.dist
    inData.requests['car_credit'] = params.tmc.credit_mode.car.base + inData.requests.dist / 1000 * params.tmc.credit_mode.car.dist
    inData.requests['pt_credit'] = params.tmc.credit_mode.pt.base + inData.requests.PTdistance / 1000 * params.tmc.credit_mode.pt.dist
    
    def rs_plf_credit(req_dist, service_type):
        '''determine required credits for solo and pooling trip for a given trip request'''
        if service_type == 'solo':
            trip_credit = params.tmc.credit_mode.solo.base + req_dist / 1000 * params.tmc.credit_mode.solo.dist
        else:
            trip_credit = params.tmc.credit_mode.pool.base + req_dist / 1000 * params.tmc.credit_mode.pool.dist
        
        return trip_credit

    # Ridesourcing, depending on solo or pooling
    inData.requests['rs_credit'] = inData.requests.apply(lambda row: np.array([rs_plf_credit(row.dist, params.platforms.service_types[plat_id]) for plat_id in range(0,len(params.platforms.service_types))]), axis=1)

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


def util_buy(params, balance, price, buy_quant, rem_days, probabilistic=True):
    '''Determine utility associated with buying and selling, trading off financial gains/costs and utility of having credits'''

    util_cost = np.array([price]).T * buy_quant * params.tmc.get('beta_monetary', -1)
    util_credit = np.sqrt((balance + buy_quant) / rem_days) - np.sqrt(balance / rem_days)
    net_util_buy = util_credit + util_cost
    if probabilistic:
        std_dev_error_term = np.sqrt((np.pi**2) / 6)
        error_terms = np.random.normal(loc=0, scale=std_dev_error_term, size=net_util_buy.shape[1])
        net_util_buy = net_util_buy + error_terms

    return net_util_buy


def buy_table_dimensions(params):
    '''Determine which values are included in the table with quantities depending on price and credit balance'''
    min_price_step = params.tmc.price.get('step', 0.01)
    min_balance_step = params.tmc.balance.get('step', 0.1)
    max_buy_quant = params.tmc.get('max_quant', 100)

    # Determine dimensions of database: balance quantities and credit price levels
    balance_values = np.arange(0, params.tmc.max_balance + min_balance_step, min_balance_step)
    price_values = np.arange(params.tmc.price.get('min',0), params.tmc.price.get('max',10) + min_price_step, min_price_step)
    buy_values = np.arange(-max_buy_quant,max_buy_quant+1)
    remaining_day_values = np.arange(1,params.nD+1)
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
        util_buy_price_quant = util_buy(params, credit_balance, value_dict['price'], value_dict['quantity'], rem_days) # observed utility
        max_indices = np.nanargmax(util_buy_price_quant, axis=1)
        quantity = value_dict['quantity'][max_indices]
    else:
        quantity = np.zeros(len(value_dict['price']))

    return quantity
