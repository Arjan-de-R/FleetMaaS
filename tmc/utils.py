import numpy as np

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


def establish_buy_quantities(params):
    '''Create database with buy/sell quantities depending on credit balance and price'''
    min_price_step = params.tmc.price.get('step', 0.01)
    max_buy_quant = params.tmc.get('max_quant', 100)

    # Determine dimensions of database: balance quantities and credit price levels
    balance_values = np.arange(0, params.tmc.max_balance+1)
    price_values = np.arange(params.tmc.price.get('min',0), params.tmc.price.get('max',10) + min_price_step, min_price_step)
    buy_values = np.arange(-max_buy_quant,max_buy_quant+1)

    # Fill the database with buy/sell values
    buy_quant_dict = {}
    for balance in balance_values:
        # Create numpy 2d-array with prices as rows and buy_values as columns
        util_buy_price_quant = util_buy(params, balance, price_values, buy_values)
        max_indices = np.nanargmax(util_buy_price_quant, axis=1)
        buy_quant_dict[balance] = buy_values[max_indices]

    return buy_quant_dict


def util_buy(params, balance, price, buy_quant):
    '''Determine utility associated with buying and selling, trading off financial gains/costs and utility of having credits'''

    util_cost = np.array([price]).T * buy_quant * params.tmc.get('beta_monetary', -1)
    util_credit = np.sqrt(balance + buy_quant) - np.sqrt(balance)
    net_util_buy = util_credit + util_cost

    return net_util_buy
