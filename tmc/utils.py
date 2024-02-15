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