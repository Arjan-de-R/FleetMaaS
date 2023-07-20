def init_solo_plf(params, plf_id):
    '''generate row for inData.platforms with (solo) platform attributes, including 0 max. rel detour to prevent pooling'''
    plf_properties = [params.platforms.fare,'Platform {}'.format(plf_id),30,params.platforms.base_fare,params.platforms.comm_rate,
                      params.platforms.min_fare, params.platforms.match_obj, params.platforms.max_wait_time, 0.01] # 0.01 to prevent pooling
    return plf_properties

def init_pooling_plf(params, plf_id):
    '''generate row for inData.platforms with (pooling) platform attributes'''
    plf_properties = [params.platforms.fare*(1-params.platforms.pool_discount),'Platform {}'.format(plf_id),30,params.platforms.base_fare,params.platforms.comm_rate,
                      params.platforms.min_fare, params.platforms.match_obj, params.platforms.max_wait_time, params.platforms.max_rel_detour]
    return plf_properties