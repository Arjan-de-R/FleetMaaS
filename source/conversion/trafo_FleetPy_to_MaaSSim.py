import os
import pandas as pd
import numpy as np

##### DFs to be generated



# sim.last_res - veh_exp
## veh - 1 to m
## nRIDES
## nREJECTED - what does this mean?
## DRIVING_TIME
## DRIVING_DIST
## REVENUE
## COST
## NET_INCOME
## OUT
## FORCED_OUT
## STARTS_DAY
## OPENS_APP
## RECEIVES_REQUEST
## ACCEPTS_REQUEST
## REJECTS_REQUES
## IS_ACCEPTED_BY_TRAVELLER
## IS_REJECTED_BY_TRAVELLER
## ARRIVES_AT_PICKUP
## MEETS_TRAVELLER_AT_PICKUP
## DEPARTS_FROM_PICKUP
## ARRIVES_AT_DROPOFF
## CONTINUES_SHIFT
## STARTS_REPOSITIONING
## REPOSITIONED
## DECIDES_NOT_TO_DRIVE
## ENDS_SHIFT
## NOT_ALLOWED_TO_DRIVE

# sim.passengers
## (index) - 0 to n
## pos
## event
##! platforms
## VoT
## U_bike
## U_car
## U_pt
## ASC_rs
## mode_choice: always day-to-day
## prob_rs
## informed
##! expected_wait
##! mode_day
###! TODO: we need to add expected_ivt
###! TODO: we need to add multi_homer

# sim.vehicles - start here, what do we need?
## (index) - from 0 to m
## event - always travellerEvent.STARTS_DAY
## shift_start: always 0
## shift_end: always determined by nhours
##! platform - currently [0], what is it used for in wd model?
##! expected_income
## res_wage
## informed
## registered
##! work_exp
## days_since_reg:
## rejected_reg: Always FALSE
###! TODO: we need to add multi_homer (iniitally all TRUE)
###! TODO: indicators per platform
#### SJ: Inf: 0, 1, None, Both
#### SJ: Reg: 0, 1 or None
#### SJ: ptcp: True or False
#### MJ: Inf: 0, 1, None, Both??
#### MJ: Reg: True or False
#### MJ: Ptcp: True or False



def load_res_fleetpy(*args, **kwargs):
    '''import within-day results from FleetPy'''
    user_stats = pd.read_csv('1_user-stats.csv', index_col=0)
    
#     rq_f = os.path.join(dtd_result_dir, "demand", "inData_requests.csv")
    
    

def update_d2d_travellers_fleetpy(*args, **kwargs):
    "updating travellers' experience and updating new expected waiting time based on FleetPy simulation result"
    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)
    stats = kwargs.get('stats', None)
    
    ret = pd.DataFrame()
    ret['pax'] = np.arange(0, len(sim.passengers))
    ret['orig'] = sim.inData.requests.origin.to_numpy()
    ret['dest'] = sim.inData.requests.destination.to_numpy()
    ret['t_req'] = sim.inData.requests.treq.to_numpy()
    ret['tt_min'] = sim.inData.requests.ttrav.to_numpy()
    ret['dist'] = sim.inData.requests.dist.to_numpy()
    ret['informed'] = sim.passengers.informed.to_numpy()
    ret['requests'] = (sim.passengers.mode_day == 'rs') #new - check if right (for solo case only initially, single platform?)
    ret['gets_offer'] = True #new - we need to decide not only if an offer is received, but also from which platform
    ret['accepts_offer'] = ret['gets_offer'] #new - need to check
    ret['xp_wait'] = stats['pickup_time'] - stats['rq_time'] #new - but is df length same?
    ret['xp_ivt'] = stats['dropoff_time'] - stats['pickup_time'] #new
    ret['xp_ops'] = 0 #new
    
    ret.loc[(ret.requests == False) | (ret.gets_offer == False) | (ret.accepts_offer == False), ['xp_wait', 'xp_ivt',
                                                                                                 'xp_ops']] = np.nan
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops

    if not params.shareability.get('offered', False): # does this still work?
        ret['init_perc_wait'] = sim.passengers.expected_wait.to_numpy()
        ret['corr_xp_wait'] = ret.xp_wait.copy()
        ret.loc[(ret.requests & (~ret.gets_offer)),['corr_xp_wait']] = params.evol.travellers.reject_penalty
        new_perc_wait = learning_travs(params = params, prev_perc = ret.init_perc_wait, exp = ret.corr_xp_wait)

        ret['new_perc_wait'] = new_perc_wait.to_numpy()
        ret.loc[ret.informed & (~ret.requests), 'new_perc_wait'] = ret.loc[ret.informed & (~ret.requests), 'init_perc_wait']
        ret['chosen_mode'] = sim.passengers.mode_day.to_numpy()
    
    else:
        ret['init_perc_wait'] = sim.passengers.expected_wait.to_numpy()
        ret['init_perc_wait_pool'] = sim.passengers.expected_wait_pool.to_numpy()
        ret['corr_xp_wait'] = ret.xp_wait.copy()
        ret.loc[(ret.requests & (~ret.gets_offer)),['corr_xp_wait']] = params.evol.travellers.reject_penalty
        
        new_perc_wait = learning_travs(params = params, prev_perc = ret.init_perc_wait, exp = ret.corr_xp_wait)
        new_perc_wait_pool = learning_travs(params = params, prev_perc = ret.init_perc_wait_pool, exp = ret.corr_xp_wait)
        ret['new_perc_wait'] = new_perc_wait.to_numpy()
        ret['new_perc_wait_pool'] = new_perc_wait_pool.to_numpy()
        
        ret['chosen_mode'] = sim.passengers.mode_day.to_numpy()
        ret.loc[ret.informed & (ret.chosen_mode != 'rs'), 'new_perc_wait'] = ret.loc[ret.informed & (ret.chosen_mode != 'rs'), 'init_perc_wait']
        ret.loc[ret.informed & (ret.chosen_mode != 'pool'), 'new_perc_wait_pool'] = ret.loc[ret.informed & (ret.chosen_mode != 'rs'), 'init_perc_wait_pool']

#         ret['act_shared'] = ret.apply(lambda x: True if (len(sim.inData.requests.loc[x.name].sim_schedule.req_id.dropna().unique()) > 1) and x.gets_offer else False, axis=1)  # which travellers actually shared a part of their ride
        ret['act_shared'] = True #new - needs to be adjusted
        ret['xp_discount'] = np.nan
        ret.loc[ret.chosen_mode == 'pool','xp_discount'] = params.shareability.min_discount # discount for opting for pooled service (without actually sharing)
        ret.loc[ret.act_shared,'xp_discount'] = params.shareability.min_discount + params.shareability.add_discount  # discount for those that actually shared
        ret['xp_detour'] = ret['xp_ivt'] - sim.inData.requests.ttrav.apply(lambda x: x.total_seconds())
        
        ret['init_perc_disc'] = sim.passengers.expected_pool_disc.to_numpy()
        new_perc_disc = learning_travs(params = params, prev_perc = ret.init_perc_disc, exp = ret.xp_discount)
        ret['new_perc_disc'] = new_perc_disc.to_numpy()
        ret.loc[ret.informed & (ret.chosen_mode != 'pool'), 'new_perc_disc'] = ret.loc[ret.informed & (ret.chosen_mode != 'pool'), 'init_perc_disc']
        
        ret['init_perc_detour'] = sim.passengers.expected_pool_detour.to_numpy()
        new_perc_detour = learning_travs(params = params, prev_perc = ret.init_perc_detour, exp = ret.xp_detour)
        ret['new_perc_detour'] = new_perc_detour.to_numpy()
        ret.loc[ret.informed & (ret.chosen_mode != 'pool'), 'new_perc_detour'] = ret.loc[ret.informed & (ret.chosen_mode != 'pool'), 'init_perc_detour']

    ret = ret.set_index('pax')

    return ret


def update_d2d_drivers_fleetpy(*args, **kwargs):
    "updating drivers' day experience - from FleetPy results - and determination of new perceived income"



    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)
    veh_state = kwargs.get('veh_state', None)
    driver_kpis = kwargs.get('standard_mod_0', None)

    ret = pd.DataFrame()
    ret['veh'] = np.arange(1, params.nV + 1)
    
    ret['pos'] = veh_state.final_node_index #new - check indices (e.g. sorting)
    ret['informed'] = sim.vehicles.informed.to_numpy()
    ret['registered'] = sim.vehicles.registered.to_numpy()
    
    ret['out'] = sim.vehicles.ptcp_day #new
    ret['init_perc_inc'] = sim.vehicles.expected_income.to_numpy()
    ret['exp_inc'] = driver_kpis.revenue #new - is revenue the (net income) of drivers?
    ret['forced_out'] = False
    ret['rejected_reg'] = sim.vehicles.rejected_reg.to_numpy()
    ret.loc[ret.out, 'exp_inc'] = np.nan
    new_perc_inc = learning_drivers(params=params, prev_perc=ret.init_perc_inc, exp=ret.exp_inc.fillna(0), out=ret.out)

    ret['new_perc_inc'] = new_perc_inc.to_numpy()
    cols = list(ret.columns)
    ret = ret[cols]
    ret = ret.set_index('veh')

    return ret



def transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData):
    '''This function transforms the output files of the within-day model (FleetPy) for the day-to-day (MaaSSim) model'''

    result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored

    # 1) Load traveller KPIs
    req_kpis = pd.read_csv(os.path.join(result_dir,'1_user-stats.csv'),index_col = 'request_id')
    pax_exp = pd.DataFrame(index = req_kpis.index)
    pax_exp.index.name = 'pax'
    pax_exp['TRAVEL'] = req_kpis.dropoff_time - req_kpis.pickup_time # total travel time
    pax_exp['WAIT'] = req_kpis.pickup_time - req_kpis.rq_time
    pax_exp['OPERATIONS'] = 60 # 30 sec both for accessing and egressing the vehicle
    pax_exp['detour'] = req_kpis.dropoff_time - req_kpis.pickup_time - req_kpis.direct_route_travel_time - pax_exp.OPERATIONS # detour time
    pax_exp['fare'] = req_kpis.fare / 100
    pax_exp['platform'] = req_kpis.operator_id
    pax_exp['LOSES_PATIENCE'] = req_kpis.apply(lambda x: 999 if x.operator_id == '' else 0, axis=1) # used later in determining which travellers are rejected by platform
    pax_exp['NO_REQUEST'] = False # True if another mode is chosen prior to the day
    pax_exp['OTHER_MODE'] = False # True if request is made but received offers are rejected
    
    # Now we add travellers that did not make a request
    all_travs = inData.passengers.index.values
    req_travs = pax_exp.index.values
    noreq_travs = list(set(all_travs) - set(req_travs))
    no_req_df = pd.DataFrame(np.nan, index=noreq_travs, columns=['TRAVEL','WAIT','OPERATIONS','detour','fare','platform','LOSES_PATIENCE'])
    no_req_df['NO_REQUEST'], no_req_df['OTHER_MODE'] = [True, False]
    pax_exp = pd.concat([pax_exp,no_req_df]).sort_index()
    sim.last_res.pax_exp = pax_exp # store in MaaSSim simulator object

    # 2) Load driver KPIs
    # driver_kpis = pd.read_csv(os.path.join(result_dir,'1_user-stats.csv'),index_col = 'request_id')


    return sim

    # indicators for those that did not request