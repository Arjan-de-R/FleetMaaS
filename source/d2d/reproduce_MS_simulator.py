from dotmap import DotMap

class repl_sim_object:
    """
    here we reproduce the MaaSSim simulator class for incorporating FleetPy results into the day-to-day process
    """
    # STATICS and kwargs
    # list of functionalities
    # that may be filled with functions to represent desired behaviour
    # FNAMES = ['f_match',
    #           'f_trav_out',
    #           'f_driver_learn',
    #           'f_driver_out',
    #           'f_trav_mode',
    #           'f_driver_decline',
    #           'f_platform_choice',
    #           'f_driver_repos',
    #           'f_timeout',
    #           'f_stop_crit',
    #           'kpi_pax',
    #           'kpi_veh']


    def __init__(self, _inData, **kwargs):
        # input
        # self.inData = dict()
        self.last_res = DotMap()
        # self.platforms = dict()
        # self.vehicles = dict()
        # self.passengers = dict()
        # self.requests = dict()

        # def demand_results(self, pax_exp):
            # self.last_res.pax_exp = pax_exp

        
        # _inData.copy()  # copy of data structure
        # self.vehicles = self.inData.vehicles  # input
        # self.platforms = self.inData.platforms  # input
        # self.passengers = self.inData.passengers
        # self.requests = self.inData.requests
        # self.last_res.pax_exp = 


        # self.defaults = DEFAULTS.copy()  # default configuration of decision functions

        # self.myinit(**kwargs)  # part that is called every run
        # output
        # self.run_ids = list()  # ids of consecutively executed runs
        # self.runs = dict()  # simulation outputs (raw)
        # self.res = dict()  # simulation results (processed)
        # self.logger = self.init_log(**kwargs)
        # self.logger.warning("""Setting up {}h simulation at {} for {} vehicles and {} passengers in {}"""
                            # .format(self.params.simTime,
                                    # self.t0, self.params.nV, self.params.nP,
                                    # self.params.city))

    ##########
    #  PREP  #
    ##########

    # def myinit(self, **kwargs):
        # part of init that is repeated every run
        # self.update_decisions_and_params(**kwargs)

        # self.make_skims()
        # self.set_variabilities()
        # self.env = simpy.Environment()  # simulation environment init
        # self.t0 = self.inData.requests.treq.min()  # start at the first request time
        # self.t1 = 60 * 60 * (self.params.simTime + 2)

        # self.trips = list()  # report of trips
        # self.rides = list()  # report of rides
          # init requests
        # self.reqQ = list()  # queue of requests (traveller ids)
        # self.vehQ = list()  # queue of idle vehicles (driver ids)
        # self.pax = dict()  # list of passengers
        # self.vehs = dict()  # list of vehicles
        # self.plats = dict()  # list of platforms
        # self.ptcp = list() # list of participating drivers
        # self.sim_start = None