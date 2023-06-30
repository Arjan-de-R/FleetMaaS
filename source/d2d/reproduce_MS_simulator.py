from dotmap import DotMap
from MaaSSim.MaaSSim.decisions import f_dummy_repos, f_match, dummy_False

class repl_sim_object:
    """
    here we reproduce the MaaSSim simulator class for incorporating FleetPy results into the day-to-day process
    """
    
    def __init__(self, _inData, **kwargs):
        # input
        self.inData = DotMap()
        self.last_res = DotMap()
        self.platforms = dict()
        self.vehicles = dict()
        self.passengers = dict()
        self.requests = dict()
        self.functions = DotMap()
        self.functions.f_stop_crit = dummy_False
