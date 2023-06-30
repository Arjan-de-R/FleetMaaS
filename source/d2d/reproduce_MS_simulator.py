from dotmap import DotMap

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