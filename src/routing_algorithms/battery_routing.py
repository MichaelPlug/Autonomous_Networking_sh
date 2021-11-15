import numpy as np
import src.utilities.utilities as util

from src.routing_algorithms.BASE_routing import BASE_routing

class BatteryRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> battery approach, take the drone with less battery """
        min_res_en= self.residual_energy
        best_drone = None

        for hpk, drone_istance in opt_neighbors:
            res_en = drone_instance.residual_energy  # without estimation, a simple geographic approach
            if res_en < min_res_en:
                min_res_en = res_en
                best_drone = drone_istance

        return best_drone
