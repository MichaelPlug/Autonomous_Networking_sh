
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        # Packets that we delivered and still need a feedback
        print(self.drone.identifier, "----------", self.taken_actions)

        # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
        # Feedback from a delivered or expired packet
        print(self.drone.identifier, "----------", drone, id_event, delay, outcome)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action = self.taken_actions[id_event]
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        """ arg min score  -> geographical approach, take the drone closest to the depot """
        
        #we take our distance from the depot
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
        
        d = self.drone.next_target()
        
        
        time_taken_best = best_drone_distance_from_depot / self.drone.speed
        
        
       
        
        #initially drone closest is us (we take to the depot the
        #packet without any help)
        best_drone = None

        #we take all hello packets and all istances of drones
        for hello_packet, drone_istance in opt_neighbors:
            
            """
            
                we take the current position, the direction and the speed
                to estimate next position
            
            """
            
            """
            print((drone_istance))
            
            print(hello_packet.next_target)
            
            print(hello_packet.cur_pos, hello_packet.cur_pos[0])
            
            
            x1 = hello_packet.cur_pos[0]            
            
            y1 = hello_packet.cur_pos[1]
            
            x2 = hello_packet.next_target[0]
            
            y2 = hello_packet.next_target[1]
            
            m = (y2 - y1) / (x2 - x1)
            
            x_new = x1 + (1 / (np.sqrt(1 + (m*m) ) ) )*0.5
            
            y_new = y1 + (m / (np.sqrt(1 + (m*m) ) ) )*0.5
        
                
            print("alo", x_new, y_new)
        
            
            """
            """
            
            d_old = util.euclidean_distance(self.simulator.depot.coords, d)
            
            d_new = util.euclidean_distance(self.simulator.depot.coords, hello_packet.next_target)
            
            
            
            exp_position = hello_packet.cur_pos  # without estimation, a simple geographic approach
            exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.coords)
            """
            
            import math
            
            final = hello_packet.next_target
            
            coords = hello_packet.cur_pos
            
            ang = math.atan2(final[1]-coords[1], final[0]-coords[0])
            
            angle = math.degrees(ang)
            
            if (angle < 0):
                
                angle = 360 + angle
            
            
            
            old_x, old_y = coords[0], coords[1]
            angle = float(angle)
            # Compute the change in position
            
            """
            
            speed = hello_packet.speed
            delta_y = speed * math.cos(math.radians(angle))
            delta_x = speed * math.sin(math.radians(angle))
            # Add that to the existing position
            new_x = old_x + delta_x 
            new_y = old_y + delta_y 
            """
            
            final_drone = self.drone.next_target()
            
            coords_drone = self.drone.coords
            
            ang_drone = math.atan2(final_drone[1]-coords_drone[1], final_drone[0]-coords_drone[0])
            
            angle_drone = math.degrees(ang_drone)
            
            if (angle_drone < 0):
                
                angle_drone = 360 + angle_drone
            
            
            
            angle_drone = float(angle_drone)
            
            
            """
            if (angle > 180):
                print("who am i:", self.drone)
                print("\tmine cur_pos:", self.drone.coords)
                print("\t mine Next:", self.drone.next_target())
                
                print("\t mine angle: ", angle_drone)
                print("i have this neighbour: ", drone_istance)
                print("with this characteristics\n\t Cur_pos: ", hello_packet.cur_pos)
                print("\t Next:", hello_packet.next_target)
                
                print("\t angle: ", angle)
                #print(new_x, new_y)
                
                #if ()
                
                "input()"
            """  
            exp_position = hello_packet.cur_pos  # without estimation, a simple geographic approach
            exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.coords)
            
            
            time_taken = exp_distance / hello_packet.speed
            
            
            
            #if (angle < 180 and angle_drone > 180):
                
             #   continue
            
            if exp_distance < best_drone_distance_from_depot:
                best_drone_distance_from_depot = exp_distance
                best_drone = drone_istance
                time_taken_best = best_drone_distance_from_depot / hello_packet.speed
                
                
            elif (best_drone != None and exp_distance >= best_drone_distance_from_depot and best_drone.next_target() == hello_packet.next_target):
                
                if (time_taken < time_taken_best):
                    
                    best_drone_distance_from_depot = exp_distance
                    best_drone = drone_istance
                    time_taken_best = best_drone_distance_from_depot / hello_packet.speed
            
            

            
            """
            if exp_distance < best_drone_distance_from_depot or hello_packet.speed > self.drone.speed:
                best_drone_distance_from_depot = exp_distance
                best_drone = drone_istance
            
            """


     

        return best_drone

        # Only if you need --> several features:
        # cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
        #                                                width_area=self.simulator.env_width,
        #                                                x_pos=self.drone.coords[0],  # e.g. 1500
        #                                                y_pos=self.drone.coords[1])[0]  # e.g. 500
        # print(cell_index)
        action = None

        # self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
        # self.drone.residual_energy (that tells us when I'll come back to the depot).
        #  .....

        # Store your current action --- you can add several stuff if needed to take a reward later
        self.taken_actions[pkd.event_ref.identifier] = (action)

        return None  # here you should return a drone object!

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass
