"""
PER CHI LEGGERÀ: NON SO COME IMPORTARE DA SRC.UTILITIES.CONFIG IL NUMERO 
DI DRONI CHE SONO INIZIALIZZATI LÀ
"""

import numpy as np
import math
from numpy.core.defchararray import array 
from numpy.core.numeric import NaN
from numpy.lib.type_check import real_if_close
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt


#TO FIX TO OBTAIN THE NUMBER OF DRONES (COSTANT OF CONFIG.PY)
import src.utilities.config as config

#GLOBAL THINGS

#import the library for random values
import random

#each element indicates scores calculated for each drone
q = []

#each element indicates attempts executed for each drone
n = []

""" ##!!


FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY"""

#create lists
for _ in range(5):
    
    #we assume to have optimistic initial value as strategy for action selection
    q.append(2)
    
    #initially we have zero attempts for each element
    n.append(0)

"""END FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
    
"""   
    
#seed for random values, just to have consistence on values 
#TODO
#eliminate after a while
random.seed(2)

#epsilon must be smaller and it represents probability for epsilon-greedy
min_epsilon = 0
max_epsilon = 0.5#0.50

georouting_on_next_step = True

#take a random value between 0,1
epsilon = random.random()

#normalize the random value from min_epsilon to max_epsilon
epsilon = min_epsilon + (epsilon * (max_epsilon - min_epsilon))

#list of yet taken feedback
yet_happened = []


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
        
        
        #if the packet isn't still treated, then we train system for it
        if (id_event not in yet_happened):
        
            #add it to list of visited packet (to avoid duplicates)
            yet_happened.append(id_event)    

            "Doubt: i don't know the utility of this"        
            if id_event in self.taken_actions:
                action = self.taken_actions[id_event]
                del self.taken_actions[id_event]
            "End of doubt"
                
            #if the packet is arrived isn't more valid
            if (outcome == -1):
                
                #we obtain a small reward 
                R = 0.5
            
            #if the packet arrived
            else:
                          
                #opposite of delay
                temp = 2000 - delay
                
                #we obtain a linear reward based on the delay --> 
                #more the delay and value more close value to 1... 
                #less the delay and value more close to 2 
                temp = (temp - 0) / (2000 - 0)
                
                #take the reward
                R = 1 + temp 
                
            #add attempts for the starting drone that has initially the packet
            #TODO
            #maybe also for all the path of packets to incentive themù

            #Error with 10 drones
            n[drone.identifier] += 1
            
             ##!!
            
            ##!!FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
            
            #calculate incrementally the reward
            q[drone.identifier] = q[drone.identifier] + ((1/(n[drone.identifier]))*(R - q[drone.identifier])) 
            
            ##!!END FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
            
             
            
            """
            
            try:
                cc
                q[drone.identifier] = q[drone.identifier] + ((1/(n[drone.identifier]))*(R - q[drone.identifier]))
                
            except Exception as e:
                
                print(e)
                print("alo")
                input()
            
            """
            
    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        """ arg min score  -> geographical approach, take the drone closest to the depot """
        
        #we take our distance from the depot
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
        best_drone_distance_from_depot = self.compute_distance_to_trajectory_s()
        #initially drone closest is us (we take to the depot the
        #packet without any help)
        best_drone = None
        
        
        """ ##!!
        
        "REINFORCEMENT LEARNING RANDOM"
        max_action = None
        list_neightbours = []
        for hello_packet, drone_istance in opt_neighbors:
            
            list_neightbours.append(drone_istance)
        
        max_action = random.choice(list_neightbours)
        
        return max_action
        "END REINFORCEMENT LEARNING RANDOM"
        
        """
        
        '''
        #generate a random value between 0 and 1
        rand = random.random()
        
        #with 1 - epsilon probability we choose the greedy approach
        if (rand < (1-epsilon)):
            
            #take the maximum value of q
            max_q = q[self.drone.identifier]
            
            #initially the packet remains with us
            max_action = None
            
            #loop for every neighbors
            for hello_packet, drone_istance in opt_neighbors:
                
                #if we have a more reliable node
                if (q[drone_istance.identifier] > max_q):
                    
                    #select its best value for q function
                    max_q = q[drone_istance.identifier]
                    
                    #select it
                    max_action = drone_istance
                    
            
        #with epsilon probability we choose the random approach
        else:
            
            #create the list of neighbors
            list_neighbors = []
            
            #loop for every drones
            for hello_packet, drone_istance in opt_neighbors:
                
                #append istances of the drones
                list_neighbors.append(drone_istance)
                
            #select one drone randomly
            max_action = random.choice(list_neighbors)
            
        #return this random drone
        #return max_action
         '''               
        
        #HERE BEGIN THE GEOGRAPHICAL ROUTING, BUT WE DON'T ARRIVE UNTIL HERE
        #TODO
        #A GOOD POSSIBLE IDEA IS TO COMBINE THIS REINFORCEMENT LEARNING WITH GEOGRAPHICAL ROUTING
        
        #we take all hello packets and all istances of drones
        for hello_packet, drone_istance in opt_neighbors:            
        
            
            import math
            
            
            
            
           
         #   exp_position = hello_packet.cur_pos  # without estimation, a simple geographic approach
         #   exp_position = self.compute_cross_point(hello_packet)

            exp_position = self.compute_extimed_position(hello_packet)
          #  exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.coords)

            exp_distance = self.compute_distance_to_trajectory(hello_packet)
            

            
           # time_taken = exp_distance / hello_packet.speed
            
            
            
            #if (angle < 180 and angle_drone > 180):
                
             #   continue
            

            if exp_distance < best_drone_distance_from_depot:
                best_drone_distance_from_depot = exp_distance
                best_drone = drone_istance
              #  time_taken_best = best_drone_distance_from_depot / hello_packet.speed
                


     

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
        
        print("Hello", q,n)
        pass

    def compute_extimed_position(self, hello_packet):
        """ estimate the current position of the drone """

        # get known info about the neighbor drone
        hello_message_time = hello_packet.time_step_creation
        known_position = hello_packet.cur_pos
        known_speed = hello_packet.speed
        known_next_target = hello_packet.next_target

        # compute the time elapsed since the message sent and now
        # elapsed_time in seconds = elapsed_time in steps * step_duration_in_seconds
        elapsed_time = (self.simulator.cur_step - hello_message_time) * self.simulator.time_step_duration  # seconds

        # distance traveled by drone
        distance_traveled = elapsed_time * known_speed

        # direction vector
        a, b = np.asarray(known_position), np.asarray(known_next_target)
        v_ = (b - a) / np.linalg.norm(b - a)

        # compute the expect position
        c = a + (distance_traveled * v_)

        return tuple(c)

#Unused
    def compute_next_position(self, hello_packet):
        hello_packet_time = hello_packet.time_step_creation
        hello_packet_position = hello_packet.cur_pos
        hello_packet_speed = hello_packet.speed
        hello_packet_next_target = hello_packet.next_target        
   
#Unused
    def compute_cross_point(self, hello_packet):

        exp_pos = self.compute_extimed_position(hello_packet)

        hello_packet_speed = hello_packet.speed
        hello_packet_next_target = hello_packet.next_target

        # compute the direction vector
        a, b = np.asarray(exp_pos), np.asarray(hello_packet_next_target)
    #    v = (b - a) / np.linalg.norm(b - a)

        return self.myFunction(a, b, hello_packet_speed , -1)

    def compute_distance_to_trajectory_s(self):
        p1 = np.array([self.drone.coords[0], self.drone.coords[1]])
        p3 = np.array([self.drone.depot.coords[0],self.drone.depot.coords[1]])
        print(type(self.drone))
        p2 = np.array([self.drone.next_target()[0], self.drone.next_target()[1]])

        print(type(p2))

        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

    def compute_distance_to_trajectory(self, hello_packet):

        exp_position = self.compute_extimed_position(hello_packet)

        #MAYBE IT SHOULD BE p1 = np.array([exp_position[0][0], exp_position[0][1]])
        p1 = np.array([exp_position[0], exp_position[1]])
        print(exp_position[0])
        print(exp_position[1])
        print(hello_packet.next_target[0])
        print(self.drone.depot.coords[0])
        p2 = np.array([hello_packet.next_target[0], hello_packet.next_target[1]])
        p3 = np.array([self.drone.depot.coords[0],self.drone.depot.coords[1]])

        print(type(p2))

        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
