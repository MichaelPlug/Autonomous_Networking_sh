"""
PER CHI LEGGERÃ€: NON SO COME IMPORTARE DA SRC.UTILITIES.CONFIG IL NUMERO 
DI DRONI CHE SONO INIZIALIZZATI LÃ€
"""

from operator import ne
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
import src.utilities.config as config #try self.simulator.n_drones

#GLOBAL THINGS

#import the library for random values
import random

#each element indicates scores calculated for each drone
q = {}

#each element indicates attempts executed for each drone
n = {}


c = {}
c2 = {}

Reward = {}



"""##!!
##FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
#create dictionaries (because they are more indicated)
for i in range(5):
    
    #we assume to have optimistic initial value as strategy for action selection
    q[i] = 2
    
    #initially we have zero attempts for each element
    n[i] = 0
##END FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
"""   
    
#seed for random values, just to have consistence on values 
#TODO
#eliminate after a while
random.seed(2)

#epsilon must be smaller and it represents probability for epsilon-greedy
second_epsilon = 0.05
min_epsilon = 0.05
max_epsilon = 0.25  

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
                R = -1
            
                #R = -2
            
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
                print("ciao")
                print(R)
                
                #R = 2 * (1 - delay/self.simulator.event_duration)
                
            #add attempts for the starting drone that has initially the packet
            #TODO
            #maybe also for all the path of packets to incentive themÃ¹

            #Error with 10 drones
            
            
            """##!!
            
            ##FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
            
            n[drone.identifier] += 1
            
            
            #calculate incrementally the reward
            q[drone.identifier] = q[drone.identifier] + ((1/(n[drone.identifier]))*(R - q[drone.identifier])) 
            
            ##!!END FOR NORMAL REINFORCEMENT LEARNING AND NORMAL Q ARRAY
            
             
            
            """
            
            """##!!
            
            #METHOD THAT ASSIGN THE REWARD TO THE FIRST INITIAL DRONE
            
            try:
                
                n[(drone.identifier,drone.next_target())] += 1
                q[(drone.identifier,drone.next_target())] = q[(drone.identifier,drone.next_target())] + ((1/(n[(drone.identifier,drone.next_target())]))*(R - q[(drone.identifier,drone.next_target())]))
                
            except Exception as e:
                
                n[(drone.identifier,drone.next_target())] = 1
                q[(drone.identifier,drone.next_target())] = 0
                q[(drone.identifier,drone.next_target())] = q[(drone.identifier,drone.next_target())] + ((1/(n[(drone.identifier,drone.next_target())]))*(R - q[(drone.identifier,drone.next_target())]))
                
            #END OF THE METHOD THAT ASSIGN THE REWARD TO THE FIRST INITIAL DRONE
            
            """
            
            
            """##!!
            
            #METHOD THAT ASSIGN THE REWARD ONLY AT THE LAST DRONE
            try:
                
                n[(self.drone.identifier,self.drone.next_target())] += 1
                q[(self.drone.identifier,self.drone.next_target())] = q[(self.drone.identifier,self.drone.next_target())] + ((1/(n[(self.drone.identifier,self.drone.next_target())]))*(R - q[(self.drone.identifier,self.drone.next_target())]))
                
            except Exception as e:
                
                n[(self.drone.identifier,self.drone.next_target())] = 1
                q[(self.drone.identifier,self.drone.next_target())] = 0
                q[(self.drone.identifier,self.drone.next_target())] = q[(self.drone.identifier,self.drone.next_target())] + ((1/(n[(self.drone.identifier,self.drone.next_target())]))*(R - q[(self.drone.identifier,self.drone.next_target())]))
            
            #END OF THE METHOD THAT ASSIGN THE REWARD ONLY AT THE LAST DRONE
            
            """
            print(delay)
            
            try:
                drone_iden = Reward[id_event]
                
            except Exception as e:
                
                drone_iden = drone
            
            try:
                n[(drone_iden.identifier,drone_iden.next_target())] += 1
                q[(drone_iden.identifier,drone_iden.next_target())] = q[(drone_iden.identifier,drone_iden.next_target())] + ((1/(n[(drone_iden.identifier,drone_iden.next_target())]))*(R - q[(drone_iden.identifier,drone_iden.next_target())]))
            except Exception as e:
                n[(drone_iden.identifier,drone_iden.next_target())] = 1
                q[(drone_iden.identifier,drone_iden.next_target())] = R #0



            
            
            
            
            
    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        """ arg min score  -> geographical approach, take the drone closest to the depot """
        
        #we take our distance from the depot
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
        best_drone_distance_from_depot = self.compute_distance_to_trajectory_s()
        #initially drone closest is us (we take to the depot the
        #packet without any help)
        best_drone = None
        
       
        
        
        
        #generate a random value between 0 and 1
        rand = random.random()
        
        
        ##!!
        
        #GEOROUTING EPSILON-GREEDY NORMALE-CASUALE
        
        
        
        #with 1 - epsilon probability we choose the greedy approach
        if (rand < (1-epsilon)):
            
            
            
            
            try:
            
                #take the maximum value of q
                max_q = q[(self.drone.identifier,self.drone.next_target())]
            
            except Exception as e:
                
                q[(self.drone.identifier,self.drone.next_target())] = 0
                
                max_q = q[(self.drone.identifier,self.drone.next_target())]
            
            
            
            #initially the packet remains with us
            max_action = None
            
            #loop for every neighbors
            for hello_packet, drone_istance in opt_neighbors:
                
                
                try:                
                
                    #if we have a more reliable node
                    if (q[(drone_istance.identifier,hello_packet.next_target)] > max_q):
                    
                        #select its best value for q function
                        max_q = q[(drone_istance.identifier,hello_packet.next_target)]
                    
                        #select it
                        max_action = drone_istance
                
                
                except Exception as e:
                    
                    q[(drone_istance.identifier,hello_packet.next_target)] = 0
                    
                    #if we have a more reliable node
                    if (q[(drone_istance.identifier,hello_packet.next_target)] > max_q):
                    
                        #select its best value for q function
                        max_q = q[(drone_istance.identifier,hello_packet.next_target)]
                    
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
            
        
        Reward[pkd.identifier] = max_action
        
        #return this random drone
        return max_action
        
        
        #FINE GEOROUTING EPSILON-GREEDY NORMALE-CASUALE
    

        

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        
        """
        print("Hello", q)
        print("Alo", n)
        print("Salut", c)
        print(epsilon)
        """
        
        pass