import gym
from gym import spaces
from wns2.basestation.nrbasestation import NRBaseStation
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.userequipment.userequipment import UserEquipment
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import random
import logging
import numpy as np
import copy
import math


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class CACGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def init_env(self, x_lim, y_lim, terr_parm, sat_parm, n_ue, datarate, max_datarate=None, max_symbols=None, service_class=None):
        """init `self.env` as Environment using inputs"""
        self.n_step = 1
        self.env = Environment(x_lim, y_lim, renderer = CustomRenderer())
        self.init_pos = []  # for reset method
        
        # random determine user position
        for i in range(0, n_ue):
            pos = (0, 0, 1)
            
            service_datarate = datarate
            if (service_class is not None):
                class_id = self.class_list[i]
                service_datarate = service_class[class_id][0] # a tuple containing base DR and level DR
            self.datarate = service_datarate

            self.env.add_user(
                UserEquipment(
                    self.env, i, service_datarate, pos, speed = 0, direction = 0, _lambda_c=5, _lambda_d = 15
                ))
            self.init_pos.append(pos)

        for i in range(len(terr_parm)):
            self.env.add_base_station(NRBaseStation(self.env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))
        
        for i in range(len(sat_parm)):
            self.env.add_base_station(
                SatelliteBaseStation(
                    env=self.env,
                    bs_id=len(terr_parm) + i,
                    position=sat_parm[i]["pos"],
                    altitude=sat_parm[i].get("altitude", 1200),  
                    angular_velocity=sat_parm[i].get("angular_velocity", (0, 0)),
                    max_data_rate=max_datarate,
                    max_symbol=max_symbols,  
                ))
        
        self.terr_parm = terr_parm
        self.sat_parm = sat_parm
        

    def __init__(self, x_lim, y_lim, class_list, terr_parm, sat_parm, datarate = 25, service_class=None, n_action=3):
            """
            ### parameters
            - `datarate`: this member value is filled in `CACGymEnv.init_env` with `service_datarate`
            - `service_class`: service class defined in `my_gym.py`
            - `n_action`: number of possible actions in action space

            ### state space
            - `power_level` : `self.current_power` stores current power level 
            (note: should set power level higher, at least 10, to actually allocate resource to multi-users)
            - `avg channel capacity` : calculated by subtracting current allocated capacity from total capacity

            """
            super(CACGymEnv, self).__init__()
            # elapsed step
            self.n_step = 1

            # QoS
            self.n_drop = 0 # drop count
            self.qos_bonus = 0 # increase for each qos level
            
            self.n_ap = len(terr_parm) + len(sat_parm)
            # also known as power control level
            self.n_action = n_action 

            # set limits on states to control state space size
            self.max_power_level = 10
            self.max_connected_user = 100
            self.bs_max_datarate = 10_000
            self.bs_max_symbols = 10_000

            # RL stuff
            self.action_space = spaces.Discrete(self.n_action)
            self.observation_space = spaces.Discrete(((self.n_action+1) ** self.n_ap))
            self.a1 = 0.4

            # map bounadry
            self.x_lim = x_lim
            self.y_lim = y_lim
            
            # same as the `self.connected_ues` in `satellitebasestation.py`
            self.connected_ues = {}
            self.datarate = datarate
            self.class_list = []
            self.n_ue = len(class_list)
            # pick randomly `self.n_next_connecting_ue` users for next step connection
            self.n_next_connecting_ue = 2
            self.service_class = service_class

            self.init_env(x_lim, y_lim, terr_parm, sat_parm, self.n_ue, datarate, 
                          max_datarate=self.bs_max_datarate, max_symbols=self.bs_max_symbols, service_class=service_class)
            
            self.advertised_connections = []

    def observe(self):
        """
        return list of current state for each BS
        """
        bs_obs = []
        total_capacity = self.bs_max_datarate

        for j in range(self.n_ap):
            bs_j = self.env.bs_by_id(j)
            allocated_cap = 0
            ue_allocated_bitrates = bs_j.ue_bitrate_allocation
            for ue in ue_allocated_bitrates:
                allocated_cap += ue_allocated_bitrates[ue]

            # states
            current_power = bs_j.get_power()
            chnl_cap = total_capacity - allocated_cap
            connected_users = len(ue_allocated_bitrates)
            n_drop = self.n_drop // self.n_step
            sat_pos = bs_j.get_position()

            bs_obs.append([current_power, chnl_cap, connected_users, n_drop, sat_pos])

        return bs_obs
    

    def step(self, action):
        """
        given next power level (action), move on to next time slot
        """
        self.n_step += 1

        # we have only one BS to connect...
        select_bs = 0
        
        bs_j = self.env.bs_by_id(select_bs)
        ue_allocated_bitrates = bs_j.ue_bitrate_allocation

        # compute drop amount
        for ue in ue_allocated_bitrates:
            dr_ue = ue_allocated_bitrates[ue]
            dr_desired_ue = self.env.ue_by_id(ue).data_rate
            if (dr_ue == None) or (dr_ue < dr_desired_ue):
                self.n_drop += 1
            else:
                ue_class = self.env.class_list[ue]
                dr_level = self.service_class[ue_class][1]
                # extra reward from more data rate
                self.qos_bonus += (dr_desired_ue - dr_ue) // (dr_level + 1)

        reward = self.a1 * (self.n_drop // self.n_step) + (1 - self.a1) * bs_j.get_power()
        reward *= -1

        # disconnect all UEs that are not wanting to connect
        for ue_id in range(self.n_ue):
            if ue_id not in self.env.connection_advertisement:
                self.env.ue_by_id(ue_id).disconnect()

        # select next ue that will be scheduled (if all the UEs are scheduled yet, fast-forward steps in the environment)
        if len(self.advertised_connections) > 0:
            # make the env go 1 substep forward
            self.env.step(substep = True)
        else:
            # while len(self.advertised_connections) == 0:
            self.env.step()
            self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
            # if a UE is already connected to an AP, skip it and focus only on the unconnected UEs
            for ue_id in self.advertised_connections:
                if self.env.ue_by_id(ue_id).get_current_bs() != None:
                    self.advertised_connections.remove(ue_id)
        
        # random pick next connecting users
        if (self.n_next_connecting_ue <= len(self.advertised_connections)):
            next_ue_ids = random.sample(self.advertised_connections, self.n_next_connecting_ue)
        else:
            next_ue_ids = self.advertised_connections

        for ue in next_ue_ids:
            self.advertised_connections.remove(ue)
            self.env.ue_by_id(ue).connect_bs(select_bs)
        
        # set to next power level
        bs_j.set_power_action(action)

        # after the step(), the user that have to appear in the next state is the next user, not the current user
        observation = self.observe()

        # TODO: determine if satellite reaches boundary
        done = False
        bs_j_pos = bs_j.get_position()
        # if (bs_j_pos reaches boundary):
            # done = True

        # nothing for `info` for now
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.init_env(self.x_lim, self.y_lim, self.terr_parm, self.sat_parm, self.n_ue, self.datarate)
        # self.env.step()
        # self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
        # # step until at least one UE wants to connect
        # while len(self.advertised_connections) == 0:
        #     self.env.step()
        #     self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
        # ue_id = random.choice(self.advertised_connections)
        # self.current_ue_id = ue_id
        # go back 1 time instant, so at the next step() the connection_advertisement list will not change
        observation = self.observe()
        # self.advertised_connections.remove(ue_id)
        return observation

    def render(self, mode='human'):
        return self.env.render()
    
    def close (self):
        return
