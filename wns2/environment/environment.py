import logging
from wns2.userequipment.multipath_userequipment import MultiPathUserEquipment
from wns2.userequipment.userequipment import UserEquipment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json

MIN_RSRP = -140

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Environment:
    def __init__(self, h, l, sampling_time = 1, renderer = None):
        self.h = h
        self.l = l
        self.ue_list = {}
        self.class_list = {}
        self.connection_advertisement = []
        self.bs_list = {}
        self.sampling_time = sampling_time # in seconds
        self.renderer = renderer
        self.plt_run = 0

        self.drone_aps = []
        
        """    
        - `required_data_rate`: base level of required data rate (in bps)
        - `qos_level_data_rate`: every rate of this, obtain another qos level (in bps)
        """
        self.service_class = [
                    (100, 100), # BUD traffic
                    (350, 100), # Non real-time video
                    (20, 100), # BAD traffic
                    (0, 0), # No service
                ]
        
        self.all_ue_pos = self.get_all_user_pos()

    def get_all_user_pos(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        cart_dict_path = os.path.join(dir_name, "../environment/pop_data/user_cart_dict.json")
        with open(cart_dict_path, 'r') as file:
            ue_data = json.load(file)

        # Combine all UEs from all regions into a single dictionary ##########
        ue_positions = {}
        count = 0
        for _, ue_list in ue_data.items():
            for _, ue_pos in enumerate(ue_list):
                # ue_id = f"{region}_{i}"  # Create a unique UE ID
                ue_id = count
                ue_positions[ue_id] = ue_pos
                count += 1


        return ue_positions
        
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
            if self.ue_list[ue_id].get_current_bs() != None:
                bs = self.ue_list[ue_id].get_current_bs()
                self.ue_list[ue_id].disconnect()
            # del self.ue_list[ue_id]

    def add_base_station(self, bs):
        if bs.get_id() in self.bs_list:
            raise Exception("BS ID mismatch for ID %s", bs.get_id())
        self.bs_list[bs.get_id()] = bs
        if bs.bs_type == "drone":
            self.drone_aps.append(bs.get_id())
        return

    def compute_rsrp(self, ue):
        rsrp = {}
        for bs in self.bs_list:
            rsrp_i = self.bs_list[bs].compute_rsrp(ue)
            if rsrp_i > MIN_RSRP or self.bs_list[bs].get_bs_type() == "sat":
                rsrp[bs] = rsrp_i
        return rsrp
    
    def advertise_connection(self, ue_id):
        self.connection_advertisement.append(ue_id)
        return
    

    def determine_service(self):
        """
        ### Service definition (TC2 from Table XVI)
        - 0.4 Bursty-User Driven (BUD) traffic (web traffic model) --> 100 Kbps  (60 ~ 1500 KB uniform distribution), every 100 Kbps a Qos level
            - e.g. if allocated 200 Kbps for a BUD traffic user, his QoS level is 2
        - 0.4 Non real-time video                                  --> 35 Mbps (with frame rate of 24 fps), every 10 Mbps a QoS level 
        - 0.075 Bursty-Application Driven (BAD) traffic            --> 2 Mbps, every 10 Mbps a QoS level
        - else no request

        (X) file size -- each follows FTP traffic model

        returns:
        - `class id`: id of the service class
        """
        prob = np.random.uniform(0, 1)
        class_id = 0

        # BUD traffic
        if (prob < 0.4):
            class_id = 0
        # Non real-time video
        elif (0.4 <= prob and prob < 0.8):
            class_id = 1
        # BAD traffic
        elif (0.8 <= prob and prob < 0.875):
            class_id = 2
        # No service
        else:
            class_id = 3

        return class_id
    

    def update_user_pos(self, connected_ue:dict):
        """
        Due to satellite movement, we need to update user list

        - `connected_ue`: (ue_id : ue_pos)
        """
        for ue in self.ue_list:
            self.remove_user(ue)
        self.ue_list = {}
        self.class_list = {}

        for new_ue in connected_ue:
            class_id = self.determine_service()
            self.class_list[new_ue] = class_id
            service_datarate = self.service_class[class_id][0]
            
            self.add_user(
                UserEquipment(
                    self, new_ue, service_datarate, connected_ue[new_ue], speed = 0, direction = 0,# _lambda_c=5, _lambda_d = 15
                ))


    def step(self, substep = False): 
        # substep indicates if it is just a substep or it is a complete step
        # if substep, then the connection advertisement list should not be updated
        # nor the UEs should update their internal timers to decide if it is time to connect/disconnect
        # if not substep:
        #     self.connection_advertisement.clear()

        # update satellite position
        for _, bs in self.bs_list.items():
            if hasattr(bs, 'bs_type') and bs.bs_type == "sat":
                bs.update_position()
                self.update_user_pos(bs.connected_ues)
                # print(f"Environment: Satellite {bs_id} Position: {bs.get_position()}")
                # update covered users
                

        for ue in self.ue_list:
            self.ue_list[ue].step(substep)

        for bs in self.bs_list:
            self.bs_list[bs].step()
        '''       
        # disconnect all the users, in order to reconnect them to the right AP and with the right datarate
        for i in range(len(self.ue_list)):
            self.ue_by_id(i).disconnect_all()
        # call here the optimizator, taking the w from users (with data_generation_status and input_data_rate)    
        N = len(self.connection_advertisement)
        M = len(self.bs_list)
        q = np.zeros(N)
        w = np.zeros(N)
        P = np.zeros((N, M))
        for i in range(N):
            ue = self.ue_by_id(self.connection_advertisement[i])
            q[i] = ue.get_queue()
            w[i] = ue.get_current_input_data_rate()
            rsrp = ue.measure_rsrp()
            for bs in rsrp:
                P[i, bs] = 1
        u_final = util.output_datarate_optimization(q, w, N, M, P, self.sampling_time, Q_max = MultiPathUserEquipment.MAX_QUEUE)
        # set the desired data rate for all the users, connecting to the BSs
        for i in range(N):
            ue = self.ue_by_id(self.connection_advertisement[i])
            connection_list = []
            for bs in range(M):
                if u_final[i, bs] > 0 and P[i, bs] == 1:
                    ue.output_data_rate[bs] = u_final[i, bs]
                    connection_list.append(bs)
            ue.connect_bs(connection_list)
        # move the drone AP according to the weighted average of the positions of the UEs connected to it
        for drone_ap in self.drone_aps:
            u_drone = []
            x_ue = []
            y_ue = []
            for i in range(N):
                if P[i, drone_ap] == 1:
                    u_drone.append(u_final[i, drone_ap])
                    pos = self.ue_by_id(i).get_position()
                    x_ue.append(pos[0])
                    y_ue.append(pos[1])
            bs = self.bs_by_id(drone_ap)
            u_0 = 0 # weight of the current position of the drone
            x_drone = bs.get_position()[0]
            y_drone = bs.get_position()[1]
            z_drone = bs.get_position()[2]
            new_drone_pos = util.build_drone_pos_ref(u_drone, x_ue, y_ue, u_0, x_drone, y_drone)
            new_drone_pos = (new_drone_pos[0], new_drone_pos[1], z_drone)
            #print("OLD DRONE POSITION: (%s, %s) ---> NEW DRONE POSITION: (%s, %s)" %(x_drone, y_drone, new_drone_pos[0], new_drone_pos[1]))
            bs.move(new_drone_pos, speed=15)
        '''
        return            

    def render(self):
        if self.renderer != None:
            return self.renderer.render(self)

    def bs_by_id(self, id):
        return self.bs_list[id]
    def ue_by_id(self, id):
        return self.ue_list[id]
    def get_sampling_time(self):
        return self.sampling_time
    def get_x_limit(self):
        return self.l
    def get_y_limit(self):
        return self.h
