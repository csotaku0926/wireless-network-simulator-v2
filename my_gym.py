from json import load

from pkg_resources import load_entry_point
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.gym.cac_env import CACGymEnv
import logging
import lexicographicqlearning
import signal
import numpy as np

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)

def exit_handler(signum, frame):
    res = input("Ctrl-c was pressed, do you want to save your current model? y/n ")
    if res == "y":
        global learner
        learner.save_model()
        exit(1)
    else: 
        exit(1)

signal.signal(signal.SIGINT, exit_handler)

def evaluate_model(learner, env:CACGymEnv, terr_parm:list, sat_parm:list, quantization:int):
    """
    load model and evaluates

    - `learner`: custom Q table learner. e.g., `lexicographicqlearning.LexicographicQTableLearner`
    - `env`: predefined gym env, e.g., `CACGymEnv`
    - `terr_parm`: list containing terrestrial BS parameters
    - `sat_parm`: satellite BS parameters
    - `quantization`: state space quan.
    """
    #learner.train(train_episodes=100000)
    #learner.save_model()

    learner.load_model("CAC_Env", path="saved_models/100UE_50mbps_5BS_100000_1000/")
    print("Model loaded")
    LQL_rewards = learner.test(test_episodes=1000)
    print("Model tested")

    LL_rewards = ([], [])

    for i in range(1000):
        curr_state = env.reset()
        total_reward = 0
        total_constraint_reward = np.zeros(3)

        for j in range(1000):
            load_levels = np.zeros(len(terr_parm)+len(sat_parm))
            reminder = curr_state
            print(curr_state)

            for k in range(len(load_levels)):
                load_levels[k] = reminder % quantization
                reminder = reminder // quantization

            print(f"Load Level: {load_levels}")
            action = np.argmin(load_levels)
            print(f"Action chosen: {action}")

            new_state, reward, done, info = env.step(action+1)
            curr_state = new_state

            for _ in range(len(info)):
                reward_constr = info[_]

                if reward_constr == -1:
                    reward_constr = 0

                total_constraint_reward[_] += reward_constr

            total_reward += reward

        LL_rewards[0].append(total_reward)
        LL_rewards[1].append(total_constraint_reward)

    np.save("LQL_rewards", LQL_rewards)
    np.save("LL_rewards", LL_rewards)

    print(np.mean(LQL_rewards[0]))
    print(np.mean(LL_rewards[0]))


"""    
- `required_data_rate`: base level of required data rate (in bps)
- `qos_level_data_rate`: every rate of this, obtain another qos level (in bps)
"""
SERVICE_CLASS = [
    (100_000, 100_000), # BUD traffic
    (35_000_000, 10_000_000), # Non real-time video
    (2_000_000, 10_000_000), # BAD traffic
    (0, 0), # No service
]

def determine_service():
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

"""
TODO
1. get users in satellite coverage
2. pick random N users to make request, N = covered area * 0.1
3. each picked user determine its requested service (using above prob.)
    - each service has specific requirements : data rate 
---

power -> signal strength -> data rate 
C = B * log(1 + SNR)

connection (table XVI) 0.1 UE / m^2

time unit?
"""
def main():
    """main function & defintion"""
    # how big is the map
    x_lim = 1000
    y_lim = 600

    # TODO: count the covered UE under satellite
    # how many user making request
    n_ue = 100 

    # TODO: UE's positions (waiting for satellite coverage model)
    # ue_positions = {0:(x_0, y_0), 1:(x_1, y_1), ...}

    # their requested services
    class_list = []
    for _ in range(n_ue):
        class_id = determine_service()
        class_list.append(class_id)

    quantization = 6

    # we ignore terrestrial bs
    '''terr_parm = [{"pos": (500, 500, 30),
        "freq": 800,
        "numerology": 1, 
        "power": 20,
        "gain": 16,
        "loss": 3,
        "bandwidth": 20,
        "max_bitrate": 1000},
        

        #BS2
        {"pos": (250, 300, 30),
        "freq": 1700,
        "numerology": 1, 
        "power": 20,
        "gain": 16,
        "loss": 3,
        "bandwidth": 40,
        "max_bitrate": 1000},

        #BS3
        {"pos": (500, 125, 30),
        "freq": 1900,
        "numerology": 1, 
        "power": 20,
        "gain": 16,
        "loss": 3,
        "bandwidth": 40,
        #15
        "max_bitrate": 1000},

        #BS4
        {"pos": (750, 300, 30),
        "freq": 2000,
        "numerology": 1, 
        "power": 20,
        "gain": 16,
        "loss": 3,
        "bandwidth": 25,
        "max_bitrate": 1000}
    ] '''
    terr_parm = []

    # satellite BS parameters
    sat_parm = [{"pos": (250, 500, 786000)}, {"pos": (50, 200, 35786000)}]
    
    # define environment
    env = CACGymEnv(x_lim, y_lim, class_list, terr_parm, sat_parm, datarate = 50, quantization=quantization, service_class=SERVICE_CLASS)
    ue_0 = env.env.ue_list[0]
    # power action goes here
    action = 1
    env.env.bs_list[0].set_power_action(action)
    actual_dr = ue_0.connect_bs(0)
    print("set power to 1:", actual_dr)

    action = 2
    env.env.bs_list[0].set_power_action(action)
    actual_dr = ue_0.connect_bs(0)
    print("set power to 2:", actual_dr)

    # define my learner
    # learner = lexicographicqlearning.LexicographicQTableLearner(env, "CAC_Env", [0.075, 0.10, 0.15])

if __name__ == '__main__':
    main()