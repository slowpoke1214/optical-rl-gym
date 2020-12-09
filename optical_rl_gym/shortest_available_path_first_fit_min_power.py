import gym
from optical_rl_gym.envs.power_aware_rmsa_env import PowerAwareRMSA
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.gnpy_utils import propagation
import numpy as np

import pickle
import logging


load = 250
logging.getLogger('rmsacomplexenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 1000

monitor_files = []
policies = []

with open(f'../examples/topologies/germany50_eon_gnpy_5-paths.h5', 'rb') as f:
# with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))


def shortest_available_path_first_fit_fixed_power(env: PowerAwareRMSA):
    """
    Validation to find shortest available path. Finds the first fit with a given fixed power.

    :param env: The environment of the simulator
    :return: action of iteration (path, spectrum resources, power)
    """
    power = 100   # Fixed power variable for validation method. Gets passed through simulator.
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                min_osnr = path.best_modulation["minimum_osnr"]
                min_power = 21
                for i in range(-45, 20):
                    osnr = np.mean(propagation(i, env.gnpy_network, path.node_list, initial_slot, num_slots, env.eqpt_library))
                    if osnr >= min_osnr and i < min_power:
                        min_power = i
                return [idp, env.topology.graph['modulations'].index(path.best_modulation), initial_slot, min_power]
    return [env.topology.graph['k_paths'], env.topology.graph['modulations'], env.topology.graph['num_spectrum_resources'], power]


env_sap_ff_mp = gym.make('PowerAwareRMSA-v0', **env_args)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_sap_ff_mp, shortest_available_path_first_fit_fixed_power, n_eval_episodes=episodes)
print('SAP-FF-MP:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('Bit rate blocking:', (env_sap_ff_mp.episode_bit_rate_requested - env_sap_ff_mp.episode_bit_rate_provisioned) / env_sap_ff_mp.episode_bit_rate_requested)
print('Request blocking:', (env_sap_ff_mp.episode_services_processed - env_sap_ff_mp.episode_services_accepted) / env_sap_ff_mp.episode_services_processed)
print('Throughput:', env_sap_ff_mp.topology.graph['throughput'])
print('Total power:', 10 * np.log10(env_sap_ff_mp.total_power))
print('Average power:', 10 * np.log10(env_sap_ff_mp.total_power / env_sap_ff_mp.services_accepted))