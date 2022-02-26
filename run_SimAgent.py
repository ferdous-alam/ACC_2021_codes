# import dependencies
from lib import implement_algorithm
from lib import plotPolicy
from lib import build_data
import numpy as np


def run_SimAgent(exp_num, max_steps, alpha, beta, epsilon):
    # -----------------------------------
    # Actual learning environment
    # gaussian process parameters
    # -----------------------------------
    length_scale = 52.0
    sigma_f = 50.0
    sigma_y = 1.0  # sigma_y maintains the stochasticity of the data

    # ----------------------------------------------
    # Initialize Q-table
    # ----------------------------------------------
    n_state1 = 68
    n_state2 = 68
    n_actions = 8
    Q_value = np.zeros((n_state1, n_state2, n_actions))
    store_all_state = []
    # Build dataset from physics based model
    data, filament_distance, filament_diameter, loss, mu = build_data.dataset(
        length_scale, sigma_f, sigma_y)

    # ----------------------------------------------
    # begin experiment
    # ----------------------------------------------
    for num in range(exp_num):
        print(f"experiment no # {num + 1}")

        store_all_state, timestep_reward, filament_distance, filament_diameter, loss_sim, avg_reward, Q_value = \
            implement_algorithm.differential_SARSA(alpha, beta, epsilon, max_steps,
                                                   store_all_state, Q_value, filament_distance,
                                                   filament_diameter, loss)

    # ---------- plot agent policy -------------------
    plotPolicy.agentPos(store_all_state, filament_distance, filament_diameter, loss)

    return Q_value, loss_sim, avg_reward


# run sim_agent to update Q-table with simulation data
# ---------------------------
# sim_agent algorithm (SARSA) hyperparameters
# ---------------------------
sim_alpha = 0.5
sim_beta = 0.5
sim_epsilon = 0.25

sim_max_steps = 10000000  # maximum time steps in each experiment
sim_exp_num = 1  # number of experiments
Q_value, loss_sim, avg_reward = run_SimAgent(sim_exp_num, sim_max_steps, sim_alpha,
                                             sim_beta, sim_epsilon)
np.save('Q_function_2', Q_value)
np.save('loss_simulation_2', loss_sim)
np.save('avg_reward_2', avg_reward)



