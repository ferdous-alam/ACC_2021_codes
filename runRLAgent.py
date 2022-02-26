import numpy as np
from lib import build_data
from lib import plotPolicy
from lib import implement_real_system
from tqdm import tqdm


def real_system_learning(init_states, base_reward, alpha, beta, H, total_trials,
                         exp_num, avg_reward):
    progress_bar = tqdm(total=total_trials)
    optstate_list = []
    for trial in range(total_trials):
        progress_bar.update()
        store_all_state = []
        storeOptState = []
        Q_value_sim = Q_value_org
        # ------------------------------------------------
        # # Initialize state (random)
        # length_idx, d_idx = [i for i in range(0, 68)], [i for i in range(0, 68)]
        # lxy_idx, dia_idx = np.random.choice(length_idx), np.random.choice(d_idx)
        # state = [lxy_idx, dia_idx, loss_model[lxy_idx, dia_idx]]
        state = init_states[trial]
        for num in range(exp_num):
            store_all_state, optState, max_visit_idx, first_visit, optState_info, \
            initState_info, Q_value, timestep_reward, first_visit_states =\
                implement_real_system.onpolicy_SARSA(
                    alpha, beta, epsilon, H, store_all_state, state, avg_reward,
                    Q_value_sim, loss_model)

            state = optState
            storeOptState.append(optState)

            # Update Q table using TD update rule (On policy)
            loss_stochastic = np.load('stochastic_models/loss_stochastic{}.npy'.format(num))
            loss_val = loss_stochastic[int(optState[0]), int(optState[1])]
            reward = base_reward - loss_val
            optQ_value = Q_value[optState_info[0], optState_info[1], optState_info[2]]
            initQ_value = Q_value[initState_info[0], initState_info[1], initState_info[2]]
            TD_delta = np.sum(timestep_reward) + reward - avg_reward + (optQ_value - initQ_value)
            # #
            avg_reward += beta * TD_delta
            Q_value[int(optState_info[0]), int(optState_info[1]), optState_info[2]] +=\
                alpha * TD_delta
            # # ---------- plot agent policy -------------------
            # plotPolicy.agentPos(store_all_state, filament_distance, filament_diameter, loss)
            Q_value_sim = Q_value

        optstate_list.append(storeOptState[-1])
        # plotPolicy.optAgentPos(store_all_state, storeOptState, loss_stochastic)
    progress_bar.close()
    # # # # ---------- plot agent policy -------------------
    # plotPolicy.optAllAgents(np.array(optstate_list), loss_model)  # stochastic loss
    plotPolicy.optAllAgents(np.array(optstate_list), loss_stochastic)  # model loss

    val = np.array([optstate_list[i][2] for i in range(len(optstate_list))]) < 25.0
    l_val = []
    for i in range(len(val)):
        if val[i]:
            temp = 1
            l_val.append(temp)
    print("Optimum position obtained # {}/{}".format(
        len(l_val), total_trials))
    return len(l_val)


# ------------------------------------------
# import previously saved numpy data
Q_value_org = np.load('Q_function_2.npy')
loss_model = np.load('loss_simulation_2.npy')
avg_reward = np.load('avg_reward_2.npy')
# loss_stochastic = loss_stochastic.reshape(68, 68)`
loss_model = loss_model.reshape(68, 68)
# -----------------------------------------------
# Hyperparameters
# -----------------------------------------------
base_reward = 80.0
alpha = 0.5
beta = 0.5
epsilon = 0.05
# H = [5, 10, 25, 50, 100]  # horizon length
H = 100
total_trials = 1
# exp_num = [5, 10, 25, 50, 100]
exp_num = 25
# --------------------------------------------
# Initialize 100 states (random)
init_states = []
for i in range(1):
    length_idx, d_idx = [i for i in range(0, 68)], [i for i in range(0, 68)]
    lxy_idx, dia_idx = np.random.choice(length_idx), np.random.choice(d_idx)
    state = [lxy_idx, dia_idx, loss_model[lxy_idx, dia_idx]]
    init_states.append(state)

tot_success = []
param_val = []
for i in range(1):
    success = real_system_learning(init_states, base_reward, alpha, beta, H,
                                   total_trials, exp_num, avg_reward)
    success_rate = success / 100
    tot_success.append(success_rate)
    param_val.append(H)

# plotPolicy.plot_success_rate(tot_success, param_val, parameter)
