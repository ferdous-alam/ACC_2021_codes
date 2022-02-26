import numpy as np
from lib import build_data
import random
from lib import plotPolicy
from lib import implement_real_system
from lib import implement_real_system_temp
from tqdm import tqdm
import matplotlib.pyplot as plt


# ------------------------------------------
# import previously saved numpy data
Q_value_org = np.load('Q_function_2.npy')
loss_model = np.load('loss_simulation_2.npy')
avg_reward = np.load('avg_reward_2.npy')
# loss_stochastic = np.load('loss_stochastic_single.npy')
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
H = 25
total_trials = 1
# exp_num = [5, 10, 25, 50, 100]
exp_num = 1
# --------------------------------------------
# Initialize 100 states (random)
init_states = []
for i in range(total_trials):
    length_idx, d_idx = [i for i in range(0, 68)], [i for i in range(0, 68)]
    lxy_idx, dia_idx = np.random.choice(length_idx), np.random.choice(d_idx)
    state = [lxy_idx, dia_idx, loss_model[lxy_idx, dia_idx]]
    init_states.append(state)

optstate_list = []
model_num = 0
rewards_cache = []
all_states = []
for trial in range(total_trials):
    print('Experiment number #{}'.format(trial+1))
    store_all_state = []
    storeOptState = []
    Q_value_sim = Q_value_org
    # ------------------------------------------------
    # # Initialize state (random)
    # length_idx, d_idx = [i for i in range(0, 68)], [i for i in range(0, 68)]
    # lxy_idx, dia_idx = np.random.choice(length_idx), np.random.choice(d_idx)
    # state = [lxy_idx, dia_idx, loss_model[lxy_idx, dia_idx]]
    state = init_states[trial]
    progress_bar = tqdm(total=exp_num)
    exp_states = []
    for num in range(exp_num):
        progress_bar.update()
        exp_states.append(state)
        epsilon = epsilon/(num+1)
        store_all_state, optState, max_visit_idx, first_visit, optState_info,  initState_info,\
            Q_value, timestep_reward, first_visit_states =\
            implement_real_system.onpolicy_SARSA(
                alpha, beta, epsilon, H, store_all_state, state, avg_reward,
                Q_value_sim, loss_model)
        state = optState   # subgoal state
        opt_state_index = store_all_state.index(optState)
        multitime_rewards = timestep_reward[0:opt_state_index]
        temporal_reward = np.sum(multitime_rewards)
        storeOptState.append(optState)

        # Update Q table using TD update rule (On policy)
        # Build stochastic model

        loss_stochastic = np.load('stochastic_models/loss_stochastic{}.npy'.format(model_num))
        model_num += 1
        loss_val = loss_stochastic[int(optState[0]), int(optState[1])]
        real_system_reward = base_reward - loss_val
        reward = real_system_reward + temporal_reward
        optQ_value = Q_value[optState_info[0], optState_info[1], optState_info[2]]
        initQ_value = Q_value[initState_info[0], initState_info[1], initState_info[2]]
        TD_delta = np.sum(timestep_reward) + reward - avg_reward + (optQ_value - initQ_value)
        # #
        avg_reward += beta * TD_delta
        Q_value[int(optState_info[0]), int(optState_info[1]), optState_info[2]] +=\
            alpha * TD_delta
        # # ---------- plot agent policy -------------------
        # plotPolicy.agentPos(store_all_state)
        Q_value_sim = Q_value
        # plot subgoal
        plotPolicy.plot_subgoal(first_visit, optState, max_visit_idx)
        all_states.append(storeOptState)
        # TEMPORARY PLOT FOR DEBUGGING
        # plotPolicy.plot_horizon_rewards(store_all_state, optState, base_reward,
        #                                 loss_stochastic)

    exp_rewards = np.zeros(exp_num)
    for i in range(exp_num):
        exp_rewards[i] = base_reward - exp_states[i][-1]

    progress_bar.close()
    optstate_list.append(storeOptState[-1])
    # plotPolicy.optAgentPos(store_all_state, storeOptState, loss_stochastic)
    rewards_cache.append(exp_rewards)

# # # # ---------- plot agent policy -------------------
# plotPolicy.optAllAgents(np.array(optstate_list), loss_model)  # model loss
# plotPolicy.optAllAgents(np.array(optstate_list), loss_stochastic)  # stochastic loss
# plot mean rewards with error bar
# plotPolicy.plot_mean_rewards(total_trials, rewards_cache, exp_num)


val = np.array([optstate_list[i][2] for i in range(len(optstate_list))]) < 25.0
l_val = []
for i in range(len(val)):
    if val[i]:
        temp = 1
        l_val.append(temp)
success = len(l_val)
success_rate = success / total_trials
print("Optimum position obtained # {}/{}".format(
    len(l_val), total_trials))
print("S_R = {}".format(success_rate))

# plotPolicy.trajectory_allstate(all_states)