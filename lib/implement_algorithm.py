from lib import actionSpace
from lib.policy import Policy
from lib.Environment import Environment
from tqdm import tqdm
import numpy as np


def differential_SARSA(alpha, beta, epsilon, max_steps, store_all_state,
                       Q_value, filament_distance, filament_diameter, loss):
    """
    This an implementation of differential SARSA which uses average reward instead of
    discounted rewards. This algorithm is used because of its suitability in continuing
    tasks.
    """
    # Initialize state (random)
    length_idx = [i for i in range(10, 60)]
    d_idx = [i for i in range(10, 60)]

    lxy_idx = np.random.choice(length_idx)  # random index of l_xy
    dia_idx = np.random.choice(d_idx)  # random index of dia

    # reshape data to 2D arrays to find the random state
    lossVal = loss.reshape(68, 68)
    # loss at random l_xy and d
    loss_val = lossVal[lxy_idx, dia_idx]
    state = [lxy_idx, dia_idx, loss_val]

    avg_reward = 0  # initialize average reward
    # initialization of local variables
    total_reward = 0
    step = 0
    timestep_reward = []
    # take initial action
    follow_policy = Policy(state, epsilon, Q_value)
    action = follow_policy.policy()  # initialize action

    progress_bar = tqdm(total=max_steps)
    # main loop
    while step < max_steps:
        store_all_state.append(state)
        # track time steps
        progress_bar.update()
        # Take action A and observe R, S' from the environment
        environment = Environment(state, action, filament_distance,
                                  filament_diameter, loss, Q_value,
                                  epsilon)
        next_state, reward = environment.env()

        # # update action using the same epsilon-greedy policy, hence "on policy learning"
        follow_policy = Policy(state, epsilon, Q_value)
        next_action = follow_policy.policy()

        # keep track of total reward
        total_reward = total_reward + reward

        state1, state2 = int(state[0]), int(state[1])
        next_state1, next_state2 = int(next_state[0]), int(next_state[1])

        # Update Q table using TD update rule (On policy)
        TD_delta = reward - avg_reward + (
            Q_value[next_state1, next_state2, next_action] -
            Q_value[state1, state2, action])

        avg_reward += beta * TD_delta
        Q_value[state1, state2, action] += alpha * TD_delta

        # update state only, action will be chosen by same policy (hence off policy)
        state = next_state
        action = next_action

        timestep_reward.append(reward)
        # update time step
        step += 1

    progress_bar.close()
    return store_all_state, timestep_reward, \
        filament_distance, filament_diameter, loss, avg_reward, Q_value
