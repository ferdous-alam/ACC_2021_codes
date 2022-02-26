from lib import actionSpace
import matplotlib.pyplot as plt
from lib.policy import Policy
from lib.Environment import Environment
import numpy as np


def onpolicy_SARSA(alpha, beta, epsilon, horizon, store_all_state, state, avg_reward,
                   Q_value, loss):
    """
    This an implementation of differential SARSA which uses average reward instead of
    discounted rewards. This algorithm is used because of its suitability in continuing
    tasks.
    """
    # initialization of local variables
    step = 0
    timestep_reward = []
    store_all_state = []
    # take initial action
    follow_policy = Policy(state, epsilon, Q_value)
    action = follow_policy.policy()  # initialize action
    first_visit = []
    first_visit_states = []
    initState_info = [state[0], state[1], action]
    # main loop
    while step < horizon:
        if state not in first_visit_states:
            cache_state = [state[0], state[1], action, 1]
            first_visit.append(cache_state)
            first_visit_states.append(state)
        else:
            # get the index of the existing current state
            curridx = first_visit_states.index(state)
            cache_state = first_visit[curridx]
            first_visit[curridx] = [cache_state[0], cache_state[1],
                                    cache_state[2], cache_state[3] + 1]

        store_all_state.append(state)

        # Take action A and observe R, S' from the environment
        environment = Environment(state, action, loss)
        next_state, reward = environment.env()
        # update action using the same epsilon-greedy policy, hence "on policy learning"
        follow_policy = Policy(state, epsilon, Q_value)
        next_action = follow_policy.policy()

        state1, state2 = int(state[0]), int(state[1])
        next_state1, next_state2 = int(next_state[0]), int(next_state[1])

        # Update Q table using TD update rule (On policy)
        TD_delta = reward - avg_reward + (
            Q_value[next_state1, next_state2, next_action]
            - Q_value[state1, state2, action])

        avg_reward += beta * TD_delta
        Q_value[state1, state2, action] += alpha * TD_delta

        # update state only, action will be chosen by same policy (hence off policy)
        state = next_state
        action = next_action

        cache_reward = reward - avg_reward
        timestep_reward.append(cache_reward)
        # update time step
        step += 1

    count_visits = [first_visit[i][3] for i in range(0, len(first_visit))]
    max_visit_idx = np.argmax(count_visits)
    # max_visit_count = max(count_visits)    # in case we want to print maximum visit count
    max_visit_state = first_visit_states[max_visit_idx]
    optState = max_visit_state
    optState_info = [first_visit[max_visit_idx][0], first_visit[max_visit_idx][1],
                     first_visit[max_visit_idx][2]]

    # TEMPORARY PLOT FOR DEBUGGING
    # x = [i for i in range(len(store_all_state))]
    # y = np.zeros(len(store_all_state))
    # for i in range(len(store_all_state)):
    #     y[i] = store_all_state[i][2]
    # plt.plot(x, 80-y)
    # plt.show()

    return store_all_state, optState, max_visit_idx, first_visit, optState_info, \
        initState_info, Q_value, timestep_reward, first_visit_states


def offpolicy_Q(alpha, beta, epsilon, horizon, store_all_state, state, avg_reward,
                   Q_value, loss):
    """
    This an implementation of differential Q-learning which uses average reward instead of
    discounted rewards. This algorithm is used because of its suitability in continuing
    tasks.
    """
    # initialization of local variables
    step = 0
    timestep_reward = []
    store_all_state = []
    # take initial action
    follow_policy = Policy(state, epsilon, Q_value)
    action = follow_policy.policy()  # initialize action
    first_visit = []
    first_visit_states = []
    initState_info = [state[0], state[1], action]
    # main loop
    while step < horizon:
        if state not in first_visit_states:
            cache_state = [state[0], state[1], action, 1]
            first_visit.append(cache_state)
            first_visit_states.append(state)
        else:
            # get the index of the existing current state
            curridx = first_visit_states.index(state)
            cache_state = first_visit[curridx]
            first_visit[curridx] = [cache_state[0], cache_state[1],
                                cache_state[2], cache_state[3] + 1]
        store_all_state.append(state)

        # Take action A and observe R, S' from the environment
        environment = Environment(state, action, loss)
        next_state, reward = environment.env()
        # update action using the same epsilon-greedy policy, hence "on policy learning"
        follow_policy = Policy(state, epsilon, Q_value)
        next_action = follow_policy.policy()

        # update state only, action will be chosen by same policy (hence off policy)
        state = next_state
        action = next_action

        cache_reward = reward
        timestep_reward.append(cache_reward)

        # update time step
        step += 1

    count_visits = [first_visit[i][3] for i in range(0, len(first_visit))]
    max_visit_idx = np.argmax(count_visits)
    # max_visit_count = max(count_visits)    # in case we want to print maximum visit count
    max_visit_state = first_visit_states[max_visit_idx]
    optState = max_visit_state
    optState_info = [first_visit[max_visit_idx][0], first_visit[max_visit_idx][1],
                     first_visit[max_visit_idx][2]]

    return store_all_state, optState, max_visit_idx, first_visit, optState_info, \
        initState_info, Q_value, timestep_reward, first_visit_states
