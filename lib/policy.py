import numpy as np


class Policy:
    """


    """
    def __init__(self, state, epsilon, Q_value):
        self.state = state
        self.epsilon = epsilon
        self.Q_value = Q_value

    def policy(self):
        action_forward = 0
        action_backward = 1
        action_left = 2
        action_right = 3
        action_ne = 4
        action_se = 5
        action_nw = 6
        action_sw = 7

        actions = [action_forward, action_backward,
                   action_left, action_right, action_ne,
                   action_se, action_nw, action_sw]
        n_actions = len(actions)

        int_state1 = int(self.state[0])
        int_state2 = int(self.state[1])
        if np.random.binomial(1, self.epsilon) == 1:
            action = np.random.choice(actions)
        else:
            values_ = self.Q_value[int_state1, int_state2, :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        return action
