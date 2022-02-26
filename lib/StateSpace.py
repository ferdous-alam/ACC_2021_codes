import numpy as np


class StateSpace:
    """
    This class has the description of all the state-space
    requirements. It can find a new state, restart a state
    if the agent hits the boundary or determine whether the
    state is at the boundary
    """

    def __init__(self, loss, state, xCord, yCord, currentState):
        """

        :param loss:
        :param state:
        :param xCord:
        :param yCord:
        :param currentState:
        """
        self.loss = loss
        self.state = state
        self.xCord = xCord
        self.yCord = yCord
        self.currentState = currentState

    def starting_state(self):
        """
        randomly choose the starting state, this is a one time
        operation only.
        :return: next state
        """
        max_idx = 68  # maximum index number
        edge_margin = 0  # 0.25  # keep 20% margin to avoid starting points at edges

        temp_val = edge_margin * max_idx
        low_range = int(temp_val / 2)
        high_range = max_idx - int(temp_val / 2)
        length_idx = [i for i in range(low_range, high_range)]
        dia_idx = [i for i in range(low_range, high_range)]

        l_xy_idx = np.random.choice(length_idx)  # random index of l_xy
        dia_idx = np.random.choice(dia_idx)  # random index of dia

        # loss at random l_xy and d
        loss_val = self.loss[l_xy_idx, dia_idx]

        # define state
        state = [l_xy_idx, dia_idx, loss_val]
        return state
        # define function to find new state

    def find_new_state(self):
        """
        Check if the current state is at the boundary,
        if not, then choose the next state and make
        the full state, otherwise stay at the previous state
        :return: new state
        """
        # convert float data to integar because index is integar
        l_xy_idx = int(self.xCord)
        dia_idx = int(self.yCord)

        if self.terminal_state() is True:
            new_state = self.state
        else:
            loss_val = self.loss[l_xy_idx, dia_idx]

            # convert to array
            new_state = [l_xy_idx, dia_idx, loss_val]

        return new_state

    def restart_state(self):
        """
        This function defines the agent's behavior when the agent
        hits the boundary of the state space (search domain). It
        picks the next state randomly from a state space with a
        certain margin which helps it not to choose any state near
        the boundary.
        :return: next state
        """
        max_idx = 68  # maximum index number
        length_idx_val = [i for i in range(max_idx)]
        dia_idx_val = [i for i in range(max_idx)]

        i, j, k = self.currentState
        if i >= 67 or i < 0:
            l_xy_idx = np.random.choice(length_idx_val)
            dia_idx = j
        elif j >= 67 or j < 0:
            l_xy_idx = i
            dia_idx = np.random.choice(dia_idx_val)
        else:
            l_xy_idx = i
            dia_idx = j
        # convert to integer for calculating corresponding loss
        l_xy_idx = int(l_xy_idx)
        dia_idx = int(dia_idx)

        # loss at random l_xy and d
        loss_val = self.loss[l_xy_idx, dia_idx]

        # define state
        new_state = [l_xy_idx, dia_idx, loss_val]

        return new_state

    # define terminal state
    def terminal_state(self):
        """
        returns: whether this is a terminal state or not
        """
        if self.state[0] >= 67 or self.state[0] < 0:
            return True
        elif self.state[1] >= 67 or self.state[1] < 0:
            return True
        else:
            return False
