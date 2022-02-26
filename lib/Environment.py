from lib.StateSpace import StateSpace


class Environment:
    """
    Environment takes the current state as the input and gives
    next state and reward as the output, if the agent hits the boundary of the domain
    it starts from a random state within the domain
    """
    def __init__(self, state, action, loss):
        self.state = state
        self.action = action
        self.loss = loss

    def env(self):

        # define step size for moving to the next state
        delta = 1  # np.random.randint(0, 10,1)
        # derive elements of states
        i, j, k = self.state  # elements of the state
        ss = StateSpace(self.loss, self.state,
                        i, j, self.state)
        if ss.terminal_state() is True:
            next_state = ss.starting_state()
            reward = 0
        else:
            # action forward (l_xy_idx + 1)
            if self.action == 0:
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

            # action backward (l_xy_idx - 1)
            elif self.action == 1:
                ss = StateSpace(self.loss, self.state,
                                i-delta, j, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

            # action left (dia_idx - 1)
            elif self.action == 2:
                ss = StateSpace(self.loss, self.state,
                                i, j-delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

                    # action right (dia_idx + 1)
            elif self.action == 3:
                ss = StateSpace(self.loss, self.state,
                                i, j+delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

                    # action northeast
            elif self.action == 4:
                ss = StateSpace(self.loss, self.state,
                                i-delta, j+delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

            # action south east
            elif self.action == 5:
                ss = StateSpace(self.loss, self.state,
                                i+delta, j+delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

            # action north west
            elif self.action == 6:
                ss = StateSpace(self.loss, self.state,
                                i-delta, j-delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)

            # action south west
            elif self.action == 7:
                ss = StateSpace(self.loss, self.state,
                                i+delta, j-delta, self.state)
                next_state = ss.find_new_state()
                if ss.terminal_state() is True:
                    next_state = ss.starting_state()
                    reward = 0
                else:
                    next_state = next_state
                    reward = self.get_reward(next_state)
            else:
                assert False

        return next_state, reward

    def get_reward(self, state):
        """

        :param state:
        :return:
        """
        loss_val = state[2]
        # reward obtained by the learning agent
        #     reward = 100 - (((length_val - 840)**2)/(a**2) + ((dia_val - 420)**2)/(b**2))/1e2
        reward_RL = 80.0 - loss_val
        reward = reward_RL

        return reward
