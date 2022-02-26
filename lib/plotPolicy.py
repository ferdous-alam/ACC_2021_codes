import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import statistics
import pandas as pd


def build_dataset():
    # import data ---> Add column header
    col_names = ['filament_distance', 'filament_diameter', 'Loss']
    phononic_data = pd.read_csv('phononic_dataset_new.csv', names=col_names, header=None)
    simulation_data = pd.DataFrame(phononic_data)  # convert into data frame
    filament_distance = simulation_data.iloc[:, 0] + 400
    filament_diameter = simulation_data.iloc[:, 1]
    loss_values = simulation_data.iloc[:, 2]
    loss = loss_values.values.reshape((68, 68))

    return filament_distance, filament_diameter, loss


def plot_data(X1, X2, Y):
    fig = plt.figure(figsize=(12, 10))
    ax = Axes3D(fig)
    ax.plot_trisurf(X1, X2, Y, linewidth=0.1)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    plt.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x_1$', fontsize=20)
    ax.set_ylabel('x_2$', fontsize=20)
    ax.set_zlabel('reward', fontsize=20)
    plt.show()


def trajectory_allstate(all_states):
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()

    filament_distance, filament_diameter, _ = build_dataset()

    lxy_2D = filament_distance.values.reshape(68, 68)
    dia_2D = filament_diameter.values.reshape(68, 68)

    all_states = np.array(all_states)
    p, q, r = all_states.shape
    for i in range(len(all_states)):
        single_trajectory = all_states[i]
        x = single_trajectory[:, 0]
        y = single_trajectory[:, 1]
        agentPosition_x = lxy_2D[x.astype(int),
                                 x.astype(int)]
        agentPosition_y = dia_2D[y.astype(int),
                                 y.astype(int)]

        plt.plot(agentPosition_x, agentPosition_y, lw=0.5)
        # plt.scatter(agentPosition_x[-1], agentPosition_y[-1], s=250, marker='o')
    ax.set_xlim(lxy_2D[0, 0], lxy_2D[-1, -1])
    ax.set_ylim(dia_2D[0, 0], dia_2D[-1, -1])

    plt.xlabel(r'$l_{xy} \ \ \mu m$', fontsize=28)
    plt.ylabel(r'$d \ \ \mu m$', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(r'Trajectory of states', fontsize=28)
    # save figure as pdf
    # plt.savefig("figures/state_trajectory.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


def agentPos(store_all_state):
    """
    Plot the agent position during training
    :param store_all_state:
    :param loss:
    :return:
    """
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    filament_distance, filament_diameter, _ = build_dataset()

    lxy_2D = filament_distance.values.reshape(68, 68)
    dia_2D = filament_diameter.values.reshape(68, 68)
    # loss_2D = loss.reshape(68, 68)
    # loss_2D = 80.0 - loss_2D
    agentPosition = np.array(store_all_state)
    agentPosition_x = lxy_2D[agentPosition[:, 0].astype(int), agentPosition[:, 0].astype(int)]
    agentPosition_y = dia_2D[agentPosition[:, 1].astype(int), agentPosition[:, 1].astype(int)]

    # cs = plt.contourf(lxy_2D, dia_2D, loss_2D, cmap=plt.get_cmap('viridis_r'))
    plt.plot(agentPosition_x, agentPosition_y, lw=2.0)
    plt.scatter(agentPosition_x[-1], agentPosition_y[-1], s=500, marker='*')
    # cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel(r'reward', fontsize=28)
    # cbar.ax.tick_params(labelsize=20)
    plt.xlabel(r'$l_{xy} \ \ \mu m$', fontsize=28)
    plt.ylabel(r'$d \ \ \mu m$', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(r'Training agent in simulation', fontsize=28)
    # save figure as pdf
    # plt.savefig("figures/trainingPos_2.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


def optAgentPos(store_all_state, storeOptState, loss):
    """
    Plot temporal abstraction showing temporally extended action
    :param store_all_state:
    :param storeOptState:
    :param loss:
    :return:
    """
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    filament_distance, filament_diameter, _ = build_dataset()

    lxy_2D = filament_distance.values.reshape(68, 68)
    dia_2D = filament_diameter.values.reshape(68, 68)
    loss_2D = loss.reshape(68, 68)
    loss_2D = 80.0 - loss_2D

    agentPosition = np.array(store_all_state)
    agentPosition_x = lxy_2D[agentPosition[:, 0].astype(int), agentPosition[:, 0].astype(int)]
    agentPosition_y = dia_2D[agentPosition[:, 1].astype(int), agentPosition[:, 1].astype(int)]
    agentOptPosition = np.array(storeOptState)
    agentOptPosition_x = lxy_2D[agentOptPosition[:, 0].astype(int),
                                agentOptPosition[:, 0].astype(int)]
    agentOptPosition_y = dia_2D[agentOptPosition[:, 1].astype(int),
                                agentOptPosition[:, 1].astype(int)]

    cs = plt.contourf(lxy_2D, dia_2D, loss_2D, cmap=plt.get_cmap('viridis_r'))
    plt.plot(agentOptPosition_x, agentOptPosition_y, color='red', marker='o',
             linewidth=3.0, markersize=7.5, label='temporally extended action')
    plt.scatter(agentPosition_x, agentPosition_y, s=7.5, c='white', marker='o', label='primitive action')
    # identify start and end positions
    plt.scatter(agentOptPosition_x[-1], agentOptPosition_y[-1], s=200, c='red', marker='s')
    plt.scatter(agentOptPosition_x[0], agentOptPosition_y[0], s=200, c='red', marker='o')
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel(r'reward', fontsize=28)
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', prop={'size': 22})
    # plt.title(r'Temporal abstraction', fontsize=28)
    # plt.savefig("figures/temporal_abstraction_2.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


def optAllAgents(optstate_list, loss):
    """
    Plot outcome of 100 experiments showing optimum
    position of each one
    :param optstate_list:
    :param loss:
    :return:
    """
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    filament_distance, filament_diameter, _ = build_dataset()

    lxy_2D = filament_distance.values.reshape(68, 68)
    dia_2D = filament_diameter.values.reshape(68, 68)
    loss_2D = loss.reshape(68, 68)
    loss_2D = 80.0 - loss_2D
    # plot agent final positions
    cs = plt.contourf(lxy_2D, dia_2D, loss_2D, cmap=plt.get_cmap('viridis_r'))
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel(r'reward', fontsize=28)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title(r'Learning outcome of 100 experiments', fontsize=28)
    for i in range(len(optstate_list)):
        agentPosition_x = lxy_2D[optstate_list[i][0].astype(int), optstate_list[i][0].astype(int)]
        agentPosition_y = dia_2D[optstate_list[i][1].astype(int), optstate_list[i][1].astype(int)]
        plt.scatter(agentPosition_x, agentPosition_y, s=250, c='white', marker='*')
    plt.scatter(agentPosition_x, agentPosition_y, s=250, c='white',
                marker='*', label=r'final state, $\mathbf{x}_T$')
    plt.legend(fontsize=25, loc='best')

    # save figure as pdf
    # plt.savefig("figures/optAllAgent_50.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


def plot_success_rate(tot_success, param_val, parameter):
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(param_val, tot_success, marker='o', markersize=10, lw=2.0)
    plt.xlabel(r'${}$'.format(parameter), fontsize=28)
    plt.ylabel(r'$success$', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title(r'Training agent in simulation', fontsize=28)
    # save figure as pdf
    # plt.savefig("figures/success_rate_{}.pdf".format(parameter), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_mean_rewards(total_trials, rewards_cache, exp_num):

    rewards_array = np.array(rewards_cache)
    rewards_mean = np.mean(rewards_array, axis=0)
    rewards_std = np.std(rewards_array, axis=0)
    x = [i for i in range(exp_num)]
    figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(x, rewards_mean, 'k', lw=2.0)
    plt.fill_between(x, rewards_mean - rewards_std, rewards_mean + rewards_std,
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    # plt.errorbar(x, rewards_mean, yerr=rewards_std, fmt='-o', capsize=6.0, lw=2.0)
    plt.xlabel(r'budget, $T$', fontsize=28)
    plt.ylabel(r'reward', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # save figure as pdf
    # plt.savefig("figures/mean_reward3.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


def plot_subgoal(first_visit, optstate, max_visit_idx):
    filament_distance, filament_diameter, _ = build_dataset()
    filament_distance = filament_distance.values.reshape(68, 68)[:, 0]
    filament_diameter = filament_diameter.values.reshape(68, 68)[0, :]
    subgoal_state = first_visit[max_visit_idx]
    Y = np.zeros(len(first_visit))
    X1 = np.zeros(len(first_visit))
    X2 = np.zeros(len(first_visit))

    for i in range(len(first_visit)):
        Y[i] = first_visit[i][3]
        lxy_idx = first_visit[i][0]
        X1[i] = filament_distance[lxy_idx]
        dia_idx = first_visit[i][1]
        X2[i] = filament_diameter[dia_idx]
    X1_opt, X2_opt = filament_distance[subgoal_state[0]], filament_diameter[subgoal_state[1]]
    Y_opt = subgoal_state[3]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(12, 10))
    ax = Axes3D(fig)

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel(r'$x_1$', fontsize=30)
    ax.set_ylabel(r'$x_2$', fontsize=30)
    ax.set_zlabel(r'visit count', fontsize=30)
    ax.plot(X1, X2, Y, lw=5.0, c='r', label=r'$\tau^*_{\pi^*}$')
    ax.scatter(X1[0], X2[0], Y[0], s=500, marker='o',
               color='blue', label='trajectory start')
    ax.scatter(X1[-1], X2[-1], Y[-1], s=500, marker='s',
               color='blue', label='trajectory end')
    ax.scatter(X1_opt, X2_opt, Y_opt, s=500, marker='*',
               color='blue', label='max visit')
    # plot states
    # ax.scatter(X1_opt, X2_opt, 0, s=500, marker='*',
    #            color='blue', alpha=0.15, label='subgoal state')

    option_x = [[X1[0], X1_opt, X1[-1]]]
    option_y = [[X2[0], X2_opt, X2[-1]]]
    option_z = np.zeros(len(option_x))

    ax.plot(option_x, option_y, option_z, marker='.', lw=5.0, c='y')

    # save as physics model or stochastic model
    ax.legend(fontsize=25, loc='best')
    plt.savefig("figures/visit_count{}.pdf".format(np.random.choice(100)), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_horizon_rewards(store_all_state, optState, base_reward,
                         loss):
    figure(figsize=(12, 8))
    ax = plt.gca()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    axins = inset_axes(ax, width="35%", height="45%", loc=4, borderpad=0)
    x = [i for i in range(len(store_all_state))]
    y = np.zeros(len(store_all_state))
    for i in range(len(store_all_state)):
        y[i] = store_all_state[i][2]
    ax.plot(x, base_reward - y, lw=3.0)
    ax.axhline(base_reward - optState[2], color='red', lw=3.0)
    ax.set_xlabel(r'Horizon, $H$', fontsize=28)
    ax.set_ylabel(r'reward', fontsize=28)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    filament_distance, filament_diameter, _ = build_dataset()
    lxy_2D = filament_distance.values.reshape(68, 68)
    dia_2D = filament_diameter.values.reshape(68, 68)
    loss_2D = loss.reshape(68, 68)
    loss_2D = 80.0 - loss_2D
    # plot agent final positions
    cs = axins.contourf(lxy_2D, dia_2D, loss_2D, cmap=plt.get_cmap('viridis_r'))

    # plt.title(r'Learning outcome of 100 experiments', fontsize=28)
    agentPosition_x = lxy_2D[optState[0], optState[0]]
    agentPosition_y = dia_2D[optState[1], optState[1]]
    axins.scatter(agentPosition_x, agentPosition_y, s=500, c='white', marker='*')
    axins.axes.get_xaxis().set_visible(False)
    axins.axes.get_yaxis().set_visible(False)
    plt.show()
